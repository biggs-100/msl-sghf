"""
Test: mejoras a SG-HF ternario.
Compara ternario original vs 2-bit vs fine-tuning vs bloque.
"""
import torch, gc
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    'mistralai/Mistral-7B-v0.3', dtype=torch.float16, device_map='cpu', low_cpu_mem_usage=True)
mlp = model.model.layers[0].mlp
Wg, Wu, Wd = [w.data.float() for w in [mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight]]

torch.manual_seed(42)
hn = torch.randn(1, 32, 4096) * 1.5
out_o = (torch.nn.functional.silu(hn @ Wg.T) * (hn @ Wu.T)) @ Wd.T

def mlp_cos(Wg_q, Wu_q, Wd_q):
    g = torch.nn.functional.silu(hn @ Wg_q.T)
    u = hn @ Wu_q.T
    o = (g * u) @ Wd_q.T
    return (out_o.flatten() * o.flatten()).sum() / (out_o.norm() * o.norm())

# --- 1. Ternario original ---
def ternary_row(W):
    s = W.std(dim=1, keepdim=True)
    mask = torch.where(W.abs() > 0.7 * s, W.sign(), torch.zeros_like(W))
    scale = (W * mask).sum(dim=1, keepdim=True) / (mask.abs().sum(dim=1, keepdim=True) + 1e-8)
    return mask * scale

# --- 2. Ternario con threshold optimizado por fila ---
def ternary_opt_thresh(W):
    # Probar 10 thresholds, elegir el mejor MSE por fila
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    M, N = W.shape
    s = W.std(dim=1)
    best = torch.zeros(M, N)
    for i in range(M):
        best_mse = float('inf')
        for t in thresholds:
            th = t * s[i]
            mask = torch.where(W[i].abs() > th, W[i].sign(), torch.zeros_like(W[i]))
            if mask.abs().sum() == 0:
                continue
            scale = (W[i] * mask).sum() / (mask.abs().sum() + 1e-8)
            W_q = mask * scale
            mse = (W[i] - W_q).pow(2).mean()
            if mse < best_mse:
                best_mse = mse
                best[i] = W_q
    return best

# --- 3. Ternario por bloque bs=128 + threshold optimizado ---
def ternary_block_opt(W, bs=128):
    M, N = W.shape
    nb = N // bs
    Wb = W.reshape(M, nb, bs)
    stds = Wb.std(dim=2, keepdim=True)
    # Probar threshold optimo por bloque
    thresholds = [0.3, 0.5, 0.7, 0.9]
    best_q = torch.zeros(M, N)
    for ib in range(nb):
        blk = Wb[:, ib, :]  # (M, bs)
        s_b = blk.std(dim=1, keepdim=True)
        best_mse = torch.ones(M) * 1e10
        best_out = torch.zeros(M, bs)
        for t in thresholds:
            mask = torch.where(blk.abs() > t * s_b, blk.sign(), torch.zeros_like(blk))
            den = mask.abs().sum(dim=1, keepdim=True) + 1e-8
            scale = (blk * mask).sum(dim=1, keepdim=True) / den
            W_q = mask * scale
            mse = (blk - W_q).pow(2).mean(dim=1)
            better = mse < best_mse
            if better.any():
                # No es eficiente pero es un test
                for j in range(len(better)):
                    if better[j].item():
                        best_mse[j] = mse[j]
                        best_out[j] = W_q[j]
        best_q[:, ib*bs:(ib+1)*bs] = best_out
    return best_q

# --- 4. 2-bit simple: {-3,-1,+1,+3} * scale ---
def bits2_row(W):
    M, N = W.shape
    s = W.std(dim=1, keepdim=True)
    # 3 thresholds: 0, 0.4*s, 0.9*s
    abs_W = W.abs()
    sign = W.sign()
    levels = torch.zeros_like(W)
    # High: > 0.9*s -> +/- 3
    high = abs_W > 0.9 * s
    levels = torch.where(high, sign * 3.0 * s, levels)
    # Mid: > 0.4*s -> +/- 1
    mid = (abs_W > 0.4 * s) & ~high
    levels = torch.where(mid, sign * 1.0 * s, levels)
    return levels

# --- 5. 2-bit optimizado (escala por fila post-cuantizacion) ---
def bits2_opt(W):
    M, N = W.shape
    s = W.std(dim=1, keepdim=True)
    abs_W = W.abs()
    sign = W.sign()
    # Codificar en {0,1,2,3}
    code = torch.zeros_like(W, dtype=torch.long)
    code = torch.where(abs_W > 0.9*s, 3, code)
    code = torch.where((abs_W > 0.4*s) & (code == 0), 2, code)
    code = torch.where((abs_W > 0.1*s) & (code == 0), 1, code)
    # Mapa de valores: {0:0, 1:0.3, 2:1.0, 3:2.0}
    val_map = {0:0, 1:0.3, 2:1.0, 3:2.0}
    values = torch.zeros(M, N)
    for c, v in val_map.items():
        values = values + torch.where(code == c, sign * v * s, torch.zeros_like(values))
    # Optimizar escala por fila via least squares
    num = (W * values).sum(dim=1, keepdim=True)
    den = (values.pow(2)).sum(dim=1, keepdim=True) + 1e-8
    opt_scale = num / den
    return values * opt_scale

# --- 6. Fine-tuning de escalas post-ternario ---
def ternary_ft(W, steps=20, device='cpu'):
    q = torch.where(W.abs() > 0.5*W.std(dim=1, keepdim=True), W.sign(), torch.zeros_like(W))
    scale = torch.nn.Parameter(torch.ones(W.shape[0], 1) * W.std())
    opt = torch.optim.Adam([scale], lr=0.01)
    W_cuda = W.to(device)
    q_cuda = q.to(device)
    scale_cuda = scale.to(device)
    for _ in range(steps):
        W_q = q_cuda * scale_cuda
        loss = (W_cuda - W_q).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (q * scale.detach().cpu()).float()

# --- Ejecutar ---
print(f"Comparacion:")
print(f"{'Metodo':<30} {'Compres':<10} {'MLP COS':<10}")
print("-" * 55)

# Ternario original
cos = mlp_cos(ternary_row(Wg), ternary_row(Wu), ternary_row(Wd)).item()
print(f"{'Ternary row (original)':<30} {16:<10.1f} {cos:<10.4f}")

# Ternario threshold optimizado (solo gate, muy lento para full)
# saltamos este, es muy lento

# Ternario bloque bs=128 + opt threshold
cos = mlp_cos(ternary_block_opt(Wg, 128), ternary_block_opt(Wu, 128), ternary_block_opt(Wd, 128)).item()
print(f"{'Ternary block128+opt':<30} {15.1:<10.1f} {cos:<10.4f}")

# 2-bit row
cos = mlp_cos(bits2_row(Wg), bits2_row(Wu), bits2_row(Wd)).item()
print(f"{'2-bit row (fixed)':<30} {16:<10.1f} {cos:<10.4f}")

# 2-bit optimizado
cos = mlp_cos(bits2_opt(Wg), bits2_opt(Wu), bits2_opt(Wd)).item()
print(f"{'2-bit row (optimized)':<30} {16:<10.1f} {cos:<10.4f}")

# Fine-tuning de escalas (en CPU para simplificar)
cos = mlp_cos(ternary_ft(Wg, 30, 'cuda'), ternary_ft(Wu, 30, 'cuda'), ternary_ft(Wd, 30, 'cuda')).item()
print(f"{'Ternary + FT scales':<30} {16:<10.1f} {cos:<10.4f}")

del model; gc.collect(); torch.cuda.empty_cache()
print("\nOK")
