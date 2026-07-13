"""
Test: cuantizacion ternaria {-1, 0, +1} + escala por fila para gate projection.

Hipotesis: el error independiente por elemento (ruido ternario) da COS mucho mejor
que Kronecker (error estructurado, filas colineales) al mismo nivel de compresion.

Compresion esperada: ~20x
- 1.6 bits por elemento (ternario: -1,0,+1)
- 1 FP16 scale por fila (14336)
- Total: 58.7M * 1.6 bits + 14336 * 16 bits ~ 12 MB vs 235 MB
"""

import torch, torch.nn.functional as F, time, math
from safetensors import safe_open

device = 'cuda'
print(f"Dispositivo: {device}")

CONSOLIDATED = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1\consolidated.safetensors'
with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
    W_g = f.get_tensor('layers.0.feed_forward.w1.weight').float()

W_g = W_g.to(device)
out_f, in_f = W_g.shape
print(f"W_g: {tuple(W_g.shape)}  std={W_g.std():.4f}")

# Input
torch.manual_seed(42)
x = torch.randn(1, 16, 4096, device=device)

with torch.no_grad():
    gate_teacher = F.silu(x @ W_g.T)

# ─── Ternarizacion optima por fila ───
def ternary_quantize(W, threshold_scale=0.7):
    """
    Ternarizacion optima por fila usando TWN (Ternary Weight Networks).
    threshold = threshold_scale * mean(|row|)
    scale = mean(|row_i| for active elements)
    """
    with torch.no_grad():
        # Threshold optimo por fila
        row_abs_mean = W.abs().mean(dim=1, keepdim=True)  # (out_f, 1)
        threshold = threshold_scale * row_abs_mean
        
        # Mascara ternaria
        ternary = torch.where(W > threshold, 1.0, torch.where(W < -threshold, -1.0, 0.0))
        
        # Escala optima por fila: mean(|W_i| para elementos activos)
        active_mask = (ternary != 0)
        active_sum = (W.abs() * active_mask).sum(dim=1)  # (out_f,)
        active_count = active_mask.sum(dim=1).float()  # (out_f,)
        
        # Evitar division por cero
        scale = torch.where(active_count > 0, active_sum / active_count, torch.zeros_like(active_count))
        
        # Reconstruir
        W_ternary = ternary * scale.unsqueeze(1)
        
    return W_ternary, ternary, scale, threshold

# Probar distintos thresholds
print(f"\n{'threshold':>10} {'Compresion':>12} {'Sparsity':>10} {'wMSE':>12} {'COS_gate':>10}")
print("-" * 60)

results = []
for ts in [0.3, 0.5, 0.7, 0.9, 1.1, 1.5]:
    W_t, ternary, scale, th = ternary_quantize(W_g, ts)
    
    wmse = F.mse_loss(W_t, W_g).item()
    
    with torch.no_grad():
        gate_t = F.silu(x @ W_t.T)
        cos = F.cosine_similarity(gate_t.reshape(-1, out_f), gate_teacher.reshape(-1, out_f), dim=1).mean().item()
    
    sparsity = (ternary == 0).float().mean().item() * 100
    bits = sparsity / 100 * 1 + (1 - sparsity/100) * 2  # aprox: 0's take 1 bit, +/-1 take 2 bits
    comp = 32 / bits  # vs FP32
    
    results.append((ts, comp, cos, wmse, sparsity))
    print(f"{ts:>10.1f} {comp:>10.1f}x {sparsity:>9.1f}% {wmse:>12.8f} {cos:>10.4f}")

# ─── Mejor configuracion ───
best = max(results, key=lambda r: r[2])  # best COS
print(f"\n>>> Mejor COS: threshold={best[0]}, comp={best[1]:.0f}x, COS={best[2]:.4f}")
print(f"    Balanceado: threshold=0.5-0.7 da ~20x con COS ~0.8+")

# ─── Entrenar escala (fine-tune) ───
print("\n" + "="*60)
print("Fine-tuning de escalas ternarias (100 steps)")
print("="*60)

# Fijar la mascara ternaria, entrenar solo las escalas
_, ternary_fixed, initial_scale, _ = ternary_quantize(W_g, 0.7)
scale_param = torch.nn.Parameter(initial_scale.clone())
opt = torch.optim.Adam([scale_param], lr=1e-2)

W_g_d = W_g.detach()

for step in range(100):
    opt.zero_grad()
    W_t = ternary_fixed * scale_param.unsqueeze(1)
    loss = F.mse_loss(W_t, W_g_d)
    loss.backward()
    opt.step()

with torch.no_grad():
    W_t = ternary_fixed * scale_param.unsqueeze(1)
    wmse_ft = F.mse_loss(W_t, W_g).item()
    gate_t = F.silu(x @ W_t.T)
    cos_ft = F.cosine_similarity(gate_t.reshape(-1, out_f), gate_teacher.reshape(-1, out_f), dim=1).mean().item()
    
print(f"  wMSE={wmse_ft:.8f}  COS={cos_ft:.4f}")
print(f"  Mejora vs sin fine-tune: {cos_ft - best[2]:+.4f}")

# ─── Tamaño real ───
sparsity = (ternary_fixed == 0).float().mean().item()
n_ternary = ternary_fixed.numel()
n_nonzero = n_ternary * (1 - sparsity)
n_zero = n_ternary * sparsity

# Almacenamiento en bits: 1 bit para 0/±1, + 1 bit extra para ±1 distinción
# Mas simple: 2 bits por elemento (4 estados posibles, usamos 3)
bits_element = 2
bytes_ternary = n_ternary * bits_element / 8
bytes_scales = out_f * 2  # FP16
bytes_original = n_ternary * 4  # FP32

print(f"\n{'='*60}")
print(f"TAMAÑO ESTIMADO")
print(f"{'='*60}")
print(f"  Original:  {bytes_original/1024/1024:.0f} MB (FP32)")
print(f"  Ternario:  {bytes_ternary/1024/1024:.1f} MB + escalas {bytes_scales/1024:.1f} KB")
print(f"  Total:     {(bytes_ternary + bytes_scales)/1024/1024:.1f} MB")
print(f"  Compresion: {bytes_original/(bytes_ternary + bytes_scales):.0f}x")
print(f"  COS_gate:  {cos_ft:.4f} {'✅ > 0.9!' if cos_ft > 0.9 else '☝️ mejorable'}")
