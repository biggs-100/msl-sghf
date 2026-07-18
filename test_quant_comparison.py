"""
Comparacion completa: ternary vs 2-bit vs 3-bit vs 4-bit.
Mide MLP output COS en Mistral-7B layer 0.
"""
import torch, gc, time
from transformers import AutoModelForCausalLM

print("Cargando Mistral...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    'mistralai/Mistral-7B-v0.3', dtype=torch.float16, device_map='cpu', low_cpu_mem_usage=True)
mlp = model.model.layers[0].mlp
Wg, Wu, Wd = [w.data.float() for w in [mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight]]
print(f"  {time.time()-t0:.0f}s")

torch.manual_seed(42)
hn = torch.randn(1, 32, 4096) * 1.5
out_o = (torch.nn.functional.silu(hn @ Wg.T) * (hn @ Wu.T)) @ Wd.T

def mlp_cos(Wg_q, Wu_q, Wd_q):
    g = torch.nn.functional.silu(hn @ Wg_q.T)
    u = hn @ Wu_q.T
    o = (g * u) @ Wd_q.T
    return (out_o.flatten() * o.flatten()).sum() / (out_o.norm() * o.norm())

def quantize(W, n_bits, opt_scale=True):
    """Cuantizacion generica con N bits."""
    M, N = W.shape
    s = W.std(dim=1, keepdim=True)
    aw, sg = W.abs(), W.sign()
    n_levels = 2 ** n_bits
    # thresholds y values uniformes
    ths = [(i / n_levels) * 1.5 for i in range(1, n_levels)]
    vals = [(i / n_levels) * 1.5 for i in range(1, n_levels)]
    # Codificar
    code = torch.zeros(M, N, dtype=torch.long)
    for i, th in enumerate(ths):
        code = torch.where((aw > th * s) & (code == 0), i + 1, code)
    # Mapear a valores
    qv = torch.zeros_like(W)
    for ci, vi in {i+1:v for i, v in enumerate(vals)}.items():
        qv = qv + torch.where(code == ci, sg * vi * s, torch.zeros_like(qv))
    # Optimizar escala por fila
    if opt_scale:
        num = (W * qv).sum(dim=1, keepdim=True)
        den = (qv.pow(2)).sum(dim=1, keepdim=True) + 1e-8
        qv = qv * (num / den)
    return qv

# --- Referencia: Ternary original (umbral 0.7) ---
st = Wg.std(dim=1, keepdim=True)
mask_g = torch.where(Wg.abs() > 0.7*st, Wg.sign(), torch.zeros_like(Wg))
scale_g = (Wg*mask_g).sum(dim=1,keepdim=True) / (mask_g.abs().sum(dim=1,keepdim=True)+1e-8)
mask_u = torch.where(Wu.abs() > 0.7*Wu.std(dim=1,keepdim=True), Wu.sign(), torch.zeros_like(Wu))
scale_u = (Wu*mask_u).sum(dim=1,keepdim=True) / (mask_u.abs().sum(dim=1,keepdim=True)+1e-8)
mask_d = torch.where(Wd.abs() > 0.7*Wd.std(dim=1,keepdim=True), Wd.sign(), torch.zeros_like(Wd))
scale_d = (Wd*mask_d).sum(dim=1,keepdim=True) / (mask_d.abs().sum(dim=1,keepdim=True)+1e-8)

print("\nResultados:")
print("-" * 55)
print(f"  {'Metodo':<20} {'Bits':<8} {'Compres':<10} {'MLP COS':<10}")
print("-" * 55)

cos_ref = mlp_cos(mask_g*scale_g, mask_u*scale_u, mask_d*scale_d).item()
print(f"  {'Ternary (ref)':<20} {'1.58':<8} {16:<10.1f}x {cos_ref:<10.4f}")

# 2-bit
cos_2 = mlp_cos(quantize(Wg,2), quantize(Wu,2), quantize(Wd,2)).item()
print(f"  {'2-bit uniforme':<20} {'2':<8} {16:<10.1f}x {cos_2:<10.4f}")

# 3-bit
cos_3 = mlp_cos(quantize(Wg,3), quantize(Wu,3), quantize(Wd,3)).item()
print(f"  {'3-bit uniforme':<20} {'3':<8} {10.6:<10.1f}x {cos_3:<10.4f}")

# 4-bit
cos_4 = mlp_cos(quantize(Wg,4), quantize(Wu,4), quantize(Wd,4)).item()
print(f"  {'4-bit uniforme':<20} {'4':<8} {8:<10.1f}x {cos_4:<10.4f}")

# Referencia: sin cuantizar
cos_full = mlp_cos(Wg, Wu, Wd).item()
print(f"  {'Original (FP32)':<20} {'32':<8} {1:<10.1f}x {cos_full:<10.4f}")

print("-" * 55)
print(f"\nConclusion: 2-bit da COS {cos_2:.4f} vs ternary {cos_ref:.4f} a igual compresion.")
print(f"3-bit da COS {cos_3:.4f} a 10.6x (compite con GPTQ 4-bit).")

del model; gc.collect(); torch.cuda.empty_cache()
print("\nOK")
