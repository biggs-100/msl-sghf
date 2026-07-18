"""
Busqueda de umbrales optimos para 2-bit en Mistral-7B.
Estrategia: percentiles de |W| para distribuir niveles donde hay datos.
"""
import torch, gc, time, math
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

def quantile_thresholds(W, pcts):
    """Encontrar thresholds basados en percentiles de |W|."""
    aw = W.abs().flatten()
    sorted_vals = aw.sort()[0]
    n = len(sorted_vals)
    return [sorted_vals[int(p * n)].item() for p in pcts]

def quantize_2bit(W, thresholds, opt_scale=True):
    """2-bit: thresholds define limites entre niveles."""
    M, N = W.shape
    s = W.std(dim=1, keepdim=True)
    aw, sg = W.abs(), W.sign()
    # thresholds: [t1, t2, t3] separan 4 niveles (0,1,2,3)
    t1, t2, t3 = [th for th in thresholds]
    code = torch.zeros_like(W, dtype=torch.long)
    code = torch.where(aw > t3, 3, code)
    code = torch.where((aw > t2) & (code == 0), 2, code)
    code = torch.where((aw > t1) & (code == 0), 1, code)
    # Valores: media de |W| dentro de cada nivel
    v1 = W.abs()[code == 1].mean().item() if (code == 1).any() else t1 * 1.5
    v2 = W.abs()[code == 2].mean().item() if (code == 2).any() else t2 * 1.2
    v3 = W.abs()[code == 3].mean().item() if (code == 3).any() else t3 * 1.1
    qv = torch.zeros_like(W)
    qv = qv + torch.where(code == 1, sg * v1, torch.zeros_like(qv))
    qv = qv + torch.where(code == 2, sg * v2, torch.zeros_like(qv))
    qv = qv + torch.where(code == 3, sg * v3, torch.zeros_like(qv))
    if opt_scale:
        # Optimizar escala por fila via LS
        num = (W * qv).sum(dim=1, keepdim=True)
        den = (qv.pow(2)).sum(dim=1, keepdim=True) + 1e-8
        qv = qv * (num / den)
    return qv

# Distribucion de |W| (todos los pesos juntos)
all_w = torch.cat([Wg.abs().flatten(), Wu.abs().flatten(), Wd.abs().flatten()])
sorted_w = all_w.sort()[0]
n = len(sorted_w)
print(f"\nDistribucion de |W|: p50={sorted_w[int(0.5*n)]:.6f}, p80={sorted_w[int(0.8*n)]:.6f}, p90={sorted_w[int(0.9*n)]:.6f}, p95={sorted_w[int(0.95*n)]:.6f}, p99={sorted_w[int(0.99*n)]:.6f}, max={sorted_w[-1]:.6f}")

# Estrategias de thresholds (percentiles)
strategies = [
    ("ternary (ref)", None, None),  # placeholder
    ("2bit pct[60,80,95]", [0.6, 0.8, 0.95]),
    ("2bit pct[50,75,90]", [0.5, 0.75, 0.9]),
    ("2bit pct[50,80,95]", [0.5, 0.8, 0.95]),
    ("2bit pct[70,85,95]", [0.7, 0.85, 0.95]),
    ("2bit pct[60,85,97]", [0.6, 0.85, 0.97]),
    ("2bit pct[55,75,92]", [0.55, 0.75, 0.92]),
]

# Referencia ternary
st = Wg.std(dim=1, keepdim=True)
mask_g = torch.where(Wg.abs()>0.7*st, Wg.sign(), torch.zeros_like(Wg))
sc_g = (Wg*mask_g).sum(dim=1,keepdim=True)/(mask_g.abs().sum(dim=1,keepdim=True)+1e-8)
mask_u = torch.where(Wu.abs()>0.7*Wu.std(dim=1,keepdim=True), Wu.sign(), torch.zeros_like(Wu))
sc_u = (Wu*mask_u).sum(dim=1,keepdim=True)/(mask_u.abs().sum(dim=1,keepdim=True)+1e-8)
mask_d = torch.where(Wd.abs()>0.7*Wd.std(dim=1,keepdim=True), Wd.sign(), torch.zeros_like(Wd))
sc_d = (Wd*mask_d).sum(dim=1,keepdim=True)/(mask_d.abs().sum(dim=1,keepdim=True)+1e-8)

print(f"\nResultados:")
print(f"  {'Estrategia':<25} {'Umbrales':<25} {'MLP COS':<10}")
print("  " + "-" * 60)

cos_ref = mlp_cos(mask_g*sc_g, mask_u*sc_u, mask_d*sc_d).item()
print(f"  {'ternary (ref)':<25} {'0.7*std':<25} {cos_ref:<10.4f}")

for name, pcts in strategies[1:]:
    # Encontrar thresholds globales
    th = quantile_thresholds(all_w.reshape(-1,1), pcts)
    th_str = f"[{th[0]:.4f}, {th[1]:.4f}, {th[2]:.4f}]"
    cos = mlp_cos(
        quantize_2bit(Wg, th), quantize_2bit(Wu, th), quantize_2bit(Wd, th)
    ).item()
    print(f"  {name:<25} {th_str:<25} {cos:<10.4f}")

# Mejor: thresholds por capa (no global)
print(f"\n  Mejor: thresholds por capa:")
best_total = 0
for pcts in [[0.5,0.75,0.9], [0.6,0.8,0.95], [0.5,0.8,0.95]]:
    Wg_q = quantize_2bit(Wg, quantile_thresholds(Wg, pcts))
    Wu_q = quantize_2bit(Wu, quantile_thresholds(Wu, pcts))
    Wd_q = quantize_2bit(Wd, quantile_thresholds(Wd, pcts))
    cos = mlp_cos(Wg_q, Wu_q, Wd_q).item()
    th = [f"{x:.4f}" for x in quantile_thresholds(all_w, pcts)]
    print(f"    pcts={pcts}: COS={cos:.4f}")
    if cos > best_total:
        best_total = cos

print(f"\n  Mejor COS global: {best_total:.4f}")

del model; gc.collect(); torch.cuda.empty_cache()
print(f"\nOK")
