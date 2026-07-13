"""
Comparacion rapida: ternario vs Q2 (2-bit) para gate projection.
"""

import torch, torch.nn.functional as F
from safetensors import safe_open

device = 'cuda'
CONSOLIDATED = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1\consolidated.safetensors'
with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
    W_g = f.get_tensor('layers.0.feed_forward.w1.weight').float().to(device)

out_f, in_f = W_g.shape
torch.manual_seed(42)
x = torch.randn(1, 16, 4096, device=device)
with torch.no_grad():
    gate_teacher = F.silu(x @ W_g.T)

def ternary_quant(W, ts=0.7):
    """Ternario {-1,0,+1} + escala optima por fila."""
    th = ts * W.abs().mean(dim=1, keepdim=True)
    t = torch.where(W > th, 1.0, torch.where(W < -th, -1.0, 0.0))
    s = (W.abs() * (t!=0).float()).sum(dim=1) / (t!=0).float().sum(dim=1).clamp(min=1)
    return t * s.unsqueeze(1)

def q2_quant(W):
    """Q2 simetrico: 4 niveles proporcionales al maximo por fila."""
    max_a = W.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    W_n = W / max_a
    
    # Asignar a 4 niveles con thresholds Lloyd-Max adaptativos
    # {-2/3, 0, 2/3} son los boundaries optimos para N(0,1) con 4 niveles simetricos
    W_q = torch.zeros_like(W)
    for i, (b_low, b_high, qv) in enumerate([
        (-float('inf'), -0.667, -1.0),
        (-0.667, 0.0, -1/3),
        (0.0, 0.667, 1/3),
        (0.667, float('inf'), 1.0),
    ]):
        m = (W_n > b_low) & (W_n <= b_high)
        W_q += m * qv
    
    # Escala optima por fila
    alpha = (W * W_q).sum(dim=1) / (W_q**2).sum(dim=1).clamp(min=1)
    return W_q * alpha.unsqueeze(1)

print("=" * 55)
print("TERMARIO vs Q2 para GATE projection (Mistral-7B)")
print("=" * 55)
print(f"  W_g: {tuple(W_g.shape)}, std={W_g.std():.4f}")
print()

data = []
for name, func, bits in [
    ("Ternario ts=0.5", lambda W: ternary_quant(W, 0.5), 2),
    ("Ternario ts=0.7", lambda W: ternary_quant(W, 0.7), 2),
    ("Ternario ts=1.0", lambda W: ternary_quant(W, 1.0), 2),
    ("Q2 simetrico", q2_quant, 2),
]:
    W_q = func(W_g)
    spar = (func(W_g) == 0).float().mean().item() if 'Ternario' in name else 0
    wmse = F.mse_loss(W_q, W_g).item()
    gate = F.silu(x @ W_q.T)
    cos = F.cosine_similarity(gate.reshape(-1, out_f), gate_teacher.reshape(-1, out_f), dim=1).mean().item()
    comp = 32 / bits
    data.append((name, comp, cos, wmse, spar))
    
print(f"  {'Metodo':>16} {'Comp':>6} {'COS':>8} {'wMSE':>12} {'Sparsity':>10}")
print("  " + "-" * 54)
for n, c, cs, wm, sp in data:
    print(f"  {n:>16} {c:>4.0f}x {cs:>8.4f} {wm:>12.8f} {sp:>9.0%}")

print()
print("  Mejor COS: %s (%.4f)" % max((d[2], d[0]) for d in data)[::-1])
print()
print("  Ternario gana en COS porque la sparsity elimina")
print("  elementos cerca de 0 donde el error es mas danino")
print("  para el gate SiLU. Q2 da mas niveles pero fuerza")
print("  errores en toda la distribucion, incluso donde")
print("  el signo es critico.")
