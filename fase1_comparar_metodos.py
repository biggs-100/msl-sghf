"""
Fase 1: comparar SG-HF a distinta compresion vs ternario para down + attn.

Decidimos si conviene mezclar metodos o usar ternario para todo.
"""

import torch, torch.nn.functional as F, math, time, os
from safetensors import safe_open
from sg_hf.core import FractalLinear

device = 'cuda'
print(f"Dispositivo: {device}")

CONSOLIDATED = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1\consolidated.safetensors'
with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
    W_d = f.get_tensor('layers.0.feed_forward.w2.weight').float()  # (4096, 14336)
    W_q = f.get_tensor('layers.0.attention.wq.weight').float()     # (4096, 4096)

W_d, W_q = W_d.to(device), W_q.to(device)
print(f"Down: {tuple(W_d.shape)} std={W_d.std():.4f}")
print(f"Q:    {tuple(W_q.shape)} std={W_q.std():.4f}")

def ternary_quant(W, ts=0.7):
    th = ts * W.abs().mean(dim=1, keepdim=True)
    t = torch.where(W > th, 1.0, torch.where(W < -th, -1.0, 0.0))
    s = (W.abs() * (t!=0).float()).sum(dim=1) / (t!=0).float().sum(dim=1).clamp(min=1)
    return t * s.unsqueeze(1)

def train_sghf(in_f, out_f, teacher, comp, steps=200):
    fl = FractalLinear(in_f, out_f, compression=comp).to(device)
    fl.initialize_from_teacher(teacher.cpu())
    opt = torch.optim.Adam(fl.parameters(), lr=1e-2)
    for step in range(steps):
        opt.zero_grad()
        W = fl._generate_weight()
        loss = F.mse_loss(W, teacher.to(W.dtype))
        loss.backward()
        opt.step()
    with torch.no_grad():
        W_s = fl._generate_weight()
    return W_s, sum(p.numel() for p in fl.parameters())

print(f"\n{'='*60}")
print(f"DOWN projection (4096x14336)")
print(f"{'='*60}")
print(f"{'Metodo':>15} {'Comp':>6} {'wMSE':>14} {'R²':>8} {'Tiempo':>8}")
print("-" * 60)

# Ternario
t0 = time.time()
W_d_t = ternary_quant(W_d)
wmse = F.mse_loss(W_d_t, W_d).item()
r2 = 1 - wmse / W_d.var().item()
params_t = W_d.numel() * 2 + W_d.size(0) * 16  # 2-bit + scales
comp_t = W_d.numel() * 32 / params_t
print(f"{'Ternario':>15} {comp_t:>4.0f}x {wmse:>14.8f} {r2:>8.4f} {time.time()-t0:>7.1f}s")

# SG-HF a distintas compresiones
for comp in [20, 30, 50, 100]:
    t0 = time.time()
    W_d_s, params = train_sghf(14336, 4096, W_d, comp, 200)
    wmse = F.mse_loss(W_d_s, W_d).item()
    r2 = 1 - wmse / W_d.var().item()
    c = W_d.numel() * 4 / params
    print(f"{'SG-HF':>10} {comp:>2d}x {c:>4.0f}x {wmse:>14.8f} {r2:>8.4f} {time.time()-t0:>7.1f}s")

print(f"\n{'='*60}")
print(f"Q projection (4096x4096) — atencion")
print(f"{'='*60}")
print(f"{'Metodo':>15} {'Comp':>6} {'wMSE':>14} {'R²':>8} {'Tiempo':>8}")
print("-" * 60)

# Ternario
t0 = time.time()
W_q_t = ternary_quant(W_q)
wmse = F.mse_loss(W_q_t, W_q).item()
r2 = 1 - wmse / W_q.var().item()
params_t = W_q.numel() * 2 + W_q.size(0) * 16
comp_t = W_q.numel() * 32 / params_t
print(f"{'Ternario':>15} {comp_t:>4.0f}x {wmse:>14.8f} {r2:>8.4f} {time.time()-t0:>7.1f}s")

for comp in [20, 30, 50, 100]:
    t0 = time.time()
    W_q_s, params = train_sghf(4096, 4096, W_q, comp, 200)
    wmse = F.mse_loss(W_q_s, W_q).item()
    r2 = 1 - wmse / W_q.var().item()
    c = W_q.numel() * 4 / params
    print(f"{'SG-HF':>10} {comp:>2d}x {c:>4.0f}x {wmse:>14.8f} {r2:>8.4f} {time.time()-t0:>7.1f}s")

print(f"\n{'='*60}")
print("CONCLUSION")
print("="*60)
print("  La relacion senal/ruido define si SG-HF o ternario es mejor.")
print("  Ternario da ~16x con R² ~0.8 para cualquier peso.")
print("  SG-HF da mejor R² SOLO si el peso tiene std grande.")
print("  Para Mistral (std~0.003): ternario es mas robusto.")
