"""
Test: ternario {-1,0,+1} + escala por fila para TODOS los pesos del MLP.

Gate, Up, Down: todos ternario. Sin SG-HF.

Mide COS del MLP completo.
"""

import torch, torch.nn.functional as F, time
from safetensors import safe_open

device = 'cuda'
CONSOLIDATED = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1\consolidated.safetensors'
with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
    W_g = f.get_tensor('layers.0.feed_forward.w1.weight').float().to(device)
    W_u = f.get_tensor('layers.0.feed_forward.w3.weight').float().to(device)
    W_d = f.get_tensor('layers.0.feed_forward.w2.weight').float().to(device)

torch.manual_seed(42)
x = torch.randn(1, 16, 4096, device=device)

with torch.no_grad():
    gate_t = F.silu(x @ W_g.T)
    up_t = x @ W_u.T
    y_t = (gate_t * up_t) @ W_d.T

def ternary_quant(W, ts=0.7):
    """Ternario optimo por fila."""
    th = ts * W.abs().mean(dim=1, keepdim=True)
    t = torch.where(W > th, 1.0, torch.where(W < -th, -1.0, 0.0))
    s = (W.abs() * (t!=0).float()).sum(dim=1) / (t!=0).float().sum(dim=1).clamp(min=1)
    return t * s.unsqueeze(1), t, s

print("=" * 55)
print("TERNARIO PURO para MLP de Mistral-7B layer 0")
print("=" * 55)
print(f"  W_g: {tuple(W_g.shape)}, std={W_g.std():.4f}")
print(f"  W_u: {tuple(W_u.shape)}, std={W_u.std():.4f}")
print(f"  W_d: {tuple(W_d.shape)}, std={W_d.std():.4f}")

# Probar distintos thresholds
for ts in [0.5, 0.7, 1.0]:
    print(f"\n--- threshold={ts} ---")
    t0 = time.time()
    
    W_g_q, tg, sg = ternary_quant(W_g, ts)
    W_u_q, tu, su = ternary_quant(W_u, ts)
    W_d_q, td, sd = ternary_quant(W_d, ts)
    
    with torch.no_grad():
        gate_s = F.silu(x @ W_g_q.T)
        up_s = x @ W_u_q.T
        y_s = (gate_s * up_s) @ W_d_q.T
        
        cos_mlp = F.cosine_similarity(y_s.reshape(-1, 4096), y_t.reshape(-1, 4096), dim=1).mean().item()
        cos_g = F.cosine_similarity(gate_s.reshape(-1, 14336), gate_t.reshape(-1, 14336), dim=1).mean().item()
    
    sp_g = (tg == 0).float().mean().item()
    sp_d = (td == 0).float().mean().item()
    
    # Compresion: ternario 2-bit + escala FP16 por fila
    bits_g = W_g.numel() * 2  # ternario
    bits_s_g = W_g.size(0) * 16  # escala
    comp_g = (W_g.numel() * 32) / (bits_g + bits_s_g)
    
    print(f"  COS_gate={cos_g:.4f}  COS_MLP={cos_mlp:.4f}")
    print(f"  Sparsity: gate={sp_g:.0%}  down={sp_d:.0%}")
    print(f"  Compresion: {comp_g:.0f}x por peso")
    print(f"  Tiempo: {time.time()-t0:.1f}s")

# Fine-tune escalas
print("\n" + "=" * 55)
print("FINE-TUNE: escalas ternarias con MLP completo")
print("=" * 55)

ts = 0.7
W_g_q, tg, sg0 = ternary_quant(W_g, ts)
W_u_q, tu, su0 = ternary_quant(W_u, ts)
W_d_q, td, sd0 = ternary_quant(W_d, ts)

sg_p = torch.nn.Parameter(sg0.clone())
su_p = torch.nn.Parameter(su0.clone())
sd_p = torch.nn.Parameter(sd0.clone())
opt = torch.optim.Adam([sg_p, su_p, sd_p], lr=1e-3)

for step in range(100):
    opt.zero_grad()
    W_g_f = tg * sg_p.unsqueeze(1)
    W_u_f = tu * su_p.unsqueeze(1)
    W_d_f = td * sd_p.unsqueeze(1)
    
    gate_f = F.silu(x @ W_g_f.T)
    up_f = x @ W_u_f.T
    y_f = (gate_f * up_f) @ W_d_f.T
    
    loss = F.mse_loss(y_f, y_t.detach())
    loss.backward()
    torch.nn.utils.clip_grad_norm_([sg_p, su_p, sd_p], 1.0)
    opt.step()
    
    if step % 25 == 0:
        cos = F.cosine_similarity(y_f.reshape(-1, 4096), y_t.reshape(-1, 4096), dim=1).mean().item()
        print(f"  step {step}: loss={loss.item():.8f}  COS={cos:.4f}")

with torch.no_grad():
    W_g_f = tg * sg_p.unsqueeze(1)
    W_u_f = tu * su_p.unsqueeze(1)
    W_d_f = td * sd_p.unsqueeze(1)
    gate_f = F.silu(x @ W_g_f.T)
    up_f = x @ W_u_f.T
    y_f = (gate_f * up_f) @ W_d_f.T
    cos_ft = F.cosine_similarity(y_f.reshape(-1, 4096), y_t.reshape(-1, 4096), dim=1).mean().item()
    wmse_g = F.mse_loss(W_g_f, W_g).item()
    wmse_d = F.mse_loss(W_d_f, W_d).item()

# Stats finales
total_bits = (tg.numel() + tu.numel() + td.numel()) * 2
total_scales = (sg_p.numel() + su_p.numel() + sd_p.numel()) * 16
total_orig = (W_g.numel() + W_u.numel() + W_d.numel()) * 32
comp_total = total_orig / (total_bits + total_scales)

print(f"\n{'='*55}")
print(f"RESULTADO FINAL")
print(f"{'='*55}")
print(f"  COS_MLP  = {cos_ft:.4f}")
print(f"  wMSE_gate= {wmse_g:.8f}")
print(f"  wMSE_down= {wmse_d:.8f}")
print(f"  Sparsity  gate={ (tg==0).float().mean().item():.0%}, up={ (tu==0).float().mean().item():.0%}, down={ (td==0).float().mean().item():.0%}")
print(f"  Compresion total MLP: {comp_total:.0f}x")
print(f"\n  {'✅ FUNCIONAL' if cos_ft > 0.85 else '⚠️  MEJORABLE (pero mucho mejor que Kronecker)'}")
