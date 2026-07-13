"""
Test: MLP completo de Mistral-7B layer 0 con enfoque hibrido.

Gate+Up: ternario {-1,0,+1} + escala por fila (16x)
Down:    SG-HF Kronecker (100x) — carga seed existente

Mide COS del output completo del MLP (lo que importa).
"""

import torch, torch.nn.functional as F, time, math, os
from safetensors import safe_open
from sg_hf.core import FractalLinear

device = 'cuda'
print(f"Dispositivo: {device}")

# Cargar teacher layer 0
CONSOLIDATED = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1\consolidated.safetensors'
with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
    W_g = f.get_tensor('layers.0.feed_forward.w1.weight').float()  # (14336, 4096)
    W_u = f.get_tensor('layers.0.feed_forward.w3.weight').float()  # (14336, 4096)
    W_d = f.get_tensor('layers.0.feed_forward.w2.weight').float()  # (4096, 14336)

# Input
torch.manual_seed(42)
x = torch.randn(1, 16, 4096, device=device)

# Teacher reference
W_g, W_u, W_d = W_g.to(device), W_u.to(device), W_d.to(device)
with torch.no_grad():
    gate_t = F.silu(x @ W_g.T)
    up_t = x @ W_u.T
    y_t = (gate_t * up_t) @ W_d.T

print(f"Teacher MLP output: std={y_t.std():.4f}")

# ─── 1. Gate ternario ───
print("\n>>> Gate: ternario {-1,0,+1} + escala...")
t0 = time.time()
with torch.no_grad():
    th = 0.7 * W_g.abs().mean(dim=1, keepdim=True)
    ternary_g = torch.where(W_g > th, 1.0, torch.where(W_g < -th, -1.0, 0.0))
    scale_g = (W_g.abs() * (ternary_g!=0).float()).sum(dim=1) / (ternary_g!=0).float().sum(dim=1).clamp(min=1)
    W_g_q = ternary_g * scale_g.unsqueeze(1)
    
    gate_s = F.silu(x @ W_g_q.T)
    cos_g = F.cosine_similarity(gate_s.reshape(-1, 14336), gate_t.reshape(-1, 14336), dim=1).mean().item()
    spar_g = (ternary_g == 0).float().mean().item()
print(f"  COS_gate={cos_g:.4f}  sparsity={spar_g:.0%}  ({time.time()-t0:.1f}s)")

# ─── 2. Up ternario ───
print("\n>>> Up: ternario {-1,0,+1} + escala...")
t0 = time.time()
with torch.no_grad():
    th_u = 0.7 * W_u.abs().mean(dim=1, keepdim=True)
    ternary_u = torch.where(W_u > th_u, 1.0, torch.where(W_u < -th_u, -1.0, 0.0))
    scale_u = (W_u.abs() * (ternary_u!=0).float()).sum(dim=1) / (ternary_u!=0).float().sum(dim=1).clamp(min=1)
    W_u_q = ternary_u * scale_u.unsqueeze(1)
    
    up_s = x @ W_u_q.T
    cos_u = F.cosine_similarity(up_s.reshape(-1, 14336), up_t.reshape(-1, 14336), dim=1).mean().item()
print(f"  COS_up={cos_u:.4f}  ({time.time()-t0:.1f}s)")

# ─── 3. Down SG-HF ───
print("\n>>> Down: SG-HF Kronecker (seed existente)...")
t0 = time.time()
SEED_DIR = 'mistral_seeds'
seed_path = os.path.join(SEED_DIR, 'layers_0_feed_forward_w2_weight.pt')
fl_d = FractalLinear(14336, 4096, compression=100).to(device)

if os.path.exists(seed_path):
    sd = torch.load(seed_path, weights_only=False, map_location='cpu')
    fl_d.load_state_dict(sd, strict=False)  # seed viejo sin 'scale'
    fl_d.to(device)
    with torch.no_grad():
        W_d_s = fl_d._generate_weight().to(device, dtype=torch.float32)
    print(f"  Seed cargado de: {seed_path}  |  W_d_s std={W_d_s.std():.4f}")
else:
    print(f"  Seed no encontrado, comprimiendo desde teacher...")
    fl_d.initialize_from_teacher(W_d.cpu())
    opt = torch.optim.Adam(fl_d.parameters(), lr=1e-2)
    for step in range(200):
        opt.zero_grad()
        W = fl_d._generate_weight()
        loss = F.mse_loss(W, W_d.cpu())
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"    step {step}: MSE={loss.item():.8f}")
    torch.save(fl_d.cpu().state_dict(), seed_path)
    fl_d.to(device)
    with torch.no_grad():
        W_d_s = fl_d._generate_weight().to(device, dtype=torch.float32)

# Verificar que W_d_s reconstruye bien los pesos (weight MSE)
wmse_d = F.mse_loss(W_d_s, W_d).item()
print(f"  Down weight MSE={wmse_d:.8f}  (R²={1-wmse_d/W_d.var().item():.4f})")

# ─── 4. MLP completo ───
print("\n" + "="*60)
print("MLP COMPLETO: gate_tern + up_tern + down_SG-HF")
print("="*60)

with torch.no_grad():
    gate_h = F.silu(x @ W_g_q.T)
    up_h = x @ W_u_q.T
    y_s = (gate_h * up_h) @ W_d_s.T
    
    cos_mlp = F.cosine_similarity(y_s.reshape(-1, 4096), y_t.reshape(-1, 4096), dim=1).mean().item()
    mse_mlp = F.mse_loss(y_s, y_t).item()

print(f"  COS_MLP  = {cos_mlp:.4f}")
print(f"  MSE_MLP  = {mse_mlp:.8f}")

# ─── 5. Fine-tune escalas ternarias (opcional) ───
print("\n" + "="*60)
print("FINE-TUNE: escalas ternarias en contexto del MLP completo")
print("="*60)

# Parametros entrenables: solo escalas de gate y up
scale_g_p = torch.nn.Parameter(scale_g.clone())
scale_u_p = torch.nn.Parameter(scale_u.clone())
opt = torch.optim.Adam([scale_g_p, scale_u_p], lr=1e-3)

# Teacher detached
W_g_d = W_g.detach()
W_u_d = W_u.detach()
W_d_d = W_d.detach()
W_d_s_d = W_d_s.detach()

for step in range(100):
    opt.zero_grad()
    
    # Forward con escalas entrenables
    W_g_f = ternary_g * scale_g_p.unsqueeze(1)
    W_u_f = ternary_u * scale_u_p.unsqueeze(1)
    
    gate_f = F.silu(x @ W_g_f.T)
    up_f = x @ W_u_f.T
    y_f = (gate_f * up_f) @ W_d_s_d.T
    
    loss = F.mse_loss(y_f, y_t.detach())
    loss.backward()
    torch.nn.utils.clip_grad_norm_([scale_g_p, scale_u_p], 1.0)
    opt.step()
    
    if step % 50 == 0:
        with torch.no_grad():
            cos_f = F.cosine_similarity(y_f.reshape(-1, 4096), y_t.reshape(-1, 4096), dim=1).mean().item()
        print(f"  step {step}: loss={loss.item():.8f}  COS={cos_f:.4f}")

with torch.no_grad():
    W_g_f = ternary_g * scale_g_p.unsqueeze(1)
    W_u_f = ternary_u * scale_u_p.unsqueeze(1)
    gate_f = F.silu(x @ W_g_f.T)
    up_f = x @ W_u_f.T
    y_f = (gate_f * up_f) @ W_d_s_d.T
    cos_ft = F.cosine_similarity(y_f.reshape(-1, 4096), y_t.reshape(-1, 4096), dim=1).mean().item()

print(f"\n  COS antes: {cos_mlp:.4f}")
print(f"  COS despues: {cos_ft:.4f}")
print(f"  Mejoria: {cos_ft - cos_mlp:+.4f}")

# ─── Resumen ───
print("\n" + "="*60)
print("RESUMEN LAYER 0")
print("="*60)
s_g = (scale_g_p.data.std().item(), scale_g.data.std().item())
print(f"\n  Gate:  ternario 16x, sparsity={spar_g:.0%}")
print(f"  Up:    ternario 16x")
print(f"  Down:  SG-HF 100x")
print(f"  MLP output COS: {cos_ft:.4f}")
print(f"\n  {'✅ FUNCIONAL' if cos_ft > 0.85 else '⚠️  MEJORABLE'} (umbral: COS > 0.85 = calidad usable)")
