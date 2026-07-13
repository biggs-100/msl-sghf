"""
¿Funciona el SVD global para comprimir el gate projection?

W_g ≈ U @ diag(S) @ Vh  con rango R.
Cada fila puede ser CUALQUIER combinacion lineal de R vectores base.
No hay restriccion de colinealidad por bloque.

Mide: cuantos componentes SVD para COS_gate > 0.9?
"""

import torch, torch.nn.functional as F, time, math
from safetensors import safe_open

device = 'cuda'
print(f"Dispositivo: {device}")

CONSOLIDATED = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1\consolidated.safetensors'
with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
    W_g = f.get_tensor('layers.0.feed_forward.w1.weight').float()
    
print(f"W_g: {tuple(W_g.shape)}  std={W_g.std():.4f}")
print(f"Parametros totales: {W_g.numel():,}")

# Input de prueba
torch.manual_seed(42)
x = torch.randn(1, 16, 4096, device=device)

with torch.no_grad():
    gate_teacher = F.silu(x @ W_g.to(device).T)

# ─── SVD completa en CPU ───
print("\n>>> Computando SVD completo en CPU...")
t0 = time.time()
U, S, Vh = torch.linalg.svd(W_g.cpu(), full_matrices=False)
print(f"SVD completo: {time.time()-t0:.1f}s")
print(f"U: {U.shape}, S: {S.shape}, Vh: {Vh.shape}")
U = U.cuda()
S = S.cuda()
Vh = Vh.cuda()

# Varianza acumulada
var_total = (S**2).sum().item()
var_cum = (S**2).cumsum(0).cpu()

# ─── Evaluar compression con distintos rangos ───
print(f"\n{'Rango':>6} {'Params':>10} {'Compresion':>12} {'Var_acum':>10} {'COS_gate':>10}")
print("-" * 50)

target_ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

with torch.no_grad():
    for R in target_ranks:
        params = R * (W_g.size(0) + W_g.size(1) + 1)  # U + Vh + S
        comp = W_g.numel() / params
        
        var_ratio = var_cum[R-1].item() / var_total
        
        # Reconstruir
        W_r = (U[:, :R] * S[:R]) @ Vh[:R, :]
        
        gate_r = F.silu(x @ W_r.T)
        cos = F.cosine_similarity(gate_r.reshape(-1, 14336), gate_teacher.reshape(-1, 14336), dim=1).mean().item()
        
        print(f"{R:>6} {params:>10,} {comp:>10.0f}x {var_ratio:>9.4f} {cos:>10.4f}")
        
        if cos > 0.95:
            print(f"\n  ✅ R={R}: COS_gate={cos:.4f} a {comp:.0f}x compresion")
            break

# ─── Mejor SVD con seed entrenable ───
print("\n" + "="*60)
print("SVD entrenable: factorización U @ S @ Vh")
print("="*60)

for target_R in [4, 8, 16, 32]:
    # Matrices SVD como parámetros entrenables
    U_ = torch.nn.Parameter(U[:, :target_R].clone() * math.sqrt(S[:target_R].mean()))
    Vh_ = torch.nn.Parameter(Vh[:target_R, :].clone() * math.sqrt(S[:target_R].mean()))
    
    opt = torch.optim.Adam([U_, Vh_], lr=1e-3)
    
    for step in range(100):
        opt.zero_grad()
        W = U_ @ Vh_
        loss = F.mse_loss(W, W_g.cuda())
        loss.backward()
        opt.step()
    
    with torch.no_grad():
        gate_r = F.silu(x @ (U_ @ Vh_).T)
        cos = F.cosine_similarity(gate_r.reshape(-1, 14336), gate_teacher.reshape(-1, 14336), dim=1).mean().item()
        wmse = F.mse_loss(U_ @ Vh_, W_g.cuda()).item()
        r2 = 1 - wmse / W_g.var().item()
    
    params = target_R * (W_g.size(0) + W_g.size(1))
    comp = W_g.numel() / params
    print(f"  R={target_R:2d}: COS={cos:.4f}  R²={r2:.4f}  wMSE={wmse:.8f}  ({comp:.0f}x)")
