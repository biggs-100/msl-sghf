"""
Test rapido: SVD batch de todos los bloques → inicialización para Direct Kronecker.

586K bloques de 19x6 en GPU: ¿cuanto tarda torch.linalg.svd batch?
"""

import torch, math, time
from safetensors import safe_open

device = 'cuda'
print(f"Dispositivo: {device}")

CONSOLIDATED = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1\consolidated.safetensors'
with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
    W = f.get_tensor('layers.0.feed_forward.w1.weight').float()

# Bloques
out_f, in_f = 14336, 4096
p = 766; q = 766; a = 19; b = 6

# Solo bloques completos (sin bordes parciales)
p_full = out_f // a  # 754
q_full = in_f // b   # 682
W_full = W[:p_full*a, :q_full*b]  # multiplo exacto
print(f"Bloques completos: {p_full}x{q_full} = {p_full*q_full} bloques de {a}x{b}")

# Reshape a bloques
t0 = time.time()
blocks = W_full.reshape(p_full, a, q_full, b).permute(0, 2, 1, 3)  # (p, q, a, b)
blocks_2d = blocks.reshape(p_full * q_full, a, b).cuda()  # (514K, 19, 6)
print(f"Reshape: {time.time()-t0:.1f}s")
print(f"Bloques: {blocks_2d.shape}")

# SVD batch
t0 = time.time()
U, S, Vh = torch.linalg.svd(blocks_2d, full_matrices=False)
print(f"SVD batch: {time.time()-t0:.1f}s")
print(f"U: {U.shape}, S: {S.shape}, Vh: {Vh.shape}")

# Verificar reconstruccion rank-1, 2, 4
for R in [1, 2, 4]:
    W_r = (U[:, :, :R] * S[:, :R].unsqueeze(1)) @ Vh[:, :R, :]  # (586K, 19, 6)
    W_r = W_r.reshape(p, q, a, b).permute(0, 2, 1, 3).reshape(p*a, q*b)
    W_r = W_r[:out_f, :in_f]  # recortar al tamaño exacto
    
    mse = (W_r - W.cuda()).pow(2).mean().item()
    var_w = W.var().item()
    r2 = 1 - mse / var_w
    print(f"  R={R}: MSE={mse:.8f}  R²={r2:.4f}")

# Cuanto tardaria inicializar DirectKroneckerSum desde SVD
R = 4
t0 = time.time()
row_basis = torch.zeros(R, p, a, device=device)
col_basis = torch.zeros(R, q, b, device=device)

for r in range(R):
    # U: (586K, 19, 6) → tomar componente r → (586K, 19)
    # Reorganizar a (p, q, 19) y promediar sobre q
    Ur = U[:, :, r].reshape(p, q, a).mean(dim=1)  # (p, a)
    Vhr = Vh[:, r, :].reshape(p, q, b).mean(dim=0)  # (q, b)
    sr = S[:, r].reshape(p, q).mean()  # scalar promedio
    
    row_basis[r] = Ur * math.sqrt(sr)
    col_basis[r] = Vhr * math.sqrt(sr)

print(f"\nInit DirectKronecker R=4: {time.time()-t0:.3f}s")
print(f"row_basis: {row_basis.shape} std={row_basis.std():.4f}")
print(f"col_basis: {col_basis.shape} std={col_basis.std():.4f}")

# Verificar que la inicializacion da buena reconstruccion
W_init = (row_basis.reshape(R, p*a) * math.sqrt(p*q / R)).T @ (col_basis.reshape(R, q*b) * math.sqrt(p*q / R))
W_init = W_init[:out_f, :in_f]

# Escalar para que coincida con el teacher
scale = W.std().item() / W_init.std().item()
W_init = W_init * scale

mse_init = (W_init - W.cuda()).pow(2).mean().item()
r2_init = 1 - mse_init / W.var().item()
print(f"  Reconstruccion init: MSE={mse_init:.8f}  R²={r2_init:.4f}")
