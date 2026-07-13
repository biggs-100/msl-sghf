"""
Prueba SharedSeedMLP con Mistral-7B.

Un seed compartido para gate y up → errores correlacionados → COS > 0.9.
Ejecuta UNA capa como prueba (5 min).
"""

import torch, torch.nn.functional as F
from safetensors import safe_open
from sg_hf.core import SharedSeedMLP
import os, sys

device = 'cuda'
CONSOLIDATED = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1\consolidated.safetensors'

# Cargar teacher weights de capa 0
print(">>> Cargando teacher Mistral-7B layer 0...")
with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
    W_g = f.get_tensor('layers.0.feed_forward.w1.weight').float()  # (14336, 4096)
    W_u = f.get_tensor('layers.0.feed_forward.w3.weight').float()  # (14336, 4096)
    W_d = f.get_tensor('layers.0.feed_forward.w2.weight').float()  # (4096, 14336)

# Crear SharedSeedMLP
print(">>> Creando SharedSeedMLP...")
mlp = SharedSeedMLP(hidden_size=4096, intermediate_size=14336, compression=100.0)

# Inicializar desde teacher
print(">>> Inicializando desde teacher...")
with torch.no_grad():
    # Inicializar seed con promedios
    for i in range(mlp.p):
        for j in range(mlp.q):
            rs, re = i * mlp.a, min(i * mlp.a + mlp.a, 14336)
            cs, ce = j * mlp.b, min(j * mlp.b + mlp.b, 4096)
            if rs >= 14336 or cs >= 4096:
                mlp.seed[i,j] = 0; continue
            g_block = W_g[rs:re, cs:ce].mean().item()
            u_block = W_u[rs:re, cs:ce].mean().item()
            mlp.seed[i,j] = (g_block + u_block) / 2
    
    # Estandarizar seed para FFT estable
    s = mlp.seed.data.std()
    if s > 0:
        mlp.seed.data = mlp.seed.data / s * 0.1
    
    # Inicializar bases
    for name in ['row_gate', 'col_gate', 'row_up', 'col_up']:
        getattr(mlp, name).data = torch.randn_like(getattr(mlp, name)) * 0.1
    
    # Inicializar down projection
    mlp.down.initialize_from_teacher(W_d)

mlp.to(device)
W_g, W_u = W_g.to(device), W_u.to(device)
W_d = W_d.to(device)

# Optimizar seed compartido + bases gate/up
print(">>> Optimizando...")
params = [mlp.seed, mlp.freq_scale, mlp.freq_shift,
          mlp.row_gate, mlp.col_gate, mlp.row_up, mlp.col_up]
opt = torch.optim.Adam(params, lr=1e-2)

for step in range(300):
    opt.zero_grad()
    W_g_s, W_u_s = mlp._generate_gate_up()
    W_d_s = mlp.down._generate_weight()
    
    loss = F.mse_loss(W_g_s, W_g) + F.mse_loss(W_u_s, W_u) + F.mse_loss(W_d_s, W_d)
    loss.backward()
    opt.step()
    
    if step % 50 == 0:
        print(f"  Step {step}: loss={loss.item():.8f}")

# Forward MLP con input realista
print(">>> Verificando output del MLP...")
x = torch.randn(2, 64, 4096, device=device) * 0.1

with torch.no_grad():
    # Teacher
    gate_t = F.silu(x @ W_g.T)
    up_t = x @ W_u.T
    y_t = (gate_t * up_t) @ W_d.T
    
    # Fractal (SharedSeedMLP)
    y_s = mlp(x)

cos = F.cosine_similarity(y_s.view(-1,4096), y_t.view(-1,4096), dim=1).mean().item()
mse = F.mse_loss(y_s, y_t).item()
print(f"\n>>> RESULTADO: COS={cos:.4f} MSE={mse:.6f}")
print(f">>> {'✅ FUNCIONA' if cos > 0.5 else '❌ REVISAR'}")
