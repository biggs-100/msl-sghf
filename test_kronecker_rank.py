"""
Test: Kronecker rank r=1 vs r=4 en Mistral-7B layer 0.

Hipótesis: r=1 fuerza colinealidad entre filas del mismo bloque,
lo cual destruye el gate SiLU (necesita direcciones independientes).
r=4 da 4 direcciones por bloque → gate funcional.

Mide:
  - Weight MSE (gate, up, down)
  - MLP output COS (lo que realmente importa)
"""

import torch, torch.nn.functional as F, math, time
from safetensors import safe_open
from sg_hf.core import FractalLinear

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo: {device}")

CONSOLIDATED = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1\consolidated.safetensors'

# Cargar teacher layer 0
print("\n>>> Cargando teacher Mistral-7B layer 0...")
with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
    W_g_t = f.get_tensor('layers.0.feed_forward.w1.weight').float()  # (14336, 4096)
    W_u_t = f.get_tensor('layers.0.feed_forward.w3.weight').float()  # (14336, 4096)
    W_d_t = f.get_tensor('layers.0.feed_forward.w2.weight').float()  # (4096, 14336)

COMPRESSION = 100.0

def make_and_train(name, in_f, out_f, rank, teacher_W, steps=300, lr=1e-2):
    """Crea FractalLinear con kronecker_rank, entrena para minimizar weight MSE."""
    fl = FractalLinear(in_f, out_f, compression=COMPRESSION, kronecker_rank=rank)
    
    # Inicializar desde teacher
    fl.initialize_from_teacher(teacher_W)
    
    # Mover a GPU
    teacher_W = teacher_W.to(device)
    fl = fl.to(device)
    
    opt = torch.optim.Adam(list(fl.parameters()), lr=lr)
    
    t0 = time.time()
    best_mse = float('inf')
    for step in range(steps):
        opt.zero_grad()
        W = fl._generate_weight()
        loss = F.mse_loss(W, teacher_W)
        loss.backward()
        opt.step()
        if loss.item() < best_mse:
            best_mse = loss.item()
        if step % 50 == 0:
            print(f"    {name} step {step:3d}: MSE={loss.item():.8f}")
    
    dt = time.time() - t0
    
    # Evaluación final
    with torch.no_grad():
        W_final = fl._generate_weight()
        final_mse = F.mse_loss(W_final, teacher_W).item()
    
    return fl, final_mse, dt

# ─── Test 1: Gate projection r=1 vs r=4 ───
print("\n" + "="*60)
print("TEST 1: Gate projection (14336×4096)")
print("="*60)

fl_g_r1, mse_g_r1, t1 = make_and_train(
    "Gate r=1", 4096, 14336, 1, W_g_t.clone(), steps=300)

fl_g_r4, mse_g_r4, t4 = make_and_train(
    "Gate r=4", 4096, 14336, 4, W_g_t.clone(), steps=300)

print(f"\n  Gate r=1: MSE={mse_g_r1:.8f}  ({t1:.0f}s)")
print(f"  Gate r=4: MSE={mse_g_r4:.8f}  ({t4:.0f}s)")
print(f"  Mejora:   {mse_g_r1/mse_g_r4:.1f}x")

# ─── Test 2: Up projection r=1 vs r=4 ───
print("\n" + "="*60)
print("TEST 2: Up projection (14336×4096)")
print("="*60)

fl_u_r1, mse_u_r1, _ = make_and_train(
    "Up r=1", 4096, 14336, 1, W_u_t.clone(), steps=300)

fl_u_r4, mse_u_r4, _ = make_and_train(
    "Up r=4", 4096, 14336, 4, W_u_t.clone(), steps=300)

print(f"\n  Up r=1: MSE={mse_u_r1:.8f}")
print(f"  Up r=4: MSE={mse_u_r4:.8f}")
print(f"  Mejora:  {mse_u_r1/mse_u_r4:.1f}x")

# ─── Test 3: Down projection r=1 vs r=4 ───
print("\n" + "="*60)
print("TEST 3: Down projection (4096×14336)")
print("="*60)

fl_d_r1, mse_d_r1, _ = make_and_train(
    "Down r=1", 14336, 4096, 1, W_d_t.clone(), steps=300)

fl_d_r4, mse_d_r4, _ = make_and_train(
    "Down r=4", 14336, 4096, 4, W_d_t.clone(), steps=300)

print(f"\n  Down r=1: MSE={mse_d_r1:.8f}")
print(f"  Down r=4: MSE={mse_d_r4:.8f}")
print(f"  Mejora:   {mse_d_r1/mse_d_r4:.1f}x")

# ─── Test 4: MLP output COS ───
print("\n" + "="*60)
print("TEST 4: MLP output COS (lo que importa)")
print("="*60)

# Input de prueba realista
torch.manual_seed(42)
x = torch.randn(2, 64, 4096, device=device) * 0.1

configs = [
    ("r=1 gates + r=1 down", fl_g_r1, fl_u_r1, fl_d_r1),
    ("r=4 gates + r=1 down", fl_g_r4, fl_u_r4, fl_d_r1),
    ("r=4 gates + r=4 down", fl_g_r4, fl_u_r4, fl_d_r4),
]

for name, fg, fu, fd in configs:
    with torch.no_grad():
        W_g = fg._generate_weight()
        W_u = fu._generate_weight()
        W_d = fd._generate_weight()
        
        # Teacher MLP
        gate_t = F.silu(x @ W_g_t.to(device).T)
        up_t = x @ W_u_t.to(device).T
        y_t = (gate_t * up_t) @ W_d_t.to(device).T
        
        # Fractal MLP
        gate_s = F.silu(x @ W_g.T)
        up_s = x @ W_u.T
        y_s = (gate_s * up_s) @ W_d.T
        
        cos = F.cosine_similarity(y_s.view(-1, 4096), y_t.view(-1, 4096), dim=1).mean().item()
        mse_out = F.mse_loss(y_s, y_t).item()
        
        # Gate COS también (para diagnosticar)
        cos_g = F.cosine_similarity(gate_s.view(-1, 14336), gate_t.view(-1, 14336), dim=1).mean().item()
        
        print(f"\n  {name}:")
        print(f"    Output COS = {cos:.4f}")
        print(f"    Output MSE = {mse_out:.6f}")
        print(f"    Gate COS   = {cos_g:.4f}")

# ─── Resumen ───
print("\n" + "="*60)
print("RESUMEN")
print("="*60)
print(f"\nWeight MSE (gate):   r=1={mse_g_r1:.8f}  r=4={mse_g_r4:.8f}  ({mse_g_r1/mse_g_r4:.1f}x)")
print(f"Weight MSE (up):     r=1={mse_u_r1:.8f}  r=4={mse_u_r4:.8f}  ({mse_u_r1/mse_u_r4:.1f}x)")  
print(f"Weight MSE (down):   r=1={mse_d_r1:.8f}  r=4={mse_d_r4:.8f}  ({mse_d_r1/mse_d_r4:.1f}x)")
print(f"\nTiempo por capa (gate): r=1={t1:.0f}s  r=4={t4:.0f}s")
