"""
Mini-test: destilacion del gate projection SOLO.

3 FractalLinears simultáneos saturan la GTX 1650.
Probamos solo el gate: comparar weight MSE vs gate-activation COS.
"""

import torch, torch.nn.functional as F, time
from safetensors import safe_open
from sg_hf.core import FractalLinear

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo: {device}")

# Cargar SOLO gate projection
CONSOLIDATED = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1\consolidated.safetensors'
with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
    W_g_t = f.get_tensor('layers.0.feed_forward.w1.weight').float()
W_g_t = W_g_t.to(device)
print(f"W_g: {tuple(W_g_t.shape)} std={W_g_t.std():.4f}")

# Input realista (std=1.0 como despues de RMSNorm)
torch.manual_seed(42)
x_eval = torch.randn(1, 16, 4096, device=device)  # std=1

with torch.no_grad():
    gate_teacher = F.silu(x_eval @ W_g_t.T)
print(f"Gate teacher output: std={gate_teacher.std():.4f}")

# Input de entrenamiento (1 batch chico)
torch.manual_seed(123)
x_train = torch.randn(1, 16, 4096, device=device)

# Teacher gate para entrenamiento
with torch.no_grad():
    gate_train_t = F.silu(x_train @ W_g_t.T)

COMPRESSION = 100.0

# ═══ TEST A: Weight MSE (baseline) ═══
print("\n" + "="*50)
print("A: WEIGHT MSE")
print("="*50)

fl_a = FractalLinear(4096, 14336, compression=COMPRESSION, kronecker_rank=1).to(device)
fl_a.initialize_from_teacher(W_g_t.cpu())
opt_a = torch.optim.Adam(fl_a.parameters(), lr=1e-2)

t0 = time.time()
for step in range(200):
    opt_a.zero_grad()
    W = fl_a._generate_weight()
    loss = F.mse_loss(W, W_g_t)
    loss.backward()
    opt_a.step()
    if step % 50 == 0:
        with torch.no_grad():
            gate_s = F.silu(x_eval @ fl_a._generate_weight().T)
            cos_g = F.cosine_similarity(gate_s.reshape(-1, 14336), gate_teacher.reshape(-1, 14336), dim=1).mean().item()
        print(f"  step {step}: wMSE={loss.item():.8f}  COS_gate={cos_g:.4f}")

with torch.no_grad():
    gate_a = F.silu(x_eval @ fl_a._generate_weight().T)
    cos_a = F.cosine_similarity(gate_a.reshape(-1, 14336), gate_teacher.reshape(-1, 14336), dim=1).mean().item()
    wmse_a = F.mse_loss(fl_a._generate_weight(), W_g_t).item()
dt_a = time.time() - t0
print(f"\n  RESULTADO A: COS_gate={cos_a:.4f}  wMSE={wmse_a:.8f} ({dt_a:.0f}s)")

# ═══ TEST B: Gate COS distillation ═══
print("\n" + "="*50)
print("B: GATE COS DISTILLATION")
print("="*50)

fl_b = FractalLinear(4096, 14336, compression=COMPRESSION, kronecker_rank=1).to(device)
fl_b.initialize_from_teacher(W_g_t.cpu())
opt_b = torch.optim.Adam(fl_b.parameters(), lr=1e-2)

t0 = time.time()
for step in range(200):
    opt_b.zero_grad()
    W = fl_b._generate_weight()
    gate_s = F.silu(x_train @ W.T)
    cos = F.cosine_similarity(gate_s.reshape(-1, 14336), gate_train_t.reshape(-1, 14336), dim=1)
    loss = (1 - cos).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(fl_b.parameters(), 1.0)
    opt_b.step()
    if step % 50 == 0:
        with torch.no_grad():
            gate_s2 = F.silu(x_eval @ fl_b._generate_weight().T)
            cos_g = F.cosine_similarity(gate_s2.reshape(-1, 14336), gate_teacher.reshape(-1, 14336), dim=1).mean().item()
            wmse = F.mse_loss(fl_b._generate_weight(), W_g_t).item()
        print(f"  step {step}: COS_gate={cos_g:.4f}  wMSE={wmse:.8f}")

with torch.no_grad():
    gate_b = F.silu(x_eval @ fl_b._generate_weight().T)
    cos_b = F.cosine_similarity(gate_b.reshape(-1, 14336), gate_teacher.reshape(-1, 14336), dim=1).mean().item()
    wmse_b = F.mse_loss(fl_b._generate_weight(), W_g_t).item()
dt_b = time.time() - t0
print(f"\n  RESULTADO B: COS_gate={cos_b:.4f}  wMSE={wmse_b:.8f} ({dt_b:.0f}s)")

# ═══ TEST C: Weight MSE + Gate COS ═══
print("\n" + "="*50)
print("C: HIBRIDO (Weight MSE + Gate COS)")
print("="*50)

fl_c = FractalLinear(4096, 14336, compression=COMPRESSION, kronecker_rank=1).to(device)
fl_c.initialize_from_teacher(W_g_t.cpu())
opt_c = torch.optim.Adam(fl_c.parameters(), lr=1e-2)

t0 = time.time()
for step in range(200):
    opt_c.zero_grad()
    W = fl_c._generate_weight()
    gate_s = F.silu(x_train @ W.T)
    cos = F.cosine_similarity(gate_s.reshape(-1, 14336), gate_train_t.reshape(-1, 14336), dim=1)
    wmse = F.mse_loss(W, W_g_t)
    loss = wmse * 100 + (1 - cos).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(fl_c.parameters(), 1.0)
    opt_c.step()
    if step % 50 == 0:
        with torch.no_grad():
            gate_s2 = F.silu(x_eval @ fl_c._generate_weight().T)
            cos_g = F.cosine_similarity(gate_s2.reshape(-1, 14336), gate_teacher.reshape(-1, 14336), dim=1).mean().item()
        print(f"  step {step}: COS_gate={cos_g:.4f}")

with torch.no_grad():
    gate_c = F.silu(x_eval @ fl_c._generate_weight().T)
    cos_c = F.cosine_similarity(gate_c.reshape(-1, 14336), gate_teacher.reshape(-1, 14336), dim=1).mean().item()
    wmse_c = F.mse_loss(fl_c._generate_weight(), W_g_t).item()
dt_c = time.time() - t0
print(f"\n  RESULTADO C: COS_gate={cos_c:.4f}  wMSE={wmse_c:.8f} ({dt_c:.0f}s)")

# ═══ RESUMEN ═══
print("\n" + "="*50)
print("RESUMEN")
print("="*50)
print(f"  A) Weight MSE:          COS_gate={cos_a:.4f}  wMSE={wmse_a:.8f}")
print(f"  B) Gate COS distill:    COS_gate={cos_b:.4f}  wMSE={wmse_b:.8f}")
print(f"  C) Hibrido:             COS_gate={cos_c:.4f}  wMSE={wmse_c:.8f}")
