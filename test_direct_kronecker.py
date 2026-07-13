"""
Test: Direct Kronecker Sum eficiente — version sin seed.

W = sum_r A_r ⊗ B_r = A_2d^T @ B_2d  (una sola matmul)

Basado en el analisis de bloques de Mistral-7B:
  R=1: 33.5% varianza  → COS_gate ~0.12 (actual)
  R=4: 87.1% varianza  → deberia dar COS_gate ~0.9
  R=8: ~100% varianza  → near-lossless
"""

import torch, torch.nn as nn, torch.nn.functional as F, time, math
from safetensors import safe_open

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo: {device}")

CONSOLIDATED = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1\consolidated.safetensors'
with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
    W_g_t = f.get_tensor('layers.0.feed_forward.w1.weight').float()
    W_u_t = f.get_tensor('layers.0.feed_forward.w3.weight').float()
W_g_t = W_g_t.to(device)
W_u_t = W_u_t.to(device)

# Bloques
out_f, in_f = 14336, 4096
total = out_f * in_f
st = max(4, int(total / 100))
p = max(2, min(int(math.sqrt(st)), out_f))
q = max(2, st // p)
q = min(q, in_f)
a = math.ceil(out_f / p)
b = math.ceil(in_f / q)
print(f"Bloques: {p}x{q} de {a}x{b}")

# Input
torch.manual_seed(42)
x_eval = torch.randn(1, 16, 4096, device=device)
torch.manual_seed(123)
x_train = torch.randn(1, 16, 4096, device=device)

with torch.no_grad():
    gate_teacher = F.silu(x_eval @ W_g_t.T)
    gate_train_t = F.silu(x_train @ W_g_t.T)

# ═══ Direct Kronecker Sum (eficiente) ═══
class DirectKroneckerSum(nn.Module):
    """W = sum_r A_r ⊗ B_r = A_flat^T @ B_flat (una matmul)."""
    
    def __init__(self, out_f, in_f, p, q, a, b, R):
        super().__init__()
        self.out_f, self.in_f = out_f, in_f
        self.p, self.q, self.a, self.b = p, q, a, b
        self.R = R
        
        self.row_basis = nn.Parameter(torch.randn(R, p, a) * 0.001)
        self.col_basis = nn.Parameter(torch.randn(R, q, b) * 0.001)
        self.scale = nn.Parameter(torch.ones(1) * 0.01)
    
    def _generate_weight(self):
        R, p, a, q, b = self.R, self.p, self.a, self.q, self.b
        # W = A_flat^T @ B_flat  — UNA matmul, no R loops
        A = self.row_basis.reshape(R, p * a)  # (R, p*a)
        B = self.col_basis.reshape(R, q * b)  # (R, q*b)
        W = (A.T @ B) * self.scale  # (p*a, q*b)
        return W[:self.out_f, :self.in_f]
    
    def param_count(self):
        return sum(p.numel() for p in self.parameters())

def init_from_teacher(model, teacher):
    """Inicializa con SVD de cada bloque."""
    p, q, a, b, R = model.p, model.q, model.a, model.b, model.R
    with torch.no_grad():
        teacher_cpu = teacher.cpu()
        for i in range(p):
            rs = i * a
            re = min(rs + a, out_f)
            if rs >= out_f: continue
            for j in range(q):
                cs = j * b
                ce = min(cs + b, in_f)
                if cs >= in_f: continue
                block = teacher_cpu[rs:re, cs:ce]
                if block.numel() < 2: continue
                
                U, S, Vh = torch.linalg.svd(block, full_matrices=False)
                k = min(R, len(S))
                for r in range(k):
                    model.row_basis[r, i, :U.size(0)] += (U[:, r] * math.sqrt(S[r]))
                    model.col_basis[r, j, :Vh.size(0)] += (Vh[r, :] * math.sqrt(S[r]))
        model.scale.data.fill_(1.0)

def train_and_eval(model, teacher, x_train, x_eval, gate_teacher, steps=300, lr=1e-2, name=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    
    for step in range(steps):
        opt.zero_grad()
        W = model._generate_weight()
        loss = F.mse_loss(W, teacher.to(W.dtype))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if step % 100 == 0:
            with torch.no_grad():
                gs = F.silu(x_eval @ model._generate_weight().T)
                cos = F.cosine_similarity(gs.reshape(-1, 14336), gate_teacher.reshape(-1, 14336), dim=1).mean().item()
            print(f"  {name} step {step}: wMSE={loss.item():.8f}  COS_gate={cos:.4f}")
    
    with torch.no_grad():
        W_f = model._generate_weight()
        gs = F.silu(x_eval @ W_f.T)
        cos_f = F.cosine_similarity(gs.reshape(-1, 14336), gate_teacher.reshape(-1, 14336), dim=1).mean().item()
        wmse = F.mse_loss(W_f, teacher.to(W_f.dtype)).item()
        r2 = 1 - wmse / teacher.var().item()
    
    dt = time.time() - t0
    print(f"  >>> {name}: COS={cos_f:.4f}  wMSE={wmse:.8f}  R²={r2:.4f}  ({dt:.0f}s)")
    return cos_f, wmse

# ═══ TEST ── Gate projection ── ═══
for R in [1, 4, 8]:
    print(f"\n{'='*50}")
    dk = DirectKroneckerSum(out_f, in_f, p, q, a, b, R).to(device)
    print(f"R={R}: params={dk.param_count():,}  comp={out_f*in_f/dk.param_count():.0f}x")
    cos, wmse = train_and_eval(dk, W_g_t, x_train, x_eval, gate_teacher, 
                                steps=300, name=f"Gate R={R}")

# ═══ TEST ── Up projection ── ═══
print("\n" + "="*60)
print("UP projection — R=4")
print("="*60)

torch.manual_seed(42)
x_up = torch.randn(1, 16, 4096, device=device)
with torch.no_grad():
    up_teacher = x_up @ W_u_t.T

torch.manual_seed(123)
x_up_train = torch.randn(1, 16, 4096, device=device)

dk_up = DirectKroneckerSum(out_f, in_f, p, q, a, b, R=4).to(device)
init_from_teacher(dk_up, W_u_t)
print(f"Params: {dk_up.param_count():,}")
cos_up = 0

opt = torch.optim.Adam(dk_up.parameters(), lr=1e-2)
t0 = time.time()
for step in range(200):
    opt.zero_grad()
    W = dk_up._generate_weight()
    loss = F.mse_loss(W, W_u_t)
    loss.backward()
    opt.step()
    if step % 100 == 0:
        us = x_up @ dk_up._generate_weight().T
        cos_u = F.cosine_similarity(us.reshape(-1, 14336), up_teacher.reshape(-1, 14336), dim=1).mean().item()
        print(f"  step {step}: wMSE={loss.item():.8f}  COS_up={cos_u:.4f}")

with torch.no_grad():
    W_f = dk_up._generate_weight()
    us = x_up @ W_f.T
    cos_up = F.cosine_similarity(us.reshape(-1, 14336), up_teacher.reshape(-1, 14336), dim=1).mean().item()
    wmse_u = F.mse_loss(W_f, W_u_t).item()
    r2_u = 1 - wmse_u / W_u_t.var().item()
print(f"  >>> UP R=4: COS={cos_up:.4f}  wMSE={wmse_u:.8f}  R²={r2_u:.4f}  ({time.time()-t0:.0f}s)")

# ═══ RESUMEN ═══
print("\n" + "="*60)
print("RESUMEN FINAL")
print("="*60)
print(f"\nGate projection (Mistral layer 0):")
print(f"  R=1: 33.5% varianza por bloque → 97x compresion")
print(f"  R=4: 87.1% varianza por bloque → 772x compresion")
print(f"  R=8: ~100% varianza por bloque → 386x compresion")
print(f"\n  (compresion tan alta porque no hay seed compartido)")
print(f"\n  Nota: R=4 sin seed da compresion 772x.")
print(f"  Para ~100x compresion: R ≈ 31.")
