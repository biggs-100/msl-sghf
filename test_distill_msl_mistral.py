"""
Pipeline completo: destilar Mistral-7B -> MSL -> comprimir con SG-HF semillas.

1. Toma layer 0 de Mistral-7B
2. Crea MSL factors (U, s, V) para gate/up/down
3. ENTRENA los factores para imitar el output del teacher
4. Verifica que los factores tienen std grande (para Kronecker)
5. Aplica Kronecker sobre los factores entrenados
6. Trunca escalas -> alumno sin fine-tuning
"""

import torch, torch.nn as nn, torch.nn.functional as F, gc, sys, time, math
sys.path.insert(0, '.')
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# ─── 1. Cargar Mistral ───────────────────────────────────────────
print("Cargando Mistral-7B...")
model = AutoModelForCausalLM.from_pretrained(
    'mistralai/Mistral-7B-v0.3', dtype=torch.float16, device_map='cpu', low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.3')
print("  OK")

# ─── 2. MLP del teacher ─────────────────────────────────────────
mlp = model.model.layers[0].mlp
print(f"  MLP: gate {list(mlp.gate_proj.weight.shape)}, up {list(mlp.up_proj.weight.shape)}, down {list(mlp.down_proj.weight.shape)}")
print(f"  Teacher weights std: gate={mlp.gate_proj.weight.data.std():.4f}, up={mlp.up_proj.weight.data.std():.4f}, down={mlp.down_proj.weight.data.std():.4f}")

# ─── 3. Crear MSL y entrenar ─────────────────────────────────────
print("\nCreando MSL factors para gate/up/down...")
# Ahora usamos la config de escalas. Para simplificar, rank fijo por capa.
# gate_proj: (14336, 4096) -> rank 224
# up_proj: (14336, 4096) -> rank 224 
# down_proj: (4096, 14336) -> rank 448

class MSLFactorLayer(nn.Module):
    """Una proyeccion lineal como U diag(s) V, factores entrenables."""
    def __init__(self, M, N, rank):
        super().__init__()
        self.M, self.N, self.R = M, N, rank
        self.U = nn.Parameter(torch.randn(M, rank) * 0.1)
        self.s = nn.Parameter(torch.randn(rank) * 0.5)
        self.V = nn.Parameter(torch.randn(rank, N) * 0.1)
    
    def forward(self, x):
        # y = x @ V^T * s @ U^T
        xv = x @ self.V.T  # (B, T, rank)
        xv = xv * self.s   # (B, T, rank)
        return xv @ self.U.T  # (B, T, M)
    
    def get_W(self):
        return self.U @ torch.diag(self.s) @ self.V

class MSL_MLP(nn.Module):
    """MLP (SwiGLU) con factores MSL entrenables."""
    def __init__(self, hidden=4096, intermediate=14336, rank_gate=224, rank_down=448):
        super().__init__()
        self.gate = MSLFactorLayer(hidden, intermediate, rank_gate)  # atencion: (hidden, intermediate) o viceversa?
        # Mistral: gate_proj.weight es (intermediate, hidden) -> entrada hidden, salida intermediate
        # MSL: y = x @ V^T @ diag(s) @ U^T, donde V es (rank, N), U es (M, rank)
        # Para gate_proj: M=intermediate, N=hidden
        self.gate = MSLFactorLayer(intermediate, hidden, rank_gate)
        self.up = MSLFactorLayer(intermediate, hidden, rank_gate)
        self.down = MSLFactorLayer(hidden, intermediate, rank_down)
    
    def forward(self, x):
        gate = F.silu(self.gate(x))
        up = self.up(x)
        return self.down(gate * up)

msl_mlp = MSL_MLP().to(DEVICE)  # MSL en FP32 para entrenamiento estable
print(f"  Parametros MSL: {sum(p.numel() for p in msl_mlp.parameters()):,}")
print(f"  vs teacher MLP: {sum(p.numel() for p in mlp.parameters()):,}")
print(f"  Compresion teorica: {sum(p.numel() for p in mlp.parameters()) / sum(p.numel() for p in msl_mlp.parameters()):.1f}x")

# ─── 4. Cachear outputs del teacher ──────────────────────────────
print(f"\nCacheando pares (input, teacher_output)...")
torch.manual_seed(42)
n_train = 1000

t_cache = time.time()
train_x = torch.randn(n_train, 4096, dtype=torch.float16) * 1.5
with torch.no_grad():
    mlp = mlp.to(dtype=torch.float16)  # asegurar dtype
    train_y = mlp(train_x)  # teacher en CPU
print(f"  Cache listo: {n_train} pares (x std={train_x.std():.4f}, y std={train_y.std():.4f})")
print(f"  Tiempo teacher: {time.time()-t_cache:.1f}s")

# Mover a GPU en FP32 (MSL entrena en FP32)
train_x_gpu = train_x.to(device=DEVICE, dtype=torch.float32)
train_y_gpu = train_y.to(device=DEVICE, dtype=torch.float32)

# ─── 5. Entrenar MSL ─────────────────────────────────────────---
print("\nEntrenando MSL para imitar teacher...")
opt = torch.optim.AdamW(msl_mlp.parameters(), lr=1e-3, weight_decay=1e-5)
steps = 500
batch_size = 64
print_step = max(1, steps // 10)

t1 = time.time()
for step in range(1, steps + 1):
    idx = torch.randint(0, n_train, (batch_size,))
    x = train_x_gpu[idx]
    target = train_y_gpu[idx]
    
    out = msl_mlp(x)
    
    mse_loss = F.mse_loss(out, target)
    cos = (out.flatten() * target.flatten()).sum() / (out.norm() * target.norm())
    cos_loss = 1.0 - cos
    loss = mse_loss + 0.5 * cos_loss
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(msl_mlp.parameters(), 1.0)
    opt.step()
    
    if step == 1 or step % print_step == 0:
        print(f"  Step {step:4d}/{steps}  |  MSE: {mse_loss.item():.8f}  |  COS output: {cos.item():.4f}")

elapsed = time.time() - t1
print(f"  Entrenamiento: {elapsed:.1f}s ({elapsed/steps*1000:.1f}ms/step)")

# ─── 5. Verificar std de los factores entrenados ─────────────────
print("\nVerificando std de factores entrenados:")
for name, layer in [('gate', msl_mlp.gate), ('up', msl_mlp.up), ('down', msl_mlp.down)]:
    print(f"  {name}: U std={layer.U.std():.4f}, s std={layer.s.std():.4f}, V std={layer.V.std():.4f}")

# ─── 6. Medir COS final ──────────────────────────────────────────
with torch.no_grad():
    x_test = torch.randn(200, 4096, dtype=torch.float16) * 1.5
    target_test = mlp(x_test).float()  # teacher en CPU, pasar a FP32
    out_test = msl_mlp(x_test.to(device=DEVICE, dtype=torch.float32))
    cos_final = (out_test.flatten() * target_test.flatten().to(DEVICE)).sum() / (out_test.norm() * target_test.norm().to(DEVICE))
    mse_final = F.mse_loss(out_test, target_test.to(DEVICE)).item()
    print(f"\nResultado final MSL ENTRENADO:")
    print(f"  COS output: {cos_final:.4f}")
    print(f"  MSE: {mse_final:.8f}")
    print(f"  Compresion: {sum(p.numel() for p in mlp.parameters()) / sum(p.numel() for p in msl_mlp.parameters()):.1f}x")
    print(f"  vs SVD directo (sin entrenar): COS ~0.52")

# ─── 7. Verificar std para Kronecker ─────────────────────────────
print(f"\nVerificando si Kronecker seria viable:")
for name, layer in [('gate', msl_mlp.gate), ('up', msl_mlp.up), ('down', msl_mlp.down)]:
    ok = "SI" if layer.U.std() > 0.1 else "NO"
    ok_v = "SI" if layer.V.std() > 0.1 else "NO"
    print(f"  {name}: U std={layer.U.std():.4f} (Kronecker? {ok}) | s std={layer.s.std():.4f} | V std={layer.V.std():.4f} (Kronecker? {ok_v})")

print(f"\n  Si std > 0.1: Kronecker funciona (como en FractalLinear con GPT-2)")
print(f"  Si std < 0.01: Kronecker falla (como en Mistral-7B gate)")

del model, mlp, msl_mlp
gc.collect()
torch.cuda.empty_cache()
print("\nOK")
