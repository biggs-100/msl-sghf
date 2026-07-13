"""
Diagnóstico: ¿cuánto ruido en los pesos mata el output COS?

El MSE de ~0.00001 equivale a RMSE ≈ 0.003.
Si agregamos ruido Gaussiano con std=0.003 a los pesos del teacher,
¿el COS del MLP también colapsa a ~0?

Si SÍ → el problema es la sensibilidad inherente del SiLU gate
Si NO → el problema es específico de la aproximación Kronecker
"""

import torch, torch.nn.functional as F
from safetensors import safe_open

device = 'cuda'
CONSOLIDATED = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1\consolidated.safetensors'

with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
    W_g_t = f.get_tensor('layers.0.feed_forward.w1.weight').float()
    W_u_t = f.get_tensor('layers.0.feed_forward.w3.weight').float()
    W_d_t = f.get_tensor('layers.0.feed_forward.w2.weight').float()

# Input de prueba
torch.manual_seed(42)
x = torch.randn(2, 64, 4096, device=device) * 0.1

# Teacher reference
W_g_t = W_g_t.to(device)
W_u_t = W_u_t.to(device)
W_d_t = W_d_t.to(device)

with torch.no_grad():
    gate_t = F.silu(x @ W_g_t.T)
    up_t = x @ W_u_t.T
    y_t = (gate_t * up_t) @ W_d_t.T

# Estadísticas del teacher
print(f"W_g: std={W_g_t.std():.4f}, mean={W_g_t.mean():.4f}")
print(f"W_u: std={W_u_t.std():.4f}, mean={W_u_t.mean():.4f}")
print(f"W_d: std={W_d_t.std():.4f}, mean={W_d_t.mean():.4f}")

# Probar distintos niveles de ruido
noise_levels = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]

print(f"\n{'Noise std':>10} {'Weight MSE':>12} {'Output COS':>12} {'Gate COS':>12}")
print("-" * 50)

for noise_std in noise_levels:
    torch.manual_seed(123)
    with torch.no_grad():
        W_g_n = W_g_t + torch.randn_like(W_g_t) * noise_std
        W_u_n = W_u_t + torch.randn_like(W_u_t) * noise_std
        W_d_n = W_d_t + torch.randn_like(W_d_t) * noise_std
        
        mse_w = F.mse_loss(W_g_n, W_g_t).item()
        
        gate_n = F.silu(x @ W_g_n.T)
        up_n = x @ W_u_n.T
        y_n = (gate_n * up_n) @ W_d_n.T
        
        cos_out = F.cosine_similarity(y_n.view(-1, 4096), y_t.view(-1, 4096), dim=1).mean().item()
        cos_gate = F.cosine_similarity(gate_n.view(-1, 14336), gate_t.view(-1, 14336), dim=1).mean().item()
        
        print(f"{noise_std:>10.4f} {mse_w:>12.8f} {cos_out:>12.4f} {cos_gate:>12.4f}")

# Probar con ruido correlacionado por fila (más parecido al error Kronecker)
print(f"\n{'='*60}")
print("Ruido ESTRUCTURADO: escalar distinto por fila")
print("(simula el error de Kronecker: filas dentro de un bloque son colineales)")
print("="*60)

for noise_std in [0.003, 0.01, 0.03]:
    torch.manual_seed(123)
    with torch.no_grad():
        # Ruido donde CADA FILA tiene su propio escalar
        # (similar a lo que hace Kronecker: filas de un bloque son colineales)
        row_scales = torch.randn(14336, 1, device=device) * noise_std
        base_noise = torch.randn_like(W_g_t) * 0.001  # patrón base chico
        W_g_n = W_g_t + row_scales * W_g_t.std(1, keepdim=True) + base_noise
        
        mse_w = F.mse_loss(W_g_n.contiguous(), W_g_t).item()
        
        gate_n = F.silu(x @ W_g_n.T)
        up_n = x @ W_u_t.T
        y_n = (gate_n * up_n) @ W_d_t.T
        
        cos_out = F.cosine_similarity(y_n.view(-1, 4096), y_t.view(-1, 4096), dim=1).mean().item()
        cos_gate = F.cosine_similarity(gate_n.view(-1, 14336), gate_t.view(-1, 14336), dim=1).mean().item()
        
        print(f"  Row-scale {noise_std:.4f}: MSE={mse_w:.8f}  COS_out={cos_out:.4f}  COS_gate={cos_gate:.4f}")

# Probar: ¿qué pasa si solo gate tiene ruido y up/down son perfectos?
print(f"\n{'='*60}")
print("Ruido SOLO en gate (up/down perfectos)")
print("="*60)

for noise_std in [0.001, 0.003, 0.01, 0.03]:
    torch.manual_seed(123)
    with torch.no_grad():
        W_g_n = W_g_t + torch.randn_like(W_g_t) * noise_std
        
        gate_n = F.silu(x @ W_g_n.T)
        y_n = (gate_n * up_t) @ W_d_t.T
        
        cos_out = F.cosine_similarity(y_n.view(-1, 4096), y_t.view(-1, 4096), dim=1).mean().item()
        cos_gate = F.cosine_similarity(gate_n.view(-1, 14336), gate_t.view(-1, 14336), dim=1).mean().item()
        
        print(f"  Gate noise {noise_std:.4f}: COS_out={cos_out:.4f}  COS_gate={cos_gate:.4f}")
