import torch, torch.nn.functional as F
from sg_hf.core import FractalLinear
from safetensors import safe_open
import os

device = 'cuda'
path = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1\consolidated.safetensors'

# Teacher weights layer 0
with safe_open(path, framework='pt', device='cpu') as f:
    W1 = f.get_tensor('layers.0.feed_forward.w1.weight').float()
    W2 = f.get_tensor('layers.0.feed_forward.w2.weight').float()
    W3 = f.get_tensor('layers.0.feed_forward.w3.weight').float()

x = torch.randn(2, 64, 4096)

# Teacher forward
y_t = (F.silu(x @ W1.T) * (x @ W3.T)) @ W2.T

# Fractales
fl1 = FractalLinear(4096, 14336, compression=100.0)
fl2 = FractalLinear(14336, 4096, compression=100.0)
fl3 = FractalLinear(4096, 14336, compression=100.0)

for fl, W in [(fl1, W1), (fl2, W2), (fl3, W3)]:
    fl.initialize_from_teacher(W)
    # Escalar al std del teacher
    with torch.no_grad():
        fl.scale.data *= (W.std() / fl._generate_weight().std())
    
    # Optimizar
    fl.to(device)
    W = W.to(device)
    opt = torch.optim.Adam([fl.seed, fl.row_basis, fl.col_basis, fl.freq_scale, fl.freq_shift], lr=5e-3)
    for _ in range(100):
        opt.zero_grad()
        F.mse_loss(fl._generate_weight(), W).backward()
        opt.step()

# Fractal forward
with torch.no_grad():
    y_s = (F.silu(x.to(device) @ fl1._generate_weight().T) * (x.to(device) @ fl3._generate_weight().T)) @ fl2._generate_weight().T

cos = F.cosine_similarity(y_s.cpu().view(-1,4096), y_t.view(-1,4096), dim=1).mean().item()
print(f'MLP output COS = {cos:.4f}')
