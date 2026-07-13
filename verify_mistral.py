"""
Verifica Mistral-7B capa por capa: carga UN peso del SSD,
genera el mismo peso desde el seed, compara output del MLP.
Nunca carga mas de una capa en RAM.
"""

import os, torch, torch.nn.functional as F
from safetensors import safe_open
from sg_hf.core import FractalLinear

device = 'cuda'
COMPRESSION = 100.0
SEED_DIR = 'mistral_seeds'
MODEL_DIR = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1'
CONSOLIDATED = os.path.join(MODEL_DIR, 'consolidated.safetensors')

# Input de prueba (batch=4, seq=128, hidden=4096)
torch.manual_seed(42)
x = torch.randn(4, 128, 4096, device=device) * 0.1

total_cos = 0
total_layers = 0

for layer_idx in range(32):
    # Cargar pesos teacher de esta capa DESDE EL SSD
    with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
        W1_t = f.get_tensor(f'layers.{layer_idx}.feed_forward.w1.weight').float()
        W2_t = f.get_tensor(f'layers.{layer_idx}.feed_forward.w2.weight').float()
        W3_t = f.get_tensor(f'layers.{layer_idx}.feed_forward.w3.weight').float()

    # Cargar seeds
    def load_seed(name):
        fl = FractalLinear(4096, 14336, compression=COMPRESSION) if 'w2' not in name else FractalLinear(14336, 4096, compression=COMPRESSION)
        sd = torch.load(os.path.join(SEED_DIR, f'layers.{layer_idx}.feed_forward.{name}.weight.pt'), weights_only=False, map_location='cpu')
        fl.load_state_dict(sd)
        return fl.to(device)

    fl_w1 = load_seed(f'layers_{layer_idx}_feed_forward_w1_weight')
    fl_w2 = load_seed(f'layers_{layer_idx}_feed_forward_w2_weight')
    fl_w3 = load_seed(f'layers_{layer_idx}_feed_forward_w3_weight')

    # Forward teacher (CPU)
    W1_t = W1_t.to(device); W2_t = W2_t.to(device); W3_t = W3_t.to(device)
    gate_t = F.silu(x @ W1_t.T)
    up_t = x @ W3_t.T
    y_t = (gate_t * up_t) @ W2_t.T

    # Forward seed (GPU)
    with torch.no_grad():
        W1_s = fl_w1._generate_weight().to(device, dtype=torch.float32)
        W2_s = fl_w2._generate_weight().to(device, dtype=torch.float32)
        W3_s = fl_w3._generate_weight().to(device, dtype=torch.float32)
    gate_s = F.silu(x @ W1_s.T)
    up_s = x @ W3_s.T
    y_s = (gate_s * up_s) @ W2_s.T

    cos = F.cosine_similarity(y_s.view(-1, 4096), y_t.view(-1, 4096), dim=1).mean().item()
    mse = F.mse_loss(y_s, y_t).item()
    total_cos += cos
    total_layers += 1

    print(f"Layer {layer_idx:2d}: COS={cos:.4f}  MSE={mse:.6f}")

    # Liberar
    del W1_t, W2_t, W3_t, fl_w1, fl_w2, fl_w3, y_t, y_s
    if device == 'cuda':
        torch.cuda.empty_cache()

print(f"\nCOS PROMEDIO (32 capas): {total_cos/total_layers:.4f}")
