import os, json, time, torch, torch.nn.functional as F
from safetensors import safe_open
from sg_hf.core import FractalLinear

COMPRESSION = 100.0
SEED_DIR = 'mistral_seeds'
os.makedirs(SEED_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_DIR = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1'
CONSOLIDATED = os.path.join(MODEL_DIR, 'consolidated.safetensors')

print(f"Dispositivo: {device}")
print(f"Modelo: Mistral-7B-v0.3 (hidden=4096, intermediate=14336)")
print(f"Archivo: {CONSOLIDATED}")
print()

# Listar pesos lineales en el consolidated
with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
    all_keys = list(f.keys())

# Filtrar solo pesos de capas lineales (mlp + attention)
linear_keys = [k for k in all_keys if k.endswith('.weight') 
               and (any(x in k for x in ['attention.', 'feed_forward.'])
                    or k == 'output.weight')
               and 'norm' not in k]

print(f"Total pesos lineales: {len(linear_keys)}")
print()

total_orig = 0
total_seed = 0

def compress_one(key):
    """Comprime un solo peso. Todo se limpi al salir de esta funcion."""
    with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
        weight = f.get_tensor(key).float()

    out_f, in_f = weight.shape
    orig = weight.numel()

    fl = FractalLinear(in_f, out_f, compression=COMPRESSION)
    opt = torch.optim.Adam(list(fl.parameters()), lr=1e-2)

    if device == 'cuda':
        weight = weight.cuda()
        fl = fl.cuda()

    best = float('inf')
    for step in range(300):
        opt.zero_grad()
        W = fl._generate_weight()
        loss = F.mse_loss(W, weight)
        loss.backward()
        opt.step()
        if loss.item() < best:
            best = loss.item()

    safe_name = key.replace('.', '_').replace('/', '_')
    torch.save(fl.cpu().state_dict(), os.path.join(SEED_DIR, f'{safe_name}.pt'))

    return orig, fl.total_compressed, best

# ─── Loop principal ───
for idx, key in enumerate(linear_keys):
    short = key.split('.')[-3] + '.' + key.split('.')[-2]
    print(f"[{idx+1}/{len(linear_keys)}] {short}...", end=' ', flush=True)
    
    orig, seed_p, best = compress_one(key)
    ratio = orig / seed_p
    total_orig += orig
    total_seed += seed_p
    print(f"MSE={best:.6f} | {ratio:.0f}x | Seed={seed_p:,}")

print(f"\n{'='*50}")
print(f"COMPRESION COMPLETA")
print(f"{'='*50}")
print(f"Original: {total_orig:,} params")
print(f"Seeds:    {total_seed:,} params")
print(f"Total:    {total_orig/total_seed:.0f}x")
print(f"Tamano:   {total_seed * 2 / 1024 / 1024:.0f} MB (FP16)")
