"""
Pipeline completo: comprime Qwen3.5-4B entero con SG-HF y prueba inferencia.

1. Lee los safetensors de Qwen3.5-4B
2. Comprime CADA capa lineal con FractalLinear a 64x
3. Guarda todos los seeds
4. Carga seeds y ejecuta inferencia comparando con el teacher
"""

import os, json, time, torch
import torch.nn.functional as F
from safetensors import safe_open
from sg_hf.core import FractalLinear
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
COMPRESSION = 100.0
SEED_DIR = 'compressed_qwen_full'

CACHE_DIR = os.path.expanduser(r'~\.cache\huggingface\hub\models--Qwen--Qwen3.5-4B\snapshots\851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
FILE1 = os.path.join(CACHE_DIR, 'model.safetensors-00001-of-00002.safetensors')
FILE2 = os.path.join(CACHE_DIR, 'model.safetensors-00002-of-00002.safetensors')
INDEX = os.path.join(CACHE_DIR, 'model.safetensors.index.json')

with open(INDEX) as f:
    WEIGHT_MAP = json.load(f)['weight_map']


def get_tensor(name):
    file = WEIGHT_MAP[name]
    path = FILE1 if '00001' in file else FILE2
    with safe_open(path, framework='pt', device='cpu') as f:
        t = f.get_tensor(name)
        return t.float().to(device) if t.dtype == torch.bfloat16 else t.to(device)


# ─── Capas lineales por layer ───

LINEAR_KEYS = {
    'linear_attention': [
        'linear_attn.in_proj_qkv.weight',
        'linear_attn.out_proj.weight',
        'mlp.gate_proj.weight',
        'mlp.up_proj.weight',
        'mlp.down_proj.weight',
    ],
    'full_attention': [
        'self_attn.q_proj.weight',
        'self_attn.k_proj.weight',
        'self_attn.v_proj.weight',
        'self_attn.o_proj.weight',
        'mlp.gate_proj.weight',
        'mlp.up_proj.weight',
        'mlp.down_proj.weight',
    ],
}

# Capas de attention lineal (SSM) tienen ademas proj_a, proj_b, proj_z
LINEAR_KEYS_EXTRA = [
    'linear_attn.in_proj_a.weight',
    'linear_attn.in_proj_b.weight',
    'linear_attn.in_proj_z.weight',
]


def get_layer_type(layer_idx):
    with open(os.path.join(CACHE_DIR, 'config.json')) as f:
        config = json.load(f)
    return config['text_config']['layer_types'][layer_idx]


def compress_all():
    """Comprime todas las capas lineales del modelo."""
    os.makedirs(SEED_DIR, exist_ok=True)
    results = {}
    total_orig = 0
    total_seed = 0

    for layer_idx in range(32):
        lt = get_layer_type(layer_idx)
        keys = list(LINEAR_KEYS[lt])

        # linear_attention (SSM) tiene proj_a/b/z extra
        if lt == 'linear_attention':
            keys += LINEAR_KEYS_EXTRA

        print(f"\nLayer {layer_idx:2d}/{32} ({lt}): {len(keys)} weights")

        layer_orig = 0
        layer_seed = 0

        for key in keys:
            full_name = f'model.language_model.layers.{layer_idx}.{key}'

            weight = get_tensor(full_name)
            out_f, in_f = weight.shape

            fl = FractalLinear(in_f, out_f, compression=COMPRESSION).to(device)
            params = [fl.seed, fl.freq_scale, fl.freq_shift, fl.row_basis, fl.col_basis]
            opt = torch.optim.Adam(params, lr=1e-2)

            best_loss = float('inf')
            for step in range(300):
                opt.zero_grad()
                W_gen = fl._generate_weight()
                loss = F.mse_loss(W_gen, weight)
                loss.backward()
                opt.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()

            # Guardar seed
            safe_name = full_name.replace('.', '_').replace('/', '_')
            torch.save(fl.state_dict(), os.path.join(SEED_DIR, f'{safe_name}.pt'))

            layer_orig += weight.numel()
            layer_seed += fl.total_compressed
            short = key.split('.')[-1]
            print(f"  {short:<20} orig={weight.numel():>9,} seed={fl.total_compressed:>7,} "
                  f"ratio={weight.numel()/fl.total_compressed:>5.0f}x mse={best_loss:.6f}")

        total_orig += layer_orig
        total_seed += layer_seed
        print(f"  Layer total: {layer_orig:>9,} → {layer_seed:>7,} = {layer_orig/max(1,layer_seed):.0f}x")

    # Summary
    print(f"\n{'='*60}")
    print(f"  COMPLETADO")
    print(f"{'='*60}")
    print(f"  Original linear params: {total_orig:,}")
    print(f"  Seed params:           {total_seed:,}")
    print(f"  Compression ratio:     {total_orig/total_seed:.0f}x")
    print(f"  Seed size (FP16):      {total_seed * 2 / 1024 / 1024:.0f} MB")
    print(f"  Seed size (INT4):      {total_seed * 0.5 / 1024 / 1024:.0f} MB")

    return results


def verify_random():
    """
    Verifica que los seeds generan las mismas activaciones que el teacher
    para una capa aleatoria (capa 5) con entrada aleatoria.
    """
    print(f"\n{'='*60}")
    print(f"  VERIFICACION: Activaciones Layer 5")
    print(f"{'='*60}")

    layer = 5
    lt = get_layer_type(layer)
    keys = LINEAR_KEYS[lt]
    if lt == 'linear_attention':
        keys += LINEAR_KEYS_EXTRA

    # Cargar teacher weights
    W_teacher = {}
    for key in keys:
        full_name = f'model.language_model.layers.{layer}.{key}'
        W_teacher[key] = get_tensor(full_name)

    # Cargar fractal seeds
    fractal_layers = {}
    for key in keys:
        full_name = f'model.language_model.layers.{layer}.{key}'
        w = W_teacher[key]
        fl = FractalLinear(w.shape[1], w.shape[0], compression=COMPRESSION).to(device)
        safe_name = full_name.replace('.', '_').replace('/', '_')
        fl.load_state_dict(torch.load(os.path.join(SEED_DIR, f'{safe_name}.pt'),
                                       weights_only=False))
        fractal_layers[key] = fl

    # Input semi-realista: embedding + posicion
    torch.manual_seed(42)
    # Usamos un embedding simulado con estructura (no ruido puro)
    x = torch.randn(4, 128, 2560, device=device) * 0.5 + 0.1 * torch.sin(
        torch.arange(128, device=device).float()[:, None] @ torch.ones(1, 2560, device=device) * 0.1
    ).unsqueeze(0)

    # Verificar MLP (gate_proj + up_proj → down_proj)
    h_t = F.silu(x @ W_teacher['mlp.gate_proj.weight'].T) * (x @ W_teacher['mlp.up_proj.weight'].T)
    h_t = h_t @ W_teacher['mlp.down_proj.weight'].T

    with torch.no_grad():
        W_gate = fractal_layers['mlp.gate_proj.weight']._generate_weight()
        W_up = fractal_layers['mlp.up_proj.weight']._generate_weight()
        W_down = fractal_layers['mlp.down_proj.weight']._generate_weight()
    h_s = F.silu(x @ W_gate.T) * (x @ W_up.T)
    h_s = h_s @ W_down.T

    mse = F.mse_loss(h_s, h_t).item()
    # Cosine similarity de activaciones (shape: [B, T, C] → mean por token)
    cos_per_token = torch.nn.functional.cosine_similarity(
        h_s.mean(dim=0), h_t.mean(dim=0), dim=1  # (T, C) mean over batch
    ).mean().item()
    # Weight MSE (directa)
    w_mse_gate = F.mse_loss(W_gate, W_teacher['mlp.gate_proj.weight']).item()
    w_mse_up = F.mse_loss(W_up, W_teacher['mlp.up_proj.weight']).item()
    w_mse_down = F.mse_loss(W_down, W_teacher['mlp.down_proj.weight']).item()

    print(f"  Weight MSE  (gate/up/down): {w_mse_gate:.6f} / {w_mse_up:.6f} / {w_mse_down:.6f}")
    print(f"  Activation MSE (MLP output): {mse:.6f}")
    print(f"  Cosine sim  (mean over tokens): {cos_per_token:.6f}")

    return {'mse': mse, 'cos': cos_per_token}


if __name__ == '__main__':
    t0 = time.perf_counter()
    compress_all()
    elapsed = time.perf_counter() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")
    verify_random()
