"""
Comprime las capas lineales de Qwen3.5-4B con FractalLinear a 100x.
Lee los safetensors descargados, extrae pesos, entrena seeds.
"""

import os, json, time, torch
import torch.nn.functional as F
from safetensors import safe_open
from sg_hf.core import FractalLinear

device = 'cuda' if torch.cuda.is_available() else 'cpu'
COMPRESSION = 100.0

# Ruta a los safetensors descargados
CACHE_DIR = os.path.expanduser(r'~\.cache\huggingface\hub\models--Qwen--Qwen3.5-4B\snapshots\851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
FILE1 = os.path.join(CACHE_DIR, 'model.safetensors-00001-of-00002.safetensors')
FILE2 = os.path.join(CACHE_DIR, 'model.safetensors-00002-of-00002.safetensors')

# Cargar indice para saber donde esta cada peso
INDEX = os.path.join(CACHE_DIR, 'model.safetensors.index.json')
with open(INDEX) as f:
    WEIGHT_MAP = json.load(f)['weight_map']


def get_tensor(name):
    """Lee un tensor del safetensor correcto."""
    file = WEIGHT_MAP[name]
    path = FILE1 if '00001' in file else FILE2
    with safe_open(path, framework='pt', device='cpu') as f:
        tensor = f.get_tensor(name)
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.to(device)


def get_layer_config(layer_idx):
    """Determina si una capa es linear_attention o full_attention."""
    with open(os.path.join(CACHE_DIR, 'config.json')) as f:
        config = json.load(f)
    layer_types = config['text_config']['layer_types']
    return layer_types[layer_idx]  # 'linear_attention' o 'full_attention'


def compress_linear(name, weight, save_dir='compressed_qwen'):
    """
    Crea un FractalLinear para esta weight matrix y entrena el seed.
    Retorna el FractalLinear y el MSE final.
    """
    out_features, in_features = weight.shape
    orig_params = weight.numel()
    
    # Crear fractal linear
    fl = FractalLinear(in_features, out_features, compression=COMPRESSION).to(device)
    
    # Inicializar seed para que genere W ≈ teacher
    params = [fl.seed, fl.freq_scale, fl.freq_shift, fl.row_basis, fl.col_basis]
    opt = torch.optim.Adam(params, lr=1e-2)
    
    best_loss = float('inf')
    t0 = time.perf_counter()
    
    for step in range(300):
        opt.zero_grad()
        W_gen = fl._generate_weight()
        loss = F.mse_loss(W_gen, weight)
        loss.backward()
        opt.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if step % 50 == 0 or step == 299:
            print(f'    Step {step:3d}: MSE={loss.item():.8f}')
    
    elapsed = time.perf_counter() - t0
    
    seed_params = fl.total_compressed
    ratio = orig_params / seed_params
    
    # Guardar seed state
    os.makedirs(save_dir, exist_ok=True)
    safe_name = name.replace('.', '_').replace('/', '_')
    torch.save(fl.state_dict(), os.path.join(save_dir, f'{safe_name}.pt'))
    
    return fl, {
        'name': name,
        'orig_params': orig_params,
        'seed_params': seed_params,
        'ratio': ratio,
        'mse': best_loss,
        'shape': list(weight.shape),
        'time_sec': elapsed,
    }


def main():
    print(f"Device: {device}")
    print(f"Compression target: {COMPRESSION}x")
    
    # Lista de pesos a comprimir (una capa de ejemplo)
    layer_idx = 0
    layer_type = get_layer_config(layer_idx)
    print(f"\nLayer {layer_idx}: {layer_type}")
    
    weights_to_compress = [
        f'model.language_model.layers.{layer_idx}.mlp.gate_proj.weight',
        f'model.language_model.layers.{layer_idx}.mlp.up_proj.weight', 
        f'model.language_model.layers.{layer_idx}.mlp.down_proj.weight',
        f'model.language_model.layers.{layer_idx}.linear_attn.out_proj.weight',
    ]
    
    if layer_type == 'full_attention':
        weights_to_compress.append(
            f'model.language_model.layers.{layer_idx}.linear_attn.in_proj_qkv.weight'
        )
    else:
        weights_to_compress.extend([
            f'model.language_model.layers.{layer_idx}.linear_attn.in_proj_a.weight',
            f'model.language_model.layers.{layer_idx}.linear_attn.in_proj_b.weight',
            f'model.language_model.layers.{layer_idx}.linear_attn.in_proj_z.weight',
        ])
    
    results = []
    total_orig = 0
    total_seed = 0
    
    for name in weights_to_compress:
        print(f"\n  >> Loading: {name.split('.')[-1]}")
        weight = get_tensor(name)
        print(f"     Shape: {list(weight.shape)} = {weight.numel():,} params")
        
        fl, info = compress_linear(name, weight)
        results.append(info)
        total_orig += info['orig_params']
        total_seed += info['seed_params']
        
        print(f"     Ratio: {info['ratio']:.0f}x  |  MSE: {info['mse']:.6f}  |  "
              f"Seed: {info['seed_params']:,} params  |  {info['time_sec']:.1f}s")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTADOS - Layer {layer_idx}")
    print(f"{'='*60}")
    print(f"  {'Layer':<30} {'Original':>12} {'Seed':>10} {'Ratio':>8} {'MSE':>10}")
    print(f"  {'-'*70}")
    for r in results:
        layer_short = r['name'].split('.')[-3] + '.' + r['name'].split('.')[-1]
        print(f"  {layer_short:<30} {r['orig_params']:>12,} {r['seed_params']:>10,} "
              f"{r['ratio']:>7.0f}x {r['mse']:>10.6f}")
    print(f"  {'-'*70}")
    print(f"  {'TOTAL':<30} {total_orig:>12,} {total_seed:>10,} "
          f"{total_orig/total_seed:>7.0f}x")
    print(f"\n  Compression efectiva: {total_orig/total_seed:.0f}x")
    print(f"  Todos los seeds de Qwen3.5-4B (32 capas): "
          f"~{total_seed * 32 / 1e6:.1f}M params (~{(total_seed * 32 * 2) / 1e6:.0f} MB en FP16)")
    
    return results


def verify_mlp():
    """Compara la salida del MLP original vs fractal para Layer 0."""
    print(f"\n{'='*60}")
    print(f"  VERIFICACION: MLP Layer 0 - Teacher vs Fractal")
    print(f"{'='*60}")
    
    layer = 0
    names = {
        'gate': f'model.language_model.layers.{layer}.mlp.gate_proj.weight',
        'up': f'model.language_model.layers.{layer}.mlp.up_proj.weight',
        'down': f'model.language_model.layers.{layer}.mlp.down_proj.weight',
    }
    
    # Cargar teacher weights
    W_gate_t = get_tensor(names['gate']).float()
    W_up_t = get_tensor(names['up']).float()
    W_down_t = get_tensor(names['down']).float()
    
    # Cargar fractal seeds guardados
    fl_gate = FractalLinear(2560, 9216, compression=COMPRESSION).to(device)
    fl_up = FractalLinear(2560, 9216, compression=COMPRESSION).to(device)
    fl_down = FractalLinear(9216, 2560, compression=COMPRESSION).to(device)
    
    safe_dir = 'compressed_qwen'
    base = f'model_language_model_layers_{layer}_mlp'
    fl_gate.load_state_dict(torch.load(f'{safe_dir}/{base}_gate_proj_weight.pt'))
    fl_up.load_state_dict(torch.load(f'{safe_dir}/{base}_up_proj_weight.pt'))
    fl_down.load_state_dict(torch.load(f'{safe_dir}/{base}_down_proj_weight.pt'))
    
    # Input de prueba
    torch.manual_seed(42)
    x = torch.randn(128, 2560, device=device)
    
    # Teacher MLP
    with torch.no_grad():
        gate_t = torch.sigmoid(x @ W_gate_t.T)
        up_t = x @ W_up_t.T
        hidden_t = gate_t * up_t
        out_t = hidden_t @ W_down_t.T
    
    # Fractal MLP
    with torch.no_grad():
        W_gate_s = fl_gate._generate_weight()
        W_up_s = fl_up._generate_weight()
        W_down_s = fl_down._generate_weight()
        
        gate_s = torch.sigmoid(x @ W_gate_s.T)
        up_s = x @ W_up_s.T
        hidden_s = gate_s * up_s
        out_s = hidden_s @ W_down_s.T
    
    # Comparar
    mse_gate = F.mse_loss(gate_s, gate_t).item()
    mse_up = F.mse_loss(up_s, up_t).item()
    mse_hidden = F.mse_loss(hidden_s, hidden_t).item()
    mse_out = F.mse_loss(out_s, out_t).item()
    
    print(f"  MSE gate:     {mse_gate:.8f}")
    print(f"  MSE up:       {mse_up:.8f}")
    print(f"  MSE hidden:   {mse_hidden:.8f}")
    print(f"  MSE output:   {mse_out:.8f}")
    print(f"  Correlation:  {torch.nn.functional.cosine_similarity(out_s.mean(0), out_t.mean(0), dim=0).item():.6f}")
    
    return mse_out


if __name__ == '__main__':
    main()
    verify_mlp()
