"""
Streaming compression: lee pesos del SSD capa por capa, 
optimiza seeds, guarda SOLO los seeds en cloud.

No necesita cargar el modelo completo en RAM.
Funciona para modelos de cualquier tamaño (70B, 295B, 753B).
"""

import os, sys, json, time, torch
import torch.nn.functional as F
from safetensors import safe_open
from sg_hf.core import FractalLinear

# ─── Config ───
COMPRESSION = 100.0
MODEL_PATH = None  # Setear antes de correr
SEED_DIR = 'seeds'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_weight_streamer(model_path):
    """
    Devuelve un iterador que lee los pesos del SSD uno por uno.
    No carga nada en RAM hasta que se pide.
    """
    index_path = os.path.join(model_path, 'model.safetensors.index.json')
    with open(index_path) as f:
        weight_map = json.load(f)['weight_map']
    
    # Cache de archivos abiertos
    open_files = {}
    
    def get_tensor(name):
        file = weight_map[name]
        if file not in open_files:
            shard_path = os.path.join(model_path, file)
            open_files[file] = safe_open(shard_path, framework='pt', device='cpu')
        tensor = open_files[file].get_tensor(name)
        return tensor.float() if tensor.dtype == torch.bfloat16 else tensor
    
    return get_tensor, list(weight_map.keys())


def stream_compress(model_path, layer_pattern=None):
    """
    Comprime TODOS los pesos lineales del modelo streaming del SSD.
    
    Args:
        model_path: ruta al directorio con safetensors + index.json
        layer_pattern: filtro opcional (ej: 'mlp.gate_proj')
    """
    os.makedirs(SEED_DIR, exist_ok=True)
    
    get_tensor, all_weights = get_weight_streamer(model_path)
    
    # Filtrar solo pesos lineales (terminan en .weight, no son norms/embeddings)
    linear_weights = [w for w in all_weights if w.endswith('.weight') 
                      and not any(x in w for x in ['norm', 'embed', 'ln_'])]
    
    if layer_pattern:
        linear_weights = [w for w in linear_weights if layer_pattern in w]
    
    print(f"Total pesos lineales encontrados: {len(linear_weights)}")
    print(f"Streaming desde: {model_path}")
    print(f"Device: {DEVICE}")
    
    total_orig = 0
    total_seed = 0
    results = []
    
    for idx, weight_name in enumerate(linear_weights):
        print(f"\n[{idx+1}/{len(linear_weights)}] {weight_name}")
        
        # 1. Leer del SSD SOLO este peso
        weight = get_tensor(weight_name)
        out_f, in_f = weight.shape
        orig_params = weight.numel()
        
        print(f"  Shape: {list(weight.shape)} = {orig_params:,} params")
        
        # 2. Mover a GPU solo este peso
        weight = weight.to(DEVICE)
        
        # 3. Crear y optimizar seed
        fl = FractalLinear(in_f, out_f, compression=COMPRESSION).to(DEVICE)
        params = [fl.seed, fl.freq_scale, fl.freq_shift, fl.row_basis, fl.col_basis]
        opt = torch.optim.Adam(params, lr=1e-2)
        
        t0 = time.perf_counter()
        best_loss = float('inf')
        steps = 300 if orig_params > 1_000_000 else 150
        
        for step in range(steps):
            opt.zero_grad()
            W_gen = fl._generate_weight()
            loss = F.mse_loss(W_gen, weight)
            loss.backward()
            opt.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
        
        elapsed = time.perf_counter() - t0
        
        # 4. Descargar peso teacher de GPU (liberar memoria)
        del weight
        
        # 5. Guardar seed (solo ocupa seed_params bytes)
        safe_name = weight_name.replace('/', '_').replace('.', '_')
        torch.save(fl.state_dict(), os.path.join(SEED_DIR, f'{safe_name}.pt'))
        
        # 6. Descargar fractal de GPU
        del fl
        
        seed_params = seed_params_proxy(in_f, out_f, COMPRESSION)
        ratio = orig_params / seed_params if seed_params > 0 else 0
        
        total_orig += orig_params
        total_seed += seed_params
        
        print(f"  MSE: {best_loss:.6f}  |  {ratio:.0f}x comp  |  "
              f"{elapsed:.1f}s  |  Seed: {seed_params:,} params")
        
        results.append({
            'name': weight_name,
            'orig': orig_params,
            'seed': seed_params,
            'ratio': ratio,
            'mse': best_loss,
        })
        
        # Liberar cache CUDA entre capas
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
    
    # Resumen
    print(f"\n{'='*60}")
    print(f"  COMPRESION COMPLETA")
    print(f"{'='*60}")
    print(f"  Original: {total_orig:,} params")
    print(f"  Seeds:    {total_seed:,} params")
    print(f"  Total:    {total_orig/total_seed:.0f}x")
    print(f"  Tamano:   {total_seed * 2 / 1024 / 1024:.0f} MB (FP16)")
    print(f"  Archivos: {len(results)} seeds en {SEED_DIR}/")
    print(f"\n  ✓ LISTO PARA SUBIR A LA NUBE")
    
    return results


def seed_params_proxy(in_f, out_f, compression):
    """Estima params del seed sin crear el FractalLinear."""
    total = in_f * out_f
    target = max(4, int(total / compression))
    p = max(2, int(target ** 0.5))
    q = max(2, target // p)
    a = (out_f + p - 1) // p
    b = (in_f + q - 1) // q
    fft = p * (q // 2 + 1) * 2
    return p * q + p * a + q * b + fft


if __name__ == '__main__':
    if MODEL_PATH is None:
        # Ejemplo: Qwen3.5-4B ya descargado
        MODEL_PATH = os.path.expanduser(
            r'~\.cache\huggingface\hub\models--Qwen--Qwen3.5-4B'
            r'\snapshots\851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
        )
    
    results = stream_compress(MODEL_PATH)
    
    print(f"\nPara subir a cloud:")
    print(f"  scp -r {SEED_DIR}/ usuario@cloud:~/\n")
    print(f"En cloud:")
    print(f"  python -c \"\import torch\n"
          f"  from sg_hf.core import FractalLinear\n"
          f"  fl = FractalLinear(in_f, out_f, compression=100.0)\n"
          f"  fl.load_state_dict(torch.load('{SEED_DIR}/...pt'))\n"
          f"  W = fl._generate_weight()\n"
          f"  print('Seed funciona, W shape:', W.shape)\"")
