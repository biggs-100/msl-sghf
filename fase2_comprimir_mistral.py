"""
Fase 2: comprime todas las capas de Mistral-7B con ternario.

Version optimizada: abre el safetensors UNA sola vez.
"""

import torch, torch.nn.functional as F, time, os, json
from safetensors import safe_open

device = 'cuda'
MODEL_DIR = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1'
CONSOLIDATED = os.path.join(MODEL_DIR, 'consolidated.safetensors')
OUTPUT_DIR = 'mistral_ternario'
os.makedirs(OUTPUT_DIR, exist_ok=True)

THRESHOLD = 0.7

def ternary_quant(W, ts=THRESHOLD):
    th = ts * W.abs().mean(dim=1, keepdim=True)
    t = torch.where(W > th, 1.0, torch.where(W < -th, -1.0, 0.0))
    s = (W.abs() * (t!=0).float()).sum(dim=1) / (t!=0).float().sum(dim=1).clamp(min=1)
    return t * s.unsqueeze(1), t, s

print("=" * 60)
print("FASE 2: Compresion ternaria de Mistral-7B")
print("=" * 60)

# Abrir UNA vez y leer todo
print("  Leyendo pesos del safetensors...")
t_start = time.time()

with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
    keys = [k for k in f.keys() if k.endswith('.weight') and 'norm' not in k.lower()]
    print(f"  Pesos a comprimir: {len(keys)}")
    
    total_orig = 0
    total_comp = 0
    results = []
    
    for idx, key in enumerate(keys):
        t0 = time.time()
        
        W = f.get_tensor(key).float()
        orig_bytes = W.numel() * 4
        out_f = W.shape[0]
        
        # Ternarizar
        W_q, mask, scale = ternary_quant(W)
        
        # Tamaño comprimido: 2 bits por elemento + 16 bits por fila para escala
        comp_bytes = (W.numel() * 2 + out_f * 16) / 8
        ratio = orig_bytes / comp_bytes
        sparsity = (mask == 0).float().mean().item()
        wmse = F.mse_loss(W_q, W).item()
        r2 = 1 - wmse / W.var().item()
        
        # Guardar
        name = key.replace('.', '_')
        torch.save({
            'mask': mask.cpu(),
            'scale': scale.cpu(),
            'shape': W.shape,
            'wmse': wmse,
            'r2': r2,
            'sparsity': sparsity,
        }, os.path.join(OUTPUT_DIR, f'{name}.pt'))
        
        total_orig += orig_bytes
        total_comp += comp_bytes
        
        short = f"{key.split('.')[-3].split('_')[-1]}.{key.split('.')[-2]}"
        print(f"  [{idx+1}/{len(keys)}] {short:>15} R²={r2:.3f} sp={sparsity:.0%} {ratio:.0f}x ({time.time()-t0:.1f}s)")
        
        # Free memory
        del W, W_q, mask, scale

# Resumen
print(f"\n{'='*60}")
print("RESUMEN")
print(f"{'='*60}")
print(f"  Original: {total_orig/1024/1024/1024:.2f} GB (FP32)")
print(f"  Comprimido: {total_comp/1024/1024:.0f} MB")
print(f"  Ratio: {total_orig/total_comp:.0f}x")
print(f"  Tiempo: {time.time()-t_start:.0f}s")
print(f"  Archivos en: {OUTPUT_DIR}/")

with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
    json.dump({
        'model': 'Mistral-7B-v0.3',
        'method': 'ternary + per-row scale',
        'threshold': THRESHOLD,
        'ratio': total_orig / total_comp,
    }, f)

print(f"\n✅ Listo. Siguiente: Fase 3 - validar con forward completo + perplexity")
