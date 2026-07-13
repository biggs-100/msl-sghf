"""
¿Cuanta estructura tienen los bloques Kronecker en pesos reales?

Tomamos un peso de Mistral-7B, lo dividimos en bloques,
y medimos el espectro singular de cada bloque.
Si los bloques son rank-1 → Kronecker con r=1 funciona.
Si son rank-alto → necesitamos mas rangos u otra aproximacion.
"""

import torch, math
from safetensors import safe_open
import json

device = 'cuda'

CONSOLIDATED = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1\consolidated.safetensors'

with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
    weights = {
        'gate': f.get_tensor('layers.0.feed_forward.w1.weight').float(),
        'up': f.get_tensor('layers.0.feed_forward.w3.weight').float(),
        'down': f.get_tensor('layers.0.feed_forward.w2.weight').float(),
        'q': f.get_tensor('layers.0.attention.wq.weight').float(),
        'k': f.get_tensor('layers.0.attention.wk.weight').float(),
        'v': f.get_tensor('layers.0.attention.wv.weight').float(),
        'o': f.get_tensor('layers.0.attention.wo.weight').float(),
    }

# Parametros de bloque (compresion 100x)
def get_block_params(out_f, in_f, comp=100):
    tp = out_f * in_f
    st = max(4, int(tp / comp))
    p = max(2, min(int(math.sqrt(st)), out_f))
    q = max(2, st // p)
    q = min(q, in_f)
    a = math.ceil(out_f / p)
    b = math.ceil(in_f / q)
    return p, q, a, b

for name, W in weights.items():
    out_f, in_f = W.shape
    p, q, a, b = get_block_params(out_f, in_f)
    
    print(f"\n{'='*60}")
    print(f"{name}: {tuple(W.shape)}  std={W.std():.4f}")
    print(f"  Bloques: {p}x{q} = {p*q} bloques de {a}x{b}")
    
    # Muestrear 100 bloques y analizar su SVD
    singular_vals = []
    rank_1_quality = []  # S[0]² / sum(S²) = fraccion de varianza del 1er SV
    
    torch.manual_seed(42)
    sample_indices = torch.randperm(p*q)[:100]
    
    for idx in sample_indices:
        i = idx % p
        j = idx // p
        rs = i * a
        re = min(rs + a, out_f)
        cs = j * b
        ce = min(cs + b, in_f)
        if rs >= out_f or cs >= in_f: continue
        
        block = W[rs:re, cs:ce]
        if block.numel() == 0: continue
        
        U, S, Vh = torch.linalg.svd(block, full_matrices=False)
        singular_vals.append(S.cpu())
        
        var_explained = (S[0]**2 / (S**2).sum()).item()
        rank_1_quality.append(var_explained)
    
    # Estadisticas
    sv = torch.stack([s for s in singular_vals if len(s) > 0])
    avg_sv = sv.mean(dim=0)
    
    print(f"  Valores singulares promedio (primeros 10):")
    for k in range(min(10, len(avg_sv))):
        print(f"    SV[{k}] = {avg_sv[k]:.6f}  (var={avg_sv[k]**2:.8f})")
    
    print(f"  Varianza total promedio por SV: {', '.join(f'{s**2:.6f}' for s in avg_sv[:6])}")
    print(f"  Rank-1 quality (S[0]²/ΣS²): mean={torch.tensor(rank_1_quality).mean():.4f}")
    print(f"  Rank-2 quality (S[0]²+S[1]²/ΣS²): mean=", end="")
    r2q = [(sv[0]**2 + sv[1]**2) / (sv**2).sum() if len(sv) > 1 else 1.0 for sv in singular_vals]
    print(f"{torch.tensor(r2q).mean():.4f}")
    
    # Mejor aproximacion rank-R promedio
    for R_target in [1, 2, 4, 8]:
        ratios = []
        for sv in singular_vals:
            if len(sv) > R_target:
                ratio = (sv[:R_target]**2).sum() / (sv**2).sum()
            else:
                ratio = 1.0
            ratios.append(ratio)
        print(f"  R={R_target} explica en promedio {torch.tensor(ratios).mean()*100:.1f}% de varianza del bloque")
    
    # Varianza total del peso vs varianza intra-bloque
    block_var = []
    for i in range(p):
        for j in range(q):
            rs = i * a
            re = min(rs + a, out_f)
            cs = j * b
            ce = min(cs + b, in_f)
            if rs >= out_f or cs >= in_f: continue
            block = W[rs:re, cs:ce]
            block_var.append(block.var().item())
    
    total_var = W.var().item()
    within_block_var = torch.tensor(block_var).mean().item()
    between_block_var = total_var - within_block_var
    print(f"\n  Varianza total: {total_var:.8f}")
    print(f"  Varianza intra-bloque: {within_block_var:.8f} ({within_block_var/total_var*100:.1f}%)")
    print(f"  Varianza entre bloques: {between_block_var:.8f} ({between_block_var/total_var*100:.1f}%)")
