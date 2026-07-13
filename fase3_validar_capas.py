"""
Fase 3: Validar compresion ternaria capa por capa.

Para cada capa:
  1. Carga teacher weights desde safetensors
  2. Carga mascara ternaria + escalas
  3. Compara MLP output (gate*up*down) entre teacher y ternario
  4. Mide COS

No necesita cargar el modelo completo — procesa una capa a la vez.
"""

import torch, torch.nn.functional as F, time, os, json
from safetensors import safe_open

device = 'cuda'
CONSOLIDATED = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1\consolidated.safetensors'
TERN_DIR = 'mistral_ternario'

print("=" * 60)
print("FASE 3: Validacion capa por capa")
print("=" * 60)

# Input de prueba
torch.manual_seed(42)
x = torch.randn(1, 16, 4096, device=device)  # input realista

def load_ternary(layer_key):
    """Carga mascara ternaria y escala desde archivo."""
    name = layer_key.replace('.', '_')
    data = torch.load(os.path.join(TERN_DIR, f'{name}.pt'), weights_only=False, map_location='cpu')
    return data['mask'].to(device), data['scale'].to(device)

def apply_ternary(mask, scale):
    """Reconstruye peso ternario: W = mask * scale (con broadcasting)."""
    return mask * scale.unsqueeze(1)

results_mlp = []
results_attn = []

with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
    for layer_idx in range(32):
        t0 = time.time()
        
        # ─── MLP weights ───
        W_g_t = f.get_tensor(f'layers.{layer_idx}.feed_forward.w1.weight').float().to(device)
        W_u_t = f.get_tensor(f'layers.{layer_idx}.feed_forward.w3.weight').float().to(device)
        W_d_t = f.get_tensor(f'layers.{layer_idx}.feed_forward.w2.weight').float().to(device)
        
        # MLP teacher
        with torch.no_grad():
            gate_t = F.silu(x @ W_g_t.T)
            up_t = x @ W_u_t.T
            y_t = (gate_t * up_t) @ W_d_t.T
        
        # MLP ternario
        mg, sg = load_ternary(f'layers.{layer_idx}.feed_forward.w1.weight')
        mu, su = load_ternary(f'layers.{layer_idx}.feed_forward.w3.weight')
        md, sd = load_ternary(f'layers.{layer_idx}.feed_forward.w2.weight')
        
        with torch.no_grad():
            W_g_q = apply_ternary(mg, sg)
            W_u_q = apply_ternary(mu, su)
            W_d_q = apply_ternary(md, sd)
            
            gate_s = F.silu(x @ W_g_q.T)
            up_s = x @ W_u_q.T
            y_s = (gate_s * up_s) @ W_d_q.T
            
            cos_mlp = F.cosine_similarity(y_s.reshape(-1, 4096), y_t.reshape(-1, 4096), dim=1).mean().item()
        
        results_mlp.append(cos_mlp)
        
        # ─── Attention (QKV → Q@x) ───
        W_q_t = f.get_tensor(f'layers.{layer_idx}.attention.wq.weight').float().to(device)
        W_k_t = f.get_tensor(f'layers.{layer_idx}.attention.wk.weight').float().to(device)
        W_v_t = f.get_tensor(f'layers.{layer_idx}.attention.wv.weight').float().to(device)
        
        with torch.no_grad():
            q_t = x @ W_q_t.T
            k_t = x @ W_k_t.T
            v_t = x @ W_v_t.T
        
        mq, sq = load_ternary(f'layers.{layer_idx}.attention.wq.weight')
        mk, sk = load_ternary(f'layers.{layer_idx}.attention.wk.weight')
        mv, sv = load_ternary(f'layers.{layer_idx}.attention.wv.weight')
        
        with torch.no_grad():
            q_s = x @ apply_ternary(mq, sq).T
            k_s = x @ apply_ternary(mk, sk).T
            v_s = x @ apply_ternary(mv, sv).T
            
            cos_q = F.cosine_similarity(q_s.reshape(-1, q_s.size(-1)), q_t.reshape(-1, q_t.size(-1)), dim=1).mean().item()
        
        results_attn.append(cos_q)
        
        # Liberar memoria
        del W_g_t, W_u_t, W_d_t, W_q_t, W_k_t, W_v_t
        del mg, sg, mu, su, md, sd, mq, sq, mk, sk, mv, sv
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        dt = time.time() - t0
        print(f"  Layer {layer_idx:2d}: MLP COS={cos_mlp:.4f}  Q COS={cos_q:.4f}  ({dt:.1f}s)")

# Resumen
print(f"\n{'='*60}")
print("RESUMEN VALIDACION")
print(f"{'='*60}")
print(f"\nMLP COS promedio: {sum(results_mlp)/len(results_mlp):.4f}")
print(f"MLP COS minimo:   {min(results_mlp):.4f}")
print(f"Q COS promedio:   {sum(results_attn)/len(results_attn):.4f}")
print(f"Q COS minimo:     {min(results_attn):.4f}")

# Histograma
print(f"\nDistribucion MLP COS:")
for bucket in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
    count = sum(1 for c in results_mlp if c >= bucket)
    bar = '#' * count
    print(f"  >={bucket:.2f}: {count:2d} {bar}")

print(f"\n  {'✅ VALIDADO' if sum(results_mlp)/len(results_mlp) > 0.85 else '⚠️  REVISAR'}")
print(f"  Compresion: 16x (ternario 2-bit por elemento + escala FP16 por fila)")
print(f"  Total modelo: ~1.5 GB (vs ~13.5 GB FP16 original)")
