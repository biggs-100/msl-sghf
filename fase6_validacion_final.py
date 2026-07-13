"""
Fase 6: Validacion final con escalas fine-tuned.

Mide COS de MLP y atencion para todas las 32 capas
usando los archivos de mistral_ternario_ft/.
"""

import torch, torch.nn.functional as F, time, os
from safetensors import safe_open

device = 'cuda'
CONSOLIDATED = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1\consolidated.safetensors'
FT_DIR = 'mistral_ternario_ft'

print("=" * 60)
print("FASE 6: VALIDACION FINAL (escalas fine-tuned)")
print("=" * 60)

torch.manual_seed(42)
x = torch.randn(1, 32, 4096, device=device)

def load_ft(key):
    fname = key.replace('.', '_') + '.pt'
    d = torch.load(os.path.join(FT_DIR, fname), weights_only=False)
    return d['mask'].to(device), d['scale'].to(device)

results_mlp, results_attn_q, results_attn_o = [], [], []

with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
    for layer_idx in range(32):
        t0 = time.time()
        
        # ─── MLP ───
        W_g = f.get_tensor(f'layers.{layer_idx}.feed_forward.w1.weight').float().to(device)
        W_u = f.get_tensor(f'layers.{layer_idx}.feed_forward.w3.weight').float().to(device)
        W_d = f.get_tensor(f'layers.{layer_idx}.feed_forward.w2.weight').float().to(device)
        
        with torch.no_grad():
            y_t = (F.silu(x @ W_g.T) * (x @ W_u.T)) @ W_d.T
        
        mg, sg = load_ft(f'layers.{layer_idx}.feed_forward.w1.weight')
        mu, su = load_ft(f'layers.{layer_idx}.feed_forward.w3.weight')
        md, sd = load_ft(f'layers.{layer_idx}.feed_forward.w2.weight')
        
        with torch.no_grad():
            y_s = (F.silu(x @ (mg * sg.unsqueeze(1)).T) * (x @ (mu * su.unsqueeze(1)).T)) @ (md * sd.unsqueeze(1)).T
            cos_mlp = F.cosine_similarity(y_s.reshape(-1, 4096), y_t.reshape(-1, 4096), dim=1).mean().item()
        
        results_mlp.append(cos_mlp)
        
        # ─── Attention Q ───
        W_q = f.get_tensor(f'layers.{layer_idx}.attention.wq.weight').float().to(device)
        mq, sq = load_ft(f'layers.{layer_idx}.attention.wq.weight')
        with torch.no_grad():
            q_t = x @ W_q.T
            q_s = x @ (mq * sq.unsqueeze(1)).T
            cos_q = F.cosine_similarity(q_s.reshape(-1, 4096), q_t.reshape(-1, 4096), dim=1).mean().item()
        results_attn_q.append(cos_q)
        
        # ─── Attention O ───
        W_o = f.get_tensor(f'layers.{layer_idx}.attention.wo.weight').float().to(device)
        mo, so = load_ft(f'layers.{layer_idx}.attention.wo.weight')
        x_rand = torch.randn_like(x)
        with torch.no_grad():
            o_t = x_rand @ W_o.T
            o_s = x_rand @ (mo * so.unsqueeze(1)).T
            cos_o = F.cosine_similarity(o_s.reshape(-1, 4096), o_t.reshape(-1, 4096), dim=1).mean().item()
        results_attn_o.append(cos_o)
        
        dt = time.time() - t0
        print(f"  Layer {layer_idx:2d}: MLP={cos_mlp:.4f}  Q={cos_q:.4f}  O={cos_o:.4f}  [{dt:.0f}s]")
        
        del W_g, W_u, W_d, W_q, W_o
        torch.cuda.empty_cache()

# Resumen final
avg_mlp = sum(results_mlp) / len(results_mlp)
min_mlp = min(results_mlp)
avg_q = sum(results_attn_q) / len(results_attn_q)
avg_o = sum(results_attn_o) / len(results_attn_o)

# Tamaño estimado
total_params = 0
total_comp_bytes = 0
for fname in os.listdir(FT_DIR):
    if fname.endswith('.pt'):
        d = torch.load(os.path.join(FT_DIR, fname), weights_only=False)
        shape = d['shape']
        n_params = shape[0] * shape[1] if len(shape) == 2 else shape[0]
        total_params += n_params
        # Ternario 2-bit + escala FP16
        comp_bytes = n_params * 2 / 8 + shape[0] * 2
        total_comp_bytes += comp_bytes
        del d

total_orig_bytes = total_params * 4

print(f"\n{'='*60}")
print("VALIDACION FINAL — Mistral-7B")
print(f"{'='*60}")
print(f"\n  MLP COS promedio:   {avg_mlp:.4f}")
print(f"  MLP COS minimo:     {min_mlp:.4f}")
print(f"  Attention Q COS:    {avg_q:.4f}")
print(f"  Attention O COS:    {avg_o:.4f}")
print(f"\n  Tamaño original:  {total_orig_bytes/1024/1024/1024:.2f} GB (FP32)")
print(f"  Tamaño comprimido: {total_comp_bytes/1024/1024:.0f} MB")
print(f"  Compresion:        {total_orig_bytes/total_comp_bytes:.0f}x")
print(f"\n  {'✅ MODELO VALIDADO' if avg_mlp > 0.70 else '⚠️  REVISAR'}")
print(f"\n  Archivos en: {FT_DIR}/")
