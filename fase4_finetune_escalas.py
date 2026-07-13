"""
Fase 4: Fine-tune escalas ternarias capa por capa.

Para cada capa del MLP (gate+up+down):
  1. Carga teacher + mascaras ternarias
  2. 100 pasos Adam solo en escalas
  3. Loss: MSE del output MLP vs teacher

Sube COS de ~0.75 a ~0.90+ (validado en layer 0).
"""

import torch, torch.nn.functional as F, time, os
from safetensors import safe_open

device = 'cuda'
CONSOLIDATED = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1\consolidated.safetensors'
TERN_DIR = 'mistral_ternario'
OUT_DIR = 'mistral_ternario_ft'
os.makedirs(OUT_DIR, exist_ok=True)

STEPS = 50
LR = 1e-3

print("=" * 60)
print("FASE 4: Fine-tune escalas ternarias")
print(f"  Steps: {STEPS}, LR: {LR}")
print(f"  Salida: {OUT_DIR}/")
print("=" * 60)

# Input de entrenamiento (5 batches)
torch.manual_seed(42)
train_data = [torch.randn(1, 32, 4096, device=device) for _ in range(3)]

# Input de evaluacion (fijo)
torch.manual_seed(42)
x_eval = torch.randn(1, 32, 4096, device=device)

def load_ternary(key):
    data = torch.load(os.path.join(TERN_DIR, f'{key.replace(".", "_")}.pt'), weights_only=False)
    return data['mask'].to(device), data['scale'].to(device)

results = []

# Saltar capas ya fine-tuned
import glob
done_layers = set()
for fpath in glob.glob(os.path.join(OUT_DIR, 'layers_*_feed_forward_w1_weight.pt')):
    parts = os.path.basename(fpath).split('_')
    lidx = parts[1]
    done_layers.add(int(lidx))
print(f"  Capas ya procesadas: {sorted(done_layers)}")

with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
    for layer_idx in range(32):
        if layer_idx in done_layers:
            # Cargar scales ya fine-tuned para reportar
            d = torch.load(os.path.join(OUT_DIR, f'layers_{layer_idx}_feed_forward_w1_weight.pt'), weights_only=False)
            sg_ft = d['scale']
            d = torch.load(os.path.join(OUT_DIR, f'layers_{layer_idx}_feed_forward_w3_weight.pt'), weights_only=False)
            su_ft = d['scale']
            d = torch.load(os.path.join(OUT_DIR, f'layers_{layer_idx}_feed_forward_w2_weight.pt'), weights_only=False)
            sd_ft = d['scale']
            
            # Cargar masks originales y teacher para evaluar
            d_orig = torch.load(os.path.join(TERN_DIR, f'layers_{layer_idx}_feed_forward_w1_weight.pt'), weights_only=False)
            mg = d_orig['mask'].to(device)
            d_orig = torch.load(os.path.join(TERN_DIR, f'layers_{layer_idx}_feed_forward_w3_weight.pt'), weights_only=False)
            mu = d_orig['mask'].to(device)
            d_orig = torch.load(os.path.join(TERN_DIR, f'layers_{layer_idx}_feed_forward_w2_weight.pt'), weights_only=False)
            md = d_orig['mask'].to(device)
            
            W_g_t = f.get_tensor(f'layers.{layer_idx}.feed_forward.w1.weight').float().to(device)
            W_u_t = f.get_tensor(f'layers.{layer_idx}.feed_forward.w3.weight').float().to(device)
            W_d_t = f.get_tensor(f'layers.{layer_idx}.feed_forward.w2.weight').float().to(device)
            with torch.no_grad():
                y_t = F.silu(x_eval @ W_g_t.T) * (x_eval @ W_u_t.T) @ W_d_t.T
                W_g_f = mg * sg_ft.unsqueeze(1).to(device)
                W_u_f = mu * su_ft.unsqueeze(1).to(device)
                W_d_f = md * sd_ft.unsqueeze(1).to(device)
                y_f = F.silu(x_eval @ W_g_f.T) * (x_eval @ W_u_f.T) @ W_d_f.T
                cos_after = F.cosine_similarity(y_f.reshape(-1, 4096), y_t.reshape(-1, 4096), dim=1).mean().item()
            print(f"  Layer {layer_idx:2d}: ya hecho, COS={cos_after:.4f}")
            del W_g_t, W_u_t, W_d_t, mg, mu, md
            torch.cuda.empty_cache()
            continue
        t0 = time.time()
        key_g = f'layers.{layer_idx}.feed_forward.w1.weight'
        key_u = f'layers.{layer_idx}.feed_forward.w3.weight'
        key_d = f'layers.{layer_idx}.feed_forward.w2.weight'
        
        # Teacher weights
        W_g_t = f.get_tensor(key_g).float().to(device)
        W_u_t = f.get_tensor(key_u).float().to(device)
        W_d_t = f.get_tensor(key_d).float().to(device)
        
        # Teacher MLP output
        with torch.no_grad():
            gate_t = F.silu(x_eval @ W_g_t.T)
            up_t = x_eval @ W_u_t.T
            y_t = (gate_t * up_t) @ W_d_t.T
        
        # Mascaras ternarias fijas
        mg, sg0 = load_ternary(key_g)
        mu, su0 = load_ternary(key_u)
        md, sd0 = load_ternary(key_d)
        
        # Escalas entrenables
        sg = torch.nn.Parameter(sg0.clone())
        su = torch.nn.Parameter(su0.clone())
        sd = torch.nn.Parameter(sd0.clone())
        opt = torch.optim.Adam([sg, su, sd], lr=LR)
        
        # Fine-tune
        cos_before = 0
        for step in range(STEPS):
            total_loss = 0
            for x in train_data:
                opt.zero_grad()
                
                W_g = mg * sg.unsqueeze(1)
                W_u = mu * su.unsqueeze(1)
                W_d = md * sd.unsqueeze(1)
                
                gate = F.silu(x @ W_g.T)
                up = x @ W_u.T
                y = (gate * up) @ W_d.T
                
                # Teacher (detached)
                with torch.no_grad():
                    gt = F.silu(x @ W_g_t.T)
                    ut = x @ W_u_t.T
                    yt = (gt * ut) @ W_d_t.T
                
                loss = F.mse_loss(y, yt.detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_([sg, su, sd], 1.0)
                opt.step()
                total_loss += loss.item()
            
            if step == 0:
                with torch.no_grad():
                    W_g0 = mg * sg0.unsqueeze(1).to(device)
                    W_u0 = mu * su0.unsqueeze(1).to(device)
                    W_d0 = md * sd0.unsqueeze(1).to(device)
                    y0 = F.silu(x_eval @ W_g0.T) * (x_eval @ W_u0.T) @ W_d0.T
                    cos_before = F.cosine_similarity(y0.reshape(-1, 4096), y_t.reshape(-1, 4096), dim=1).mean().item()
        
        # Evaluacion final
        with torch.no_grad():
            W_g_f = mg * sg.unsqueeze(1)
            W_u_f = mu * su.unsqueeze(1)
            W_d_f = md * sd.unsqueeze(1)
            y_f = F.silu(x_eval @ W_g_f.T) * (x_eval @ W_u_f.T) @ W_d_f.T
            cos_after = F.cosine_similarity(y_f.reshape(-1, 4096), y_t.reshape(-1, 4096), dim=1).mean().item()
        
        # Guardar escalas fine-tuned
        safe_name_g = key_g.replace('.', '_')
        safe_name_u = key_u.replace('.', '_')
        safe_name_d = key_d.replace('.', '_')
        
        torch.save({'mask': mg.cpu(), 'scale': sg.data.cpu(), 'shape': W_g_t.shape},
                   os.path.join(OUT_DIR, f'{safe_name_g}.pt'))
        torch.save({'mask': mu.cpu(), 'scale': su.data.cpu(), 'shape': W_u_t.shape},
                   os.path.join(OUT_DIR, f'{safe_name_u}.pt'))
        torch.save({'mask': md.cpu(), 'scale': sd.data.cpu(), 'shape': W_d_t.shape},
                   os.path.join(OUT_DIR, f'{safe_name_d}.pt'))
        
        dt = time.time() - t0
        mejora = cos_after - cos_before
        results.append((layer_idx, cos_before, cos_after, mejora))
        
        print(f"  Layer {layer_idx:2d}: COS {cos_before:.4f} -> {cos_after:.4f} ({mejora:+.4f}) [{dt:.0f}s]")
        
        # Free memory
        del W_g_t, W_u_t, W_d_t, mg, mu, md, sg, su, sd
        if device == 'cuda':
            torch.cuda.empty_cache()

# Resumen
print(f"\n{'='*60}")
print("RESUMEN FINE-TUNE")
print(f"{'='*60}")
avg_before = sum(r[1] for r in results) / len(results)
avg_after = sum(r[2] for r in results) / len(results)
print(f"\n  COS promedio antes:  {avg_before:.4f}")
print(f"  COS promedio despues: {avg_after:.4f}")
print(f"  Mejora promedio:      {avg_after - avg_before:+.4f}")
print(f"\n  Peor capa antes:  capa {min(results, key=lambda r: r[1])[0]} (COS={min(results, key=lambda r: r[1])[1]:.4f})")
print(f"  Peor capa despues: capa {min(results, key=lambda r: r[2])[0]} (COS={min(results, key=lambda r: r[2])[2]:.4f})")
print(f"\n  Archivos fine-tuned en: {OUT_DIR}/")
print(f"  Siguiente: Fase 5 - validar perplejidad con modelo completo")
