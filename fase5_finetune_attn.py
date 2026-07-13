"""
Fase 5: Fine-tune escalas ternarias de atencion.

Q, K, V, O projection para todas las 32 capas.
"""

import torch, torch.nn.functional as F, time, os
from safetensors import safe_open

device = 'cuda'
CONSOLIDATED = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1\consolidated.safetensors'
TERN_DIR = 'mistral_ternario'
OUT_DIR = 'mistral_ternario_ft'
os.makedirs(OUT_DIR, exist_ok=True)

STEPS = 30
LR = 1e-3

print("=" * 60)
print("FASE 5: Fine-tune escalas ternarias de ATENCION")
print(f"  Steps: {STEPS}, LR: {LR}")
print(f"  Salida: {OUT_DIR}/")
print("=" * 60)

torch.manual_seed(42)
x_train = [torch.randn(1, 32, 4096, device=device) for _ in range(3)]
torch.manual_seed(42)
x_eval = torch.randn(1, 32, 4096, device=device)

import glob
done = set()
for fpath in glob.glob(os.path.join(OUT_DIR, 'layers_*_attention_wq_weight.pt')):
    parts = os.path.basename(fpath).split('_')
    done.add(int(parts[1]))
print(f"  Capas ya hechas: {sorted(done) if done else 'ninguna'}")

with safe_open(CONSOLIDATED, framework='pt', device='cpu') as f:
    for layer_idx in range(32):
        if layer_idx in done:
            print(f"  Layer {layer_idx}: ya hecho, skip")
            continue
        
        t0 = time.time()
        
        keys = {
            'wq': f'layers.{layer_idx}.attention.wq.weight',
            'wk': f'layers.{layer_idx}.attention.wk.weight',
            'wv': f'layers.{layer_idx}.attention.wv.weight',
            'wo': f'layers.{layer_idx}.attention.wo.weight',
        }
        
        teacher = {}
        masks = {}
        scales0 = {}
        
        for name, key in keys.items():
            W = f.get_tensor(key).float().to(device)
            teacher[name] = W
            
            d = torch.load(os.path.join(TERN_DIR, f'{key.replace(".", "_")}.pt'), weights_only=False)
            masks[name] = d['mask'].to(device)
            scales0[name] = d['scale'].to(device)
        
        # Eval antes
        with torch.no_grad():
            q_t = x_eval @ teacher['wq'].T
            k_t = x_eval @ teacher['wk'].T
            v_t = x_eval @ teacher['wv'].T
            o_t_input = torch.randn(1, 32, 4096, device=device) @ teacher['wo'].T
            
            cos_q_before = F.cosine_similarity(q_t.reshape(-1, 4096), 
                (x_eval @ (masks['wq'] * scales0['wq'].unsqueeze(1).to(device)).T).reshape(-1, 4096), dim=1).mean().item()
        
        # Entrenar escalas
        scales = {name: torch.nn.Parameter(scales0[name].clone().to(device)) for name in keys}
        opt = torch.optim.Adam(list(scales.values()), lr=LR)
        
        for step in range(STEPS):
            total_loss = 0
            for x in x_train:
                opt.zero_grad()
                loss = 0
                for name in ['wq', 'wk', 'wv']:
                    W_s = masks[name] * scales[name].unsqueeze(1)
                    out_s = x @ W_s.T
                    out_t = x @ teacher[name].T
                    loss += F.mse_loss(out_s, out_t.detach())
                
                # wo: entrada diferente
                x_rand = torch.randn_like(x)
                W_o_s = masks['wo'] * scales['wo'].unsqueeze(1)
                out_os = x_rand @ W_o_s.T
                out_ot = x_rand @ teacher['wo'].T
                loss += F.mse_loss(out_os, out_ot.detach())
                
                loss.backward()
                opt.step()
                total_loss += loss.item()
        
        # Eval despues
        with torch.no_grad():
            q_s = x_eval @ (masks['wq'] * scales['wq'].unsqueeze(1)).T
            cos_q_after = F.cosine_similarity(q_s.reshape(-1, 4096), q_t.reshape(-1, 4096), dim=1).mean().item()
        
        # Guardar
        for name in keys:
            key = keys[name]
            fname = key.replace('.', '_')
            torch.save({
                'mask': masks[name].cpu(),
                'scale': scales[name].data.cpu(),
                'shape': teacher[name].shape,
            }, os.path.join(OUT_DIR, f'{fname}.pt'))
        
        dt = time.time() - t0
        print(f"  Layer {layer_idx:2d}: Q COS {cos_q_before:.4f} -> {cos_q_after:.4f} ({cos_q_after-cos_q_before:+.4f}) [{dt:.0f}s]")
        
        del teacher, masks, scales
        torch.cuda.empty_cache()

print(f"\n✅ Atencion fine-tuned. Archivos en: {OUT_DIR}/")
