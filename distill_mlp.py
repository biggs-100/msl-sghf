"""
Destilacion capa por capa: teacher en CPU, seeds en GPU.
Corre en GTX 1650 (4GB) porque nunca carga el modelo completo en GPU.

Flujo:
  1. Carga teacher en CPU (8 GB RAM, tenemos 16 GB)
  2. Hookea MLPs de todas las capas
  3. Forward teacher (CPU, 30s)
  4. Para cada capa, mueve input/output a GPU y entrena seed (5s)
"""

import os, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sg_hf.core import FractalLinear

device = 'cuda'
COMPRESSION = 100.0
SEED_DIR = 'compressed_qwen_distilled'
os.makedirs(SEED_DIR, exist_ok=True)

# 1. Cargar teacher en CPU
print(">>> Cargando teacher Qwen3.5-4B en CPU...")
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3.5-4B', trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map='cpu',
).eval()
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', trust_remote_code=True)
print(f"  Teacher en CPU: {sum(p.numel() for p in model.parameters()):,} params")

# 2. Hookear MLPs de todas las capas
mlp_inputs = {}
mlp_outputs = {}
hooks = []

def make_hook(idx):
    def hook(module, inp, out):
        mlp_inputs[idx] = inp[0].detach().clone().float()
        mlp_outputs[idx] = out[0].detach().clone().float()
    return hook

for i in range(32):
    h = model.model.layers[i].mlp.register_forward_hook(make_hook(i))
    hooks.append(h)

# 3. Forward teacher (CPU)
texts = [
    "The Roman Empire fell because of its military overspending",
    "In the beginning, there was darkness and then came",
    "The general spoke of victory although the battle",
]
print(">>> Forward teacher en CPU (30s)...")
with torch.no_grad():
    model(**tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=64))
print(f"  Capturadas {len(mlp_inputs)} capas")

for h in hooks:
    h.remove()

# Liberar teacher (opcional, nos quedamos con CPU)
# del model  # lo mantenemos por si necesitamos pesos

# 4. Entrenar seeds capa por capa en GPU
print("\n>>> Entrenando seeds capa por capa en GPU...")
results = []

for layer_idx in range(32):
    if layer_idx not in mlp_inputs:
        continue
    
    x_in = mlp_inputs[layer_idx]  # (B, T, 2560) o (B*T, 2560)
    y_tgt = mlp_outputs[layer_idx]  # puede tener forma distinta
    
    # Debug: ver formas reales
    if layer_idx == 0:
        print(f"    x_in.shape={tuple(x_in.shape)}, y_tgt.shape={tuple(y_tgt.shape)}")
    
    # Normalizar formas a 2D (N, 2560), asegurando que N coincida
    if x_in.dim() == 3:
        B, T, C = x_in.shape
        x_2d = x_in.reshape(-1, C)
    else:
        x_2d = x_in
    
    if y_tgt.dim() == 3:
        y_2d = y_tgt.reshape(-1, y_tgt.size(-1))
    else:
        y_2d = y_tgt
    
    # Si N no coincide, tomar el minimo
    min_n = min(x_2d.size(0), y_2d.size(0))
    x_2d = x_2d[:min_n]
    y_2d = y_2d[:min_n]
    
    # Mover a GPU
    x_2d = x_2d.to(device)
    y_2d = y_2d.to(device)
    
    # Crear fractales e inicializar con seeds de peso-MSE
    fl_g = FractalLinear(2560, 9216, compression=COMPRESSION).to(device)
    fl_u = FractalLinear(2560, 9216, compression=COMPRESSION).to(device)
    fl_d = FractalLinear(9216, 2560, compression=COMPRESSION).to(device)
    
    seed_path = f'compressed_qwen_full/model_language_model_layers_{layer_idx}_mlp'
    for fl, name in [(fl_g, 'gate'), (fl_u, 'up'), (fl_d, 'down')]:
        path = f'{seed_path}_{name}_proj_weight.pt'
        if os.path.exists(path):
            sd = torch.load(path, weights_only=False, map_location='cpu')
            for k in ['row_basis', 'col_basis']:
                if k in sd and sd[k].dim() == 2:
                    sd[k] = sd[k].unsqueeze(0)
            fl.load_state_dict(sd, strict=False)
    
    params = list(fl_g.parameters()) + list(fl_u.parameters()) + list(fl_d.parameters())
    opt = torch.optim.Adam(params, lr=1e-4)  # fine-tuning
    
    best_cos = 0
    for step in range(100):
        opt.zero_grad()
        W_g = fl_g._generate_weight().bfloat16()
        W_u = fl_u._generate_weight().bfloat16()
        W_d = fl_d._generate_weight().bfloat16()
        
        x_bf16 = x_2d.bfloat16()
        gate = F.silu(x_bf16 @ W_g.T)
        up = x_bf16 @ W_u.T
        y_pred = (gate * up) @ W_d.T
        
        loss = F.mse_loss(y_pred.float(), y_2d)
        loss.backward()
        opt.step()
        
        if step % 25 == 0:
            cos = F.cosine_similarity(y_pred.float(), y_2d, dim=1).mean().item()
            if cos > best_cos:
                best_cos = cos
    
    # Verificacion final
    with torch.no_grad():
        W_g = fl_g._generate_weight().bfloat16()
        W_u = fl_u._generate_weight().bfloat16()
        W_d = fl_d._generate_weight().bfloat16()
        y_final = (F.silu(x_2d.bfloat16() @ W_g.T) * (x_2d.bfloat16() @ W_u.T)) @ W_d.T
    final_cos = F.cosine_similarity(y_final.float(), y_2d, dim=1).mean().item()
    
    # Guardar seeds destilados
    torch.save(fl_g.state_dict(), os.path.join(SEED_DIR, f'layer_{layer_idx}_gate.pt'))
    torch.save(fl_u.state_dict(), os.path.join(SEED_DIR, f'layer_{layer_idx}_up.pt'))
    torch.save(fl_d.state_dict(), os.path.join(SEED_DIR, f'layer_{layer_idx}_down.pt'))
    
    # Verificacion final
    with torch.no_grad():
        W_g = fl_g._generate_weight().bfloat16()
        W_u = fl_u._generate_weight().bfloat16()
        W_d = fl_d._generate_weight().bfloat16()
        y_final = (F.silu(x_2d.bfloat16() @ W_g.T) * (x_2d.bfloat16() @ W_u.T)) @ W_d.T
    final_cos = F.cosine_similarity(y_final.float(), y_2d, dim=1).mean().item()
    
    results.append({'layer': layer_idx, 'cos': final_cos})
    print(f"  Layer {layer_idx:2d}: COS={final_cos:.4f} (best={best_cos:.4f})")
    
    # Liberar GPU entre capas
    del fl_g, fl_u, fl_d, x_2d, y_2d, y_pred, y_final
    if device == 'cuda':
        torch.cuda.empty_cache()

# Resumen
print(f"\n{'='*50}")
print(f"  RESULTADOS DESTILACION")
print(f"{'='*50}")
avg_cos = sum(r['cos'] for r in results) / len(results) if results else 0
print(f"  Capas: {len(results)}/32")
print(f"  COS promedio: {avg_cos:.4f}")
print(f"  Semillas guardadas en: {SEED_DIR}/")
