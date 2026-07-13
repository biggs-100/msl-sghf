"""
Destilacion de MLPs con hooks: captura la entrada real al MLP
durante el forward del teacher y entrena los seeds.

5 capas por ejecucion para no saturar CPU.
"""

import os, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sg_hf.core import FractalLinear

device = 'cpu'
COMPRESSION = 100.0
SEED_DIR = 'compressed_qwen_distilled'
os.makedirs(SEED_DIR, exist_ok=True)

# Cargar teacher
print(">>> Cargando teacher Qwen3.5-4B...")
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3.5-4B', trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map='cpu',
).eval()
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', trust_remote_code=True)

# Texto de calibracion
text = "The Roman Empire fell because of its"
inputs = tokenizer(text, return_tensors='pt')

# Hookear MLP inputs y outputs
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

# Forward teacher
print(f"\n>>> Forward teacher...")
with torch.no_grad():
    model(inputs['input_ids'])

for h in hooks:
    h.remove()

print(f"  Hookeadas {len(mlp_inputs)} capas")

# Destilar capa por capa
for layer_idx in range(5):  # primeras 5 capas
    if layer_idx not in mlp_inputs:
        continue

    x = mlp_inputs[layer_idx]  # (B, T, 2560) - entrada REAL al MLP
    y_target = mlp_outputs[layer_idx]  # (B, T, 2560) - salida REAL del MLP

    print(f"\n>>> Layer {layer_idx}/32 | x.shape={list(x.shape)}")

    # Crear fractales con semilla de pesos viejos
    fl_g = FractalLinear(2560, 9216, compression=COMPRESSION)
    fl_u = FractalLinear(2560, 9216, compression=COMPRESSION)
    fl_d = FractalLinear(9216, 2560, compression=COMPRESSION)

    old = f'compressed_qwen_full/model_language_model_layers_{layer_idx}_mlp'
    if os.path.exists(old + '_gate_proj_weight.pt'):
        fl_g.load_state_dict(torch.load(old + '_gate_proj_weight.pt', weights_only=False, map_location='cpu'))
        fl_u.load_state_dict(torch.load(old + '_up_proj_weight.pt', weights_only=False, map_location='cpu'))
        fl_d.load_state_dict(torch.load(old + '_down_proj_weight.pt', weights_only=False, map_location='cpu'))

    params = list(fl_g.parameters()) + list(fl_u.parameters()) + list(fl_d.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    best_mse = float('inf')
    cos_best = 0

    for step in range(100):
        opt.zero_grad()

        W_g = fl_g._generate_weight().bfloat16()
        W_u = fl_u._generate_weight().bfloat16()
        W_d = fl_d._generate_weight().bfloat16()

        x_bf16 = x.bfloat16()
        gate = F.silu(x_bf16 @ W_g.T)
        up = x_bf16 @ W_u.T
        y_pred = (gate * up) @ W_d.T

        loss = F.mse_loss(y_pred.float(), y_target)
        loss.backward()
        opt.step()

        if loss.item() < best_mse:
            best_mse = loss.item()

        if step % 20 == 0 or step == 99:
            cos = F.cosine_similarity(
                y_pred.float().view(-1, y_pred.size(-1)),
                y_target.view(-1, y_target.size(-1)), dim=1
            ).mean().item()
            if cos > cos_best:
                cos_best = cos
            print(f"  Step {step:3d}: MSE={loss.item():.6f}  COS={cos:.4f}  best={best_mse:.6f}")

    # Guardar seeds destilados
    torch.save(fl_g.state_dict(), os.path.join(SEED_DIR, f'layer_{layer_idx}_gate.pt'))
    torch.save(fl_u.state_dict(), os.path.join(SEED_DIR, f'layer_{layer_idx}_up.pt'))
    torch.save(fl_d.state_dict(), os.path.join(SEED_DIR, f'layer_{layer_idx}_down.pt'))

    # Verificacion final
    with torch.no_grad():
        W_g = fl_g._generate_weight().bfloat16()
        W_u = fl_u._generate_weight().bfloat16()
        W_d = fl_d._generate_weight().bfloat16()
        gate = F.silu(x.bfloat16() @ W_g.T)
        up = x.bfloat16() @ W_u.T
        y_final = (gate * up) @ W_d.T

    final_mse = F.mse_loss(y_final.float(), y_target).item()
    final_cos = F.cosine_similarity(
        y_final.float().view(-1, y_final.size(-1)),
        y_target.view(-1, y_target.size(-1)), dim=1
    ).mean().item()

    # Weight MSE
    w_mse = F.mse_loss(fl_g._generate_weight().float(), model.model.layers[layer_idx].mlp.gate_proj.weight.float()).item()

    print(f"  >> RESULT: MSE={final_mse:.6f}  COS={final_cos:.4f}  "
          f"W_MSE={w_mse:.8f}  (ANTES: COS~0.02)")

print(f"\nDone. Seeds en {SEED_DIR}/")
