"""
Test en CPU: carga Qwen3.5-4B entero en CPU, corre un forward
con texto real, captura MLPs y compara con fractal.

Asi evitamos el limite de VRAM de la GTX 1650 (4 GB).
"""

import os, json, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sg_hf.core import FractalLinear

device = 'cpu'
COMPRESSION = 100.0
SEED_DIR = 'compressed_qwen_full'
CACHE_DIR = os.path.expanduser(r'~\.cache\huggingface\hub\models--Qwen--Qwen3.5-4B\snapshots\851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')


def load_mlp_fractal(layer_idx):
    keys = ['mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight']
    fractals = {}
    for key in keys:
        full_name = f'model.language_model.layers.{layer_idx}.{key}'
        safe_name = full_name.replace('.', '_').replace('/', '_')
        path = os.path.join(SEED_DIR, f'{safe_name}.pt')
        if not os.path.exists(path):
            return None
        sd = torch.load(path, weights_only=False, map_location='cpu')
        in_f, out_f = (2560, 9216) if 'down' not in key else (9216, 2560)
        fl = FractalLinear(in_f, out_f, compression=COMPRESSION)
        fl.load_state_dict(sd)
        fractals[key] = fl
    return fractals


def test():
    print(f"Device: CPU (evitando VRAM limitada)")
    print(f"RAM disponible: ~16 GB, modelo ~8 GB en FP16\n")

    # Cargar teacher
    print(">>> Cargando Qwen3.5-4B en CPU (puede tomar ~30s)...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen3.5-4B',
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map='cpu',
    )
    model.eval()
    print(f"  Modelo cargado: {sum(p.numel() for p in model.parameters()):,} params")

    # Prompt corto (pocos tokens = rapido en CPU)
    prompt = "The Roman Empire fell because"
    inputs = tokenizer(prompt, return_tensors='pt')
    print(f"  Prompt: '{prompt}' ({inputs['input_ids'].shape[1]} tokens)")

    # Hookear MLPs
    mlp_inputs = {}
    mlp_outputs = {}
    hooks = []

    def make_hook(idx):
        def hook(module, inp, out):
            mlp_inputs[idx] = inp[0].detach().float()
            mlp_outputs[idx] = out[0].detach().float()
        return hook

    for i in range(32):
        if hasattr(model.model.layers[i], 'mlp'):
            h = model.model.layers[i].mlp.register_forward_hook(make_hook(i))
            hooks.append(h)

    # Forward teacher (solo un paso, CPU)
    print("\n>>> Running teacher forward (CPU, 1 paso)...")
    with torch.inference_mode():
        logits = model(inputs['input_ids']).logits

    # Remover hooks
    for h in hooks:
        h.remove()

    # Top-5 predictions
    t_logits = logits[0, -1, :]
    t_probs = F.softmax(t_logits, dim=0)
    t_top5 = t_probs.topk(5)
    print(f"\nTeacher top-5:")
    for i in range(5):
        tok = tokenizer.decode([t_top5.indices[i].item()])
        print(f"  {i+1}. '{tok}' ({t_top5.values[i].item():.3f})")

    # Comparar MLPs
    print(f"\n>>> Comparando MLPs con seeds fractal...")
    results = []

    for i in range(32):
        if i not in mlp_inputs:
            continue

        x = mlp_inputs[i]  # (B, T, 2560)
        y_t = mlp_outputs[i]

        fractals = load_mlp_fractal(i)
        if fractals is None:
            continue

        # Asegurar forma consistente (sin batch duplicado)
        if x.dim() == 3:
            B, T, C = x.shape
            x_2d = x.view(-1, C)
        else:
            x_2d = x

        # Forward fractal (CPU)
        with torch.no_grad():
            W_g = fractals['mlp.gate_proj.weight']._generate_weight().float()
            W_u = fractals['mlp.up_proj.weight']._generate_weight().float()
            W_d = fractals['mlp.down_proj.weight']._generate_weight().float()

        gate = F.silu(x_2d @ W_g.T)
        up = x_2d @ W_u.T
        y_s = (gate * up) @ W_d.T  # (B*T, 2560) o (T, 2560)

        # Forma del teacher: hook output
        y_t_2d = y_t.view(-1, y_t.size(-1))

        mse = F.mse_loss(y_s, y_t_2d).item()
        cos = F.cosine_similarity(y_s, y_t_2d, dim=1).mean().item()

        results.append({'layer': i, 'mse': mse, 'cos': cos})
        status = '✅' if cos > 0.95 else '🟡' if cos > 0.8 else '❌'
        print(f"  Layer {i:2d}: MSE={mse:.6f}  COS={cos:.4f}  {status}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTADOS FINALES")
    print(f"{'='*60}")
    if results:
        avg_mse = sum(r['mse'] for r in results) / len(results)
        avg_cos = sum(r['cos'] for r in results) / len(results)
        min_cos = min(r['cos'] for r in results)
        max_cos = max(r['cos'] for r in results)
        print(f"  MLPs comparadas: {len(results)}/{32}")
        print(f"  MSE promedio:    {avg_mse:.6f}")
        print(f"  Cosine promedio: {avg_cos:.4f}")
        print(f"  Min cos:         {min_cos:.4f}")
        print(f"  Max cos:         {max_cos:.4f}")
        good = sum(1 for r in results if r['cos'] > 0.95)
        print(f"  Capas con cos>0.95: {good}/{len(results)}")
        print(f"\n  VEREDICTO: {'✅ FUNCIONA' if avg_cos > 0.95 else '🟡 PARCIAL' if avg_cos > 0.8 else '❌ REVISAR'}")


if __name__ == '__main__':
    test()
