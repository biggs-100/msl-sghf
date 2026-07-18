"""
MSL Transformer Demo: entrena un GPT chico con MSL FFN y valida truncamiento.

Fase 2 del plan MSL: reemplazar FFN de transformer con MSLinear.
Validacion clave: truncar escalas produce un modelo funcional SIN fine-tuning.

Dataset: TinyStories (descarga automatica de HF) o datos sinteticos.
Modelo: GPT con hidden=128, 4 layers, 4 heads, vocab=96 (chars).
"""

import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sg_hf.msl_transformer import MSL_GPT, create_msl_gpt, get_msl_ffn_config

# ─── config ───────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# Modelo
HIDDEN = 128
N_LAYERS = 4
N_HEADS = 4
CONTEXT = 128
VOCAB_SIZE = 96  # printable ASCII

# Escalas
FFN_PROFILE = 'small'  # [8, 16, 32] para gate/up, [16, 32, 64] para down

# Training
BATCH_SIZE = 64
TRAIN_STEPS = 1000
LR = 3e-4
WARMUP = 100
REG_SORT = 0.5
REG_ORTH = 0.1
REG_L1_S = 1e-4

# Data
DATA_FILE = None  # None = generar datos sinteticos de TinyStories


# ─── Datos sinteticos (TinyStories en texto) ──────────────────────
# Descargamos TinyStories si no existe localmente
TINYSTORIES_PATH = os.path.join(os.path.dirname(__file__), 'data', 'tinystories.txt')

def download_tinystories():
    """Descarga TinyStories como texto plano."""
    os.makedirs(os.path.dirname(TINYSTORIES_PATH), exist_ok=True)
    if not os.path.exists(TINYSTORIES_PATH):
        print("  Descargando TinyStories...")
        try:
            import requests
            url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.txt"
            r = requests.get(url, timeout=60)
            with open(TINYSTORIES_PATH, 'wb') as f:
                f.write(r.content)
            print(f"  Descargado: {TINYSTORIES_PATH}")
        except Exception as e:
            print(f"  No se pudo descargar: {e}")
            print("  Usando datos sinteticos...")
            return False
    return os.path.exists(TINYSTORIES_PATH)


def load_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Carga datos y devuelve tensores de entrenamiento y validacion."""
    # Intentar cargar TinyStories real
    if DATA_FILE and os.path.exists(DATA_FILE):
        text = open(DATA_FILE, 'r', encoding='utf-8').read()
    else:
        # Verificar si tenemos un TinyStories valido
        have_real = False
        if os.path.exists(TINYSTORIES_PATH):
            with open(TINYSTORIES_PATH, 'r', encoding='utf-8') as f:
                text = f.read()
            if len(text) > 10000:
                have_real = True

        if not have_real:
            # Generar datos sinteticos: stories variadas
            print("  Generando datos sinteticos (~200K chars)...")
            names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eva', 'Frank',
                     'Luna', 'Max', 'Bella', 'Oscar', 'Mia', 'Leo']
            animals = ['dog', 'cat', 'bird', 'fish', 'rabbit', 'fox',
                       'bear', 'frog', 'owl', 'mouse']
            actions = ['found a magical stone', 'discovered a hidden cave',
                       'learned to fly', 'saved the day', 'made a new friend',
                       'built a treehouse', 'found a treasure map',
                       'solved a mystery', 'planted a magic seed',
                       'crossed a rainbow bridge']
            places = ['forest', 'garden', 'mountain', 'river', 'village',
                      'castle', 'island', 'valley', 'meadow', 'cave']
            samples = []
            import random
            random.seed(42)
            for i in range(5000):
                n = names[i % len(names)]
                a = animals[i % len(animals)]
                act = random.choice(actions)
                pl = random.choice(places)
                samples.append(
                    f"Once upon a time, {n} was a little {a} who lived in a {pl}. "
                    f"One day, {n} {act}. "
                    f"It was an amazing adventure! "
                    f"{n} learned that being brave is the most important thing. "
                    f"The end.\n\n"
                )
            text = ''.join(samples)
        else:
            print(f"  Usando TinyStories ({len(text):,} chars)")

    # Char-level encoding
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    V = len(chars)

    # Split
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    print(f"  Vocab size: {V}")
    print(f"  Train chars: {len(train_data):,}")
    print(f"  Val chars: {len(val_data):,}")

    return train_data, val_data, V, stoi, itos


def get_batch(data: torch.Tensor, batch_size: int, context: int,
              device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Saca un batch aleatorio del dataset."""
    ix = torch.randint(len(data) - context, (batch_size,))
    x = torch.stack([data[i:i+context] for i in ix])
    y = torch.stack([data[i+1:i+context+1] for i in ix])
    return x.to(device), y.to(device)


# ─── Evaluacion ───────────────────────────────────────────────────

@torch.inference_mode()
def evaluate_loss(model, data, batch_size, context, device, n_batches=20):
    """Evalua loss promedio en n_batches."""
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, context, device)
        logits, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


@torch.inference_mode()
def evaluate_all_scales(model, train_data, val_data, batch_size,
                        context, device):
    """Evalua el modelo en cada nivel de truncamiento."""
    msl = model.get_msl_layers()
    n_scales = msl[0].n_scales

    print(f"\n{'='*65}")
    print(f"  VALIDACION DE TRUNCAMIENTO (SIN fine-tuning)")
    print(f"{'='*65}")
    print(f"  {'k':>3} | {'Rank por capa':>13} | {'Params MSL':>10} | "
          f"{'Compresion':>10} | {'Train loss':>10} | {'Val loss':>10}")
    print(f"{'-'*65}")

    results = []
    total_msl_full = model.count_params()['msl']

    for k in range(1, n_scales + 1):
        model.set_active_scales(k)

        # Contar parametros activos en MSL
        active_params = 0
        for layer in msl:
            r = layer.cum_ranks[k]
            # U: (out, r), s: (r,), V: (r, in)
            active_params += (layer.out_features * r + r + r * layer.in_features)
        n_msl_layers = len(msl)
        layers_per_ffn = 3
        n_ffn = n_msl_layers // layers_per_ffn
        rank_str = f"{msl[0].cum_ranks[k]}"
        compression = total_msl_full / max(active_params, 1)

        train_loss = evaluate_loss(model, train_data, batch_size, context, device, n_batches=10)
        val_loss = evaluate_loss(model, val_data, batch_size, context, device, n_batches=10)
        ppl = math.exp(val_loss)

        results.append((k, rank_str, active_params, compression, train_loss, val_loss, ppl))
        print(f"  {k:>3} | {rank_str:>13} | {active_params:>10,} | "
              f"{compression:>8.1f}x | {train_loss:>10.4f} | {val_loss:>10.4f}")

    print(f"{'-'*65}")
    best = results[-1]  # teacher
    student = results[0]

    print(f"\n  Profesor (k={n_scales}): val_loss={best[5]:.4f}, ppl={best[6]:.2f}")
    print(f"  Alumno (k=1):           val_loss={student[5]:.4f}, ppl={student[6]:.2f}")
    print(f"  Gap en loss: {student[5] - best[5]:.4f}")
    print(f"  Gap en ppl:  {student[6] - best[6]:.2f}")
    print(f"  Compresion relativa: {student[3]/best[3]:.1f}x entre extremos")
    print(f"  Sin fine-tuning:     SI (truncacion directa)")

    return results


# ─── Entrenamiento ────────────────────────────────────────────────

def train_step(model, x, y, optimizer, reg_kwargs):
    model.train()
    optimizer.zero_grad()

    logits, loss, reg = model(x, y, return_reg=True, reg_kwargs=reg_kwargs)
    total_loss = loss + reg['total']

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        'loss': loss.item(),
        'reg_sort': reg['sort'].item(),
        'reg_orth': reg['orth'].item(),
        'reg_l1_s': reg['l1_s'].item(),
        'reg_total': reg['total'].item(),
    }


def main():
    print(f"\n{'='*60}")
    print(f"  MSL TRANSFORMER DEMO — FASE 2")
    print(f"{'='*60}")
    print(f"  hidden={HIDDEN}, layers={N_LAYERS}, heads={N_HEADS}")
    print(f"  context={CONTEXT}, profile={FFN_PROFILE}")
    print(f"  steps={TRAIN_STEPS}, batch={BATCH_SIZE}, lr={LR}")
    print()

    # ─── Cargar datos ─────────────────────────────────────────────
    print("Cargando datos...")
    train_data, val_data, V, stoi, itos = load_data()

    # Ajustar vocab_size al real
    actual_vocab = V

    # ─── Crear modelo ─────────────────────────────────────────────
    ffn_cfg = get_msl_ffn_config(HIDDEN, 4 * HIDDEN, FFN_PROFILE)
    print(f"\nConfig de escalas FFN:")
    print(f"  Gate/up: {ffn_cfg['gate']}")
    print(f"  Down:    {ffn_cfg['down']}")

    model = MSL_GPT(
        vocab_size=actual_vocab,
        hidden=HIDDEN,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        context=CONTEXT,
        ffn_scale_config=ffn_cfg,
    ).to(DEVICE)

    params = model.count_params()
    n_msl = len(model.get_msl_layers())
    print(f"\nParametros:")
    print(f"  Total:     {params['total']:,}")
    print(f"  MSL capas: {params['msl']:,} ({n_msl} layers)")
    print(f"  No-MSL:    {params['non_msl']:,}")
    msl_comp = params['msl'] / sum(p.numel() for mod in model.get_msl_layers() for p in mod.parameters())
    print(f"  Compresion teorica MSL: {msl_comp:.1f}x")

    # ─── Evaluacion inicial ───────────────────────────────────────
    init_loss = evaluate_loss(model, val_data, BATCH_SIZE, CONTEXT, DEVICE, n_batches=5)
    print(f"\n  Loss inicial (random): {init_loss:.4f} (ppl={math.exp(init_loss):.2f})")

    # ─── Entrenar ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ENTRENAMIENTO")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_STEPS)
    reg_kwargs = {
        'alpha_sort': REG_SORT,
        'alpha_orth': REG_ORTH,
        'alpha_l1_s': REG_L1_S,
    }

    # Usar todas las escalas durante entrenamiento
    model.set_active_scales(model.n_scales)

    start = time.time()
    print_step = max(1, TRAIN_STEPS // 10)

    for step in range(1, TRAIN_STEPS + 1):
        x, y = get_batch(train_data, BATCH_SIZE, CONTEXT, DEVICE)
        stats = train_step(model, x, y, optimizer, reg_kwargs)
        scheduler.step()

        if step == 1 or step % print_step == 0:
            val_loss = evaluate_loss(model, val_data, BATCH_SIZE, CONTEXT, DEVICE, n_batches=5)
            print(f"  Step {step:5d}/{TRAIN_STEPS}  |  "
                  f"loss: {stats['loss']:.4f}  |  "
                  f"reg: {stats['reg_total']:.4f}  |  "
                  f"val: {val_loss:.4f}  |  "
                  f"ppl: {math.exp(val_loss):.2f}")

    elapsed = time.time() - start
    print(f"\n  Entrenamiento: {elapsed:.1f}s ({elapsed/TRAIN_STEPS*1000:.1f}ms/step)")

    # ─── Evaluar profesor ─────────────────────────────────────────
    model.set_active_scales(model.n_scales)
    teacher_loss = evaluate_loss(model, val_data, BATCH_SIZE, CONTEXT, DEVICE)
    print(f"\n  Profesor final: val_loss={teacher_loss:.4f}, ppl={math.exp(teacher_loss):.2f}")

    # ─── Validar truncamiento ─────────────────────────────────────
    results = evaluate_all_scales(model, train_data, val_data, BATCH_SIZE, CONTEXT, DEVICE)

    # ─── Generar texto con profesor y alumno ──────────────────────
    print(f"\n{'='*60}")
    print(f"  GENERACION DE TEXTO")
    print(f"{'='*60}")

    context_str = "Once upon a time"
    context_ids = torch.tensor(
        [[stoi.get(c, 0) for c in context_str]], dtype=torch.long
    ).to(DEVICE)

    for label, k in [("ALUMNO (k=1)", 1), ("PROFESOR", model.n_scales)]:
        model.set_active_scales(k)
        out = model.generate(context_ids, max_new_tokens=64, temperature=0.8)
        generated = ''.join(itos[int(i)] for i in out[0].cpu())
        print(f"\n  [{label}]:")
        print(f"    {generated}")

    print(f"\n  Demo completa.")


if __name__ == '__main__':
    main()
