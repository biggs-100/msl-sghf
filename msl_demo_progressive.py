"""
MSL Demo Progresivo: entrenamiento escala por escala.

Entrena MSMLP con entrenamiento progresivo:
  1. Escala 1 (rank 1): entrena solo el primer componente
  2. Escala 2 (rank 1+2): congela rank 1, entrena rank 2-3
  3. Escala 3 (rank 1+2+4): congela ranks 1-3, entrena rank 4-7
  4. Escala 4 (rank 1+2+4+8): congela ranks 1-7, entrena rank 8-15

Luego evalua cada nivel de truncamiento SIN reentrenar.
"""

import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sg_hf.msl import MSLinear, MSMLP

# ─── config ───────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
EPOCHS_PER_SCALE = 5   # epocas por escala progresiva
LR = 1e-3
TEST_LR = None         # learning rate para test (None = mismo que LR)

SCALE_RANKS = [4, 8, 16, 32]
N_SCALES = len(SCALE_RANKS)
TOTAL_RANK = sum(SCALE_RANKS)

MLP_SIZES = [784, 512, 256, 10]

print(f"MSL Demo Progresivo — Multi-Scale Linear Perceptron")
print(f"  Device: {DEVICE}")
print(f"  Scales: {SCALE_RANKS} (total rank: {TOTAL_RANK})")
print(f"  MLP: {' -> '.join(str(s) for s in MLP_SIZES)}")
print(f"  Epochs per scale: {EPOCHS_PER_SCALE}")
print()


# ─── datos ────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# ─── helpers ──────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


# ─── progresivo ───────────────────────────────────────────────────
def train_progressive(model, train_loader, test_loader, device):
    """
    Entrena escala por escala, congelando las anteriores.

    Returns: lista de (k, n_params_trainables, train_acc, test_acc)
    """
    results = []

    for k in range(1, N_SCALES + 1):
        # --- configurar para esta escala ---
        model.set_progressive_scale(k)

        # Contar parametros entrenables
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        layer = model.layers[model.msl_indices[0]]
        r_new_start = int(layer._cum_ranks[k-1])
        r_new_end   = int(layer._cum_ranks[k])
        new_rank = r_new_end - r_new_start

        print(f"\n{'='*60}")
        print(f"  Escala {k}/{N_SCALES}  |  rank[{k-1}] = {SCALE_RANKS[k-1]}  |  "
              f"param entrenables: {trainable:,}")
        print(f"  Rango activo: ranks {r_new_start}..{r_new_end-1}  |  "
              f"rango congelado: < {r_new_start}")
        print(f"{'='*60}")

        # --- evaluar antes de entrenar esta escala ---
        acc_before = evaluate(model, test_loader, device)
        print(f"  Test antes: {acc_before:.2f}%")

        # --- optimizador solo para parametros entrenables ---
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR, weight_decay=1e-5
        )

        # --- entrenar ---
        for epoch in range(1, EPOCHS_PER_SCALE + 1):
            loss, train_acc = train_epoch(model, train_loader, opt, device)
            if epoch == 1 or epoch == EPOCHS_PER_SCALE:
                print(f"  Epoch {epoch}/{EPOCHS_PER_SCALE}  |  "
                      f"loss: {loss:.4f}  |  train: {train_acc:.2f}%")

        # --- evaluar despues ---
        acc_after = evaluate(model, test_loader, device)
        delta = acc_after - acc_before
        print(f"  Test despues: {acc_after:.2f}%  (delta: {delta:+.2f}pp)")

        results.append((k, trainable, acc_before, acc_after))

    return results


# ─── evaluar truncamiento ─────────────────────────────────────────
def evaluate_truncation(model, test_loader, device):
    """
    Evalua el modelo con cada nivel de truncamiento SIN reentrenar.
    """
    sep = '-' * 65
    print(f"\n{sep}")
    print(f"  Validacion de truncamiento (alumno SIN fine-tuning)")
    print(f"{sep}")
    print(f"  {'k':>3} | {'Escalas':>8} | {'Rank':>6} | "
          f"{'Compresion':>10} | {'Eficiencia':>10} | {'Accuracy':>8}")
    print(f"{sep}")

    results = []
    all_layers = model.get_msl_layers()

    for k in range(1, N_SCALES + 1):
        model.set_active_scales(k)
        r = all_layers[0].get_active_rank()

        # Compression y eficiencia de la primera capa MSL (representativa)
        total_r = all_layers[0].total_rank
        base_comp = all_layers[0].compression_ratio
        comp = base_comp * (total_r / r)  # ajustado al rank activo
        eff = all_layers[0].get_efficiency()

        acc = evaluate(model, test_loader, device)
        results.append((k, r, comp, eff, acc))

        scales_str = f"1..{k}"
        print(f"  {k:>3} | {scales_str:>8} | {r:>6} | "
              f"{comp:>8.1f}x | {eff:>8.2f}x | {acc:>7.2f}%")

    print(f"{sep}")
    student_acc = results[0][4]
    teacher_acc = results[-1][4]
    print(f"\n  Alumno (k=1, rank 1):      {student_acc:.2f}%  "
          f"@ {results[0][2]:.1f}x compression")
    print(f"  Profesor (k={N_SCALES}, rank {TOTAL_RANK}): {teacher_acc:.2f}%  "
          f"@ {results[-1][2]:.1f}x compression")
    print(f"  Gap: {teacher_acc - student_acc:.2f} puntos")
    print(f"  Compresion relativa: {results[0][2]/results[-1][2]:.1f}x entre extremos")

    return results


# ─── main ─────────────────────────────────────────────────────────
def main():
    model = MSMLP(MLP_SIZES, scale_ranks=SCALE_RANKS).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    comp_params = model.count_compressed_params()
    full_equiv = model.count_full_params()

    print(f"\n  Modelo:")
    print(f"    Parametros totales:      {total_params:,}")
    print(f"    Comprimidos (MSL):       {comp_params:,}")
    print(f"    Equivalentes densos:     {full_equiv:,}")
    print(f"    Compresion teorica:      {model.compression_ratio():.1f}x")
    print()

    # --- 1. Entrenamiento progresivo ---
    print(f"{'='*60}")
    print(f"  FASE 1: ENTRENAMIENTO PROGRESIVO")
    print(f"{'='*60}")
    start = time.time()
    prog_results = train_progressive(model, train_loader, test_loader, DEVICE)
    elapsed = time.time() - start
    total_epochs = EPOCHS_PER_SCALE * N_SCALES
    print(f"\n  Entrenamiento completo: {elapsed:.1f}s ({elapsed/total_epochs:.2f}s/epoch)")

    # --- 2. Evaluacion de truncamiento (key validation) ---
    print(f"\n{'='*60}")
    print(f"  FASE 2: VALIDACION DE TRUNCAMIENTO")
    print(f"{'='*60}")
    trunc_results = evaluate_truncation(model, test_loader, DEVICE)

    # --- 3. Resumen ---
    print(f"\n{'='*60}")
    print(f"  RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"  Arquitectura: MSMLP con escalas {SCALE_RANKS}")
    print(f"  MLP: {' -> '.join(str(s) for s in MLP_SIZES)}")
    print(f"  Compresion profesor: {trunc_results[-1][2]:.1f}x  "
          f"@{trunc_results[-1][4]:.2f}%")
    print(f"  Compresion alumno:   {trunc_results[0][2]:.1f}x  "
          f"@{trunc_results[0][4]:.2f}%")
    print(f"  Gap alumno-profesor: {trunc_results[-1][4] - trunc_results[0][4]:.2f}pp")
    print(f"  Sin fine-tuning:     SI (truncacion directa)")
    print(f"\n  Demo completa.")

    # Guardar metrica clave
    mem_info = {
        'teacher_acc': trunc_results[-1][4],
        'student_acc': trunc_results[0][4],
        'gap': trunc_results[-1][4] - trunc_results[0][4],
        'teacher_compression': trunc_results[-1][2],
        'student_compression': trunc_results[0][2],
    }
    return mem_info


if __name__ == '__main__':
    main()
