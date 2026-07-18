"""
MSL v2 Demo: Entrenamiento CONJUNTO con regularizacion espectral.

En vez de entrenamiento progresivo, entrenamos todas las escalas
simultaneamente con losses de sorting + orthogonality + L1 espectral.

Esto evita el problema de rotacion de subespacios del QR congelado
y produce mejores alumnos por truncamiento.
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
from sg_hf.msl_v2 import MSLinear, MSMLP

# ─── config ───────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3

# Pesos de regularizacion
ALPHA_SORT = 0.5      # sorting de valores singulares
ALPHA_ORTH = 0.1      # ortogonalidad de U/V
ALPHA_L1_S = 1e-4     # L1 en s (esparsidad en componentes tardios)

# Escalas: primera capa mas ancha, segunda mas angosta
SCALE_CONFIG = {
    'first':  [16, 32, 32],    # 784->512: mas rank para input dimensional
    'hidden': [8, 16, 32],      # 512->256: rank medio
}

MLP_SIZES = [784, 512, 256, 10]

print(f"MSL v2 Demo — Entrenamiento Conjunto con Regularizacion")
print(f"  Device: {DEVICE}")
print(f"  Scales config: {SCALE_CONFIG}")
print(f"  MLP: {' -> '.join(str(s) for s in MLP_SIZES)}")
print(f"  Reg: sort={ALPHA_SORT}, orth={ALPHA_ORTH}, l1_s={ALPHA_L1_S}")
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
def train_epoch(model, loader, optimizer, device, reg_kwargs):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_reg = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        out = model(x)
        ce = F.cross_entropy(out, y)
        reg = model.compute_regularization(**reg_kwargs)
        loss = ce + reg['total']

        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_ce += ce.item() * bs
        total_reg += reg['total'].item() * bs
        correct += (out.argmax(1) == y).sum().item()
        total += bs

    n = max(total, 1)
    return {
        'loss': total_loss / n,
        'ce': total_ce / n,
        'reg': total_reg / n,
        'acc': 100.0 * correct / n,
    }


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
    return 100.0 * correct / max(total, 1)


@torch.inference_mode()
def evaluate_all_scales(model, loader, device):
    """
    Evalua TODOS los niveles de truncamiento SIN reentrenar.
    Esta es la validacion clave del concepto MSL.
    """
    msl_layers = model.get_msl_layers()
    n_scales = msl_layers[0].n_scales

    sep = '-' * 65
    print(f"\n{sep}")
    print(f"  Truncamiento progresivo (SIN fine-tuning)")
    print(f"{sep}")
    print(f"  {'k':>3} | {'Rank L1':>8} | {'Rank L2':>8} | "
          f"{'Compresion':>10} | {'Eficiencia':>10} | {'Accuracy':>8}")
    print(f"{sep}")

    results = []
    for k in range(1, n_scales + 1):
        model.set_active_scales(k)
        r1 = msl_layers[0].get_active_rank()
        r2 = msl_layers[1].get_active_rank()

        # Compresion combinada
        comp = model.compression_ratio() * (
            msl_layers[0]._R / r1 * msl_layers[1]._R / r2
        ) ** 0.5  # media geometrica

        # Efficiency de la primera capa (representativa)
        eff = msl_layers[0].get_efficiency()

        acc = evaluate(model, loader, device)
        results.append((k, r1, r2, comp, eff, acc))

        comp_str = f"{comp:.1f}x"
        print(f"  {k:>3} | {r1:>8} | {r2:>8} | "
              f"{comp_str:>10} | {eff:>8.2f}x | {acc:>7.2f}%")

    print(f"{sep}")
    print(f"  Gap alumno (k=1) -> profesor (k={n_scales}): "
          f"{results[-1][5] - results[0][5]:.2f}pp")
    print(f"  Sin fine-tuning: SI (truncacion directa)")

    return results


@torch.inference_mode()
def check_singular_values(model):
    """Muestra los primeros y ultimos valores singulares de cada capa."""
    print(f"\n  Valores singulares:")
    for i, layer in enumerate(model.get_msl_layers()):
        s = layer.s.detach().cpu()
        top5 = s[:5].tolist()
        bot5 = s[-5:].tolist()
        ordered = all(s[j] >= s[j+1] - 0.001 for j in range(len(s)-1))
        print(f"    Capa {i+1}: top={[f'{v:.3f}' for v in top5]}, "
              f"bot={[f'{v:.3f}' for v in bot5]}, "
              f"ordenado={ordered}")


# ─── main ─────────────────────────────────────────────────────────
def main():
    model = MSMLP(MLP_SIZES, scale_config=SCALE_CONFIG).to(DEVICE)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parametros totales: {params:,}")
    print(f"  Compresion teorica: {model.compression_ratio():.1f}x")
    print()

    # --- Entrenar ---
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    reg_kwargs = {
        'alpha_sort': ALPHA_SORT,
        'alpha_orth': ALPHA_ORTH,
        'alpha_l1_s': ALPHA_L1_S,
    }

    # Evaluacion inicial (random)
    model.set_active_scales(len(SCALE_CONFIG.get('first', [8,16,32])))
    acc_init = evaluate(model, test_loader, DEVICE)
    print(f"  Test inicial (random): {acc_init:.2f}%\n")

    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        stats = train_epoch(model, train_loader, opt, DEVICE, reg_kwargs)
        if epoch == 1 or epoch % 5 == 0:
            print(f"  Epoch {epoch:2d}/{EPOCHS}  |  "
                  f"loss: {stats['loss']:.4f}  |  "
                  f"ce: {stats['ce']:.4f}  |  "
                  f"reg: {stats['reg']:.4f}  |  "
                  f"train: {stats['acc']:.2f}%")
    elapsed = time.time() - start
    print(f"\n  Entrenamiento: {elapsed:.1f}s ({elapsed/EPOCHS:.2f}s/epoch)")

    # --- Profesor completo ---
    teacher_acc = evaluate(model, test_loader, DEVICE)
    print(f"\n  Profesor (todas escalas): {teacher_acc:.2f}%")

    # --- Validacion de truncamiento ---
    results = evaluate_all_scales(model, test_loader, DEVICE)

    # --- Singular values ---
    check_singular_values(model)

    # --- Resumen ---
    print(f"\n{'='*60}")
    print(f"  RESUMEN MSL v2")
    print(f"{'='*60}")
    print(f"  Profesor:  {results[-1][5]:.2f}% @ {results[-1][3]:.1f}x compression")
    print(f"  Alumno:    {results[0][5]:.2f}% @ {results[0][3]:.1f}x compression")
    print(f"  Gap:       {results[-1][5] - results[0][5]:.2f}pp")
    print(f"  Metodo:    entrenamiento conjunto con regularizacion espectral")
    print(f"  Sin FT:    SI (truncacion directa)")

    # Guardar metrica
    return {
        'teacher_acc': results[-1][5],
        'student_acc': results[0][5],
        'gap': results[-1][5] - results[0][5],
        'teacher_comp': results[-1][3],
        'student_comp': results[0][3],
        'results': results,
    }


if __name__ == '__main__':
    main()
