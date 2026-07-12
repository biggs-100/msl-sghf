"""
Destilacion hibrida: activaciones + pesos del teacher.

La clave: durante la destilacion TENEMOS los pesos del teacher,
asi que el seed aprende comparando directamente los pesos generados,
no solo las activaciones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sg_hf.core import FractalMLP


def hybrid_distillation_loss(
    teacher: nn.Module,
    student: FractalMLP,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Perdida hibrida: activaciones (indirecta) + pesos (directa).

    1. Forward teacher -> guarda activaciones y logits
    2. Forward student -> guarda activaciones y logits
    3. Weight-MSE: compara pesos generados del seed vs teacher
    4. Activation-MSE: compara activaciones capa por capa
    5. Cross-entropy: compara logits con labels
    """
    # --- teacher forward (frozen) ---
    with torch.no_grad():
        teacher_out, teacher_acts = teacher(x, return_activations=True)

    # --- student forward ---
    student_out, student_acts = student(x, return_activations=True)

    # --- 1. Weight loss: comparar pesos generados vs teacher ---
    # Teacher weights: fc1.weight(512x784), fc2.weight(256x512), fc3.weight(10x256)
    teacher_weights = [
        teacher.fc1.weight.detach(),
        teacher.fc2.weight.detach(),
    ]
    # Student generated weights (solo FractalLinear layers)
    student_weights = [layer._generate_weight() for layer in student.layers]

    w_loss = 0.0
    for tw, sw in zip(teacher_weights, student_weights):
        w_loss += F.mse_loss(sw, tw.detach().clone())
    w_loss = w_loss / len(teacher_weights)

    # --- 2. Activation loss: MSE crudo (sin normalizar) ---
    a_loss = 0.0
    for t_act, s_act in zip(teacher_acts, student_acts):
        a_loss += F.mse_loss(s_act, t_act.detach().clone())
    a_loss = a_loss / len(teacher_acts)

    # --- 3. Output loss: soft target + hard label ---
    # Soft target (KL)
    kd_loss = F.kl_div(
        F.log_softmax(student_out / 4.0, dim=1),
        F.softmax(teacher_out / 4.0, dim=1),
        reduction='batchmean',
    ) * (4.0 ** 2)

    # Hard label (CE)
    ce_loss = F.cross_entropy(student_out, y)

    # --- combinar ---
    # Peso relativo: weight loss domina porque es la senial mas directa
    return 10.0 * w_loss + 5.0 * a_loss + 1.0 * kd_loss + 1.0 * ce_loss


def distill_v2(
    teacher: nn.Module,
    student: FractalMLP,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 20,
    lr: float = 3e-4,
    device: str = 'cpu',
):
    teacher.to(device).eval()
    student.to(device).train()

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # evaluar teacher una vez
    teacher_acc = _eval_acc(teacher, test_loader, device)
    print(f"  Teacher: {teacher_acc:.2f}%  |  "
          f"Params teacher: {sum(p.numel() for p in teacher.parameters()):,}  |  "
          f"params student: {sum(p.numel() for p in student.parameters()):,}")

    for epoch in range(epochs):
        student.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = hybrid_distillation_loss(teacher, student, x, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            with torch.inference_mode():
                out, _ = student(x, return_activations=True)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

        scheduler.step()

        student_acc = 100.0 * correct / total
        avg_loss = total_loss / total
        print(f"  Epoch {epoch+1:2d}/{epochs}  |  loss: {avg_loss:.4f}  |  "
              f"acc: {student_acc:.2f}%")

        if student_acc >= teacher_acc - 0.3 and epoch >= 5:
            print(f"  >> Student alcanzo teacher en epoca {epoch+1}")

    return {'loss': [], 'student_acc': []}


@torch.inference_mode()
def _eval_acc(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.to(device).eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total
