"""
Destilación por Contraste de Activaciones.

El seed fractal aprende no copiando los pesos del teacher,
sino igualando las activaciones capa por capa.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def activation_distillation_loss(
    teacher: nn.Module,
    student: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    alpha: float = 0.9,
    temperature: float = 4.0,
) -> torch.Tensor:
    """
    Pérdida de destilación por activaciones.

    Calcula MSE entre las activaciones ocultas del teacher y del student,
    más la pérdida cross-entropy con soft target si se proporciona y.

    Args:
        teacher: modelo frozen que queremos imitar
        student: modelo con FractalLinear que estamos entrenando
        x: batch de entrada
        y: targets (opcional, para pérdida CE auxiliar)
        alpha: peso de la pérdida de activaciones (1-alpha para CE)
        temperature: temperatura para suavizar soft targets
    """
    # Forward del teacher (frozen)
    with torch.inference_mode():
        teacher_out, teacher_acts = teacher(x, return_activations=True)

    # Forward del student
    student_out, student_acts = student(x, return_activations=True)

    # --- Pérdida de activaciones (MSE capa por capa) ---
    act_loss = 0.0
    for t_act, s_act in zip(teacher_acts, student_acts):
        # normalizar cada activación para que las escalas no dominen
        t_norm = F.normalize(t_act.view(t_act.size(0), -1), dim=1)
        s_norm = F.normalize(s_act.view(s_act.size(0), -1), dim=1)
        act_loss += F.mse_loss(s_norm, t_norm)

    act_loss = act_loss / len(teacher_acts)

    # --- Pérdida de output (soft targets con temperatura) ---
    soft_teacher = F.softmax(teacher_out / temperature, dim=1)
    soft_student = F.log_softmax(student_out / temperature, dim=1)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
    kd_loss = kd_loss * (temperature ** 2)  # escala por temperatura

    # Combinar
    total = alpha * act_loss + (1.0 - alpha) * kd_loss

    # Opcional: agregar CE si tenemos labels
    if y is not None:
        ce_loss = F.cross_entropy(student_out, y)
        total = 0.5 * total + 0.5 * ce_loss

    return total


def distill(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    alpha: float = 0.9,
    temperature: float = 4.0,
    device: str = 'cpu',
):
    """
    Destila el teacher en el student (entrena solo los seeds).

    Returns:
        history: dict con loss/acc por época
    """
    teacher.to(device).eval()
    student.to(device).train()

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'loss': [], 'student_acc': [], 'teacher_acc': []}

    teacher_acc = _eval_acc(teacher, test_loader, device)
    print(f"  Teacher accuracy: {teacher_acc:.2f}% (frozen)")
    history['teacher_acc'].append(teacher_acc)

    for epoch in range(epochs):
        student.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = activation_distillation_loss(
                teacher, student, x, y,
                alpha=alpha, temperature=temperature,
            )
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
        history['loss'].append(avg_loss)
        history['student_acc'].append(student_acc)

        print(f"  Epoch {epoch + 1:2d}/{epochs}  |  loss: {avg_loss:.4f}  |  "
              f"student acc: {student_acc:.2f}%")

        # early sanity: si student ya igualó al teacher, podríamos cortar
        if student_acc >= teacher_acc - 0.5 and epoch >= 3:
            print(f"  ✓ Student alcanzó al teacher en época {epoch + 1}")
            # no cortamos, seguimos para estabilizar

    return history


@torch.inference_mode()
def _eval_acc(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.to(device).eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total
