"""
Visualización: pesos generados por el seed fractal vs teacher.
"""

import os
import torch
import matplotlib
matplotlib.use('Agg')  # headless, save to file
import matplotlib.pyplot as plt
import numpy as np

from sg_hf.teacher import TeacherMLP, load_mnist, train_teacher, evaluate_accuracy
from sg_hf.core import FractalMLP
from sg_hf.distill import distill

OUT_DIR = 'output'


def ensure_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def plot_weight_comparison(teacher_weight, student_weight, title, save_path):
    """Compara peso del teacher vs peso generado por el seed."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # clamp al percentil 99 para evitar outliers visuales
    vmax = min(
        teacher_weight.abs().quantile(0.99).item(),
        student_weight.abs().quantile(0.99).item(),
    )
    vmin = -vmax

    # Teacher
    im0 = axes[0].imshow(teacher_weight.cpu().numpy(), cmap='RdBu',
                          vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title(f'Teacher ({teacher_weight.shape[0]}x{teacher_weight.shape[1]})')
    axes[0].set_xlabel('Input features')
    axes[0].set_ylabel('Output features')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Student (generated)
    im1 = axes[1].imshow(student_weight.cpu().numpy(), cmap='RdBu',
                          vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title(f'SG-HF (seed={student_weight.numel()//400:,} param)')
    axes[1].set_xlabel('Input features')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Error
    error = (teacher_weight - student_weight).abs()
    im2 = axes[2].imshow(error.cpu().numpy(), cmap='hot',
                          aspect='auto')
    axes[2].set_title(f'|Teacher - SG-HF| (mean={error.mean():.4f})')
    axes[2].set_xlabel('Input features')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_training_curve(history, save_path):
    """Loss curve + accuracy over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    epochs = range(1, len(history['loss']) + 1)

    ax1.plot(epochs, history['loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Distillation Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['student_acc'], 'g-', linewidth=2, label='Student')
    teacher_acc = history.get('teacher_acc', [None])
    if teacher_acc and teacher_acc[0]:
        ax2.axhline(y=teacher_acc[0], color='gray', linestyle='--',
                    label=f"Teacher ({teacher_acc[0]:.1f}%)")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_compression_frontier(results, save_path):
    """Tradeoff curve: compression factor vs accuracy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    factors = [r['compression'] for r in results]
    student_accs = [r['student_acc'] for r in results]
    teacher_acc = results[0]['teacher_acc']

    ax.semilogx(factors, student_accs, 'bo-', linewidth=2, markersize=8,
                label='SG-HF Student')
    ax.axhline(y=teacher_acc, color='gray', linestyle='--', linewidth=1.5,
               label=f"Teacher ({teacher_acc:.1f}%)")

    # Anotar cada punto
    for r in results:
        ax.annotate(
            f"{r['compression']:.0f}x\n({r['student_acc']:.1f}%)",
            (r['compression'], r['student_acc']),
            xytext=(5, 12), textcoords='offset points', fontsize=9,
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
        )

    ax.set_xlabel('Compression factor (seed vs original)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('SG-HF: Compression vs Accuracy Tradeoff')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(10, 800)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def generate_all(device: str = 'cuda'):
    """Genera todas las visualizaciones."""
    ensure_dir()
    train_loader, test_loader = load_mnist(batch_size=128, max_samples=30000)

    # --- Teacher ---
    print(">>> Training teacher...")
    teacher = TeacherMLP()
    train_teacher(teacher, train_loader, epochs=5, device=device)
    teacher_acc = evaluate_accuracy(teacher, test_loader, device)
    print(f"  Teacher: {teacher_acc:.2f}%")

    # --- Multiple compressions ---
    results = []
    factors = [400, 200, 100, 50, 25]

    for factor in factors:
        print(f"\n>>> Training SG-HF at {factor}x...")
        student = FractalMLP([784, 512, 256, 10], compression=float(factor))
        history = distill(
            teacher, student, train_loader, test_loader,
            epochs=60 if factor >= 50 else 30,
            lr=5e-4, alpha=0.85, temperature=4.0, device=device,
        )
        student_acc = evaluate_accuracy(student, test_loader, device)
        results.append({
            'compression': factor,
            'student_acc': student_acc,
            'teacher_acc': teacher_acc,
        })

        # Guardar modelo
        torch.save(student.state_dict(), f'{OUT_DIR}/student_{factor}x.pt')

        # Curva de entrenamiento
        plot_training_curve(history, f'{OUT_DIR}/curve_{factor}x.png')

    # --- Tradeoff curve ---
    plot_compression_frontier(results, f'{OUT_DIR}/compression_frontier.png')

    # --- Weight comparison (best model: 50x) ---
    print(f"\n>>> Weight visualization (50x)...")
    student_best = FractalMLP([784, 512, 256, 10], compression=50.0)
    student_best.load_state_dict(torch.load(f'{OUT_DIR}/student_50x.pt',
                                             weights_only=False))
    student_best.to(device)

    teacher.to(device)
    for i, layer in enumerate(student_best.layers):
        teacher_w = teacher.state_dict()[f'fc{i+1}.weight'].detach()
        student_w = layer._generate_weight().detach()
        plot_weight_comparison(
            teacher_w, student_w,
            f'SG-HF Layer {i+1}: Teacher vs Generated (50x compression)',
            f'{OUT_DIR}/weights_layer{i+1}_50x.png',
        )

    print(f"\n>>> All visualizations saved to {OUT_DIR}/")
    return results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generate_all(device)
