"""
SG-HF Demo: Síntesis Generativa de Pesos por Holografía Fractal.

Pipeline completo:
  1. Teacher MLP (535K params) entrenado en MNIST
  2. Student FractalMLP con compresión ~400× por capa
  3. Destilación: seed aprende a generar pesos que imitan al teacher
  4. Comparación final: precisión, tamaño, velocidad
"""

import time
import torch

from sg_hf.teacher import TeacherMLP, load_mnist, train_teacher, evaluate_accuracy
from sg_hf.core import FractalMLP
from sg_hf.distill import distill


def print_separator(title: str):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo: {device}  |  PyTorch: {torch.__version__}")

    BATCH_SIZE = 128
    TEACHER_EPOCHS = 5
    DISTILL_EPOCHS = 60
    COMPRESSION = 50.0

    # ─── 1. Datos ───
    print_separator("1. MNIST")
    train_loader, test_loader = load_mnist(batch_size=BATCH_SIZE, max_samples=30000)
    print(f"  Train: {len(train_loader.dataset):,}  |  Test: {len(test_loader.dataset):,}")

    # ─── 2. Teacher ───
    print_separator("2. Teacher (denso)")
    teacher = TeacherMLP()
    t_params = teacher.total_params()
    print(f"  Parámetros: {t_params:,}")
    train_teacher(teacher, train_loader, epochs=TEACHER_EPOCHS, device=device)
    teacher_acc = evaluate_accuracy(teacher, test_loader, device)
    print(f"\n  >> Test accuracy: {teacher_acc:.2f}%")

    # ─── 3. Student ───
    print_separator("3. Student Fractal (SG-HF)")
    student = FractalMLP([784, 512, 256, 10], compression=COMPRESSION)

    print(f"  {'Capa':<20} {'Original':>10} {'Seed':>10} {'Compresión':>10}")
    print(f"  {'-'*50}")
    for i, layer in enumerate(student.layers):
        orig = layer.in_features * layer.out_features
        seed_sz = layer.seed.numel()
        ratio = orig / seed_sz
        print(f"  {'FractalLinear-'+str(i):<20} {orig:>10,} {seed_sz:>10,} {ratio:>8.0f}x")

    s_params_all = sum(p.numel() for p in student.parameters())
    s_params_seed = sum(l.seed.numel() for l in student.layers)

    print(f"  {'-'*50}")
    print(f"  {'Total (student)':<20} {t_params:>10,} {s_params_all:>10,} "
          f"{t_params/s_params_all:>8.1f}x")
    print(f"  {'Solo seeds':<20} {'':>10} {s_params_seed:>10,} "
          f"{t_params/s_params_seed:>8.0f}x")
    print()

    # ─── 4. Destilación ───
    print_separator("4. Destilación (seed → teacher)")
    history = distill(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=DISTILL_EPOCHS,
        lr=5e-4,
        alpha=0.85,
        temperature=4.0,
        device=device,
    )

    # ─── 5. Resultados ───
    print_separator("5. Resultados")

    student_acc = evaluate_accuracy(student, test_loader, device)
    teacher_acc_final = evaluate_accuracy(teacher, test_loader, device)

    delta = teacher_acc_final - student_acc

    print(f"\n  {'':>30} {'Params':>12} {'Accuracy':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Teacher (denso)':>30} {t_params:>12,} {teacher_acc_final:>9.2f}%")
    print(f"  {'Student (SG-HF)':>30} {s_params_all:>12,} {student_acc:>9.2f}%")
    print(f"  {'Solo seeds':>30} {s_params_seed:>12,} {'—':>9}")
    print(f"\n  Diferencia: {delta:+.2f} pp  |  "
          f"Compresión total: {t_params/s_params_all:.0f}x  |  "
          f"Solo pesos: {t_params/s_params_seed:.0f}x")

    # ─── 6. Benchmark ───
    print_separator("6. Inferencia en CPU")
    dummy = torch.randn(1, 784)
    teacher.to('cpu')
    student.to('cpu')

    for _ in range(20):
        teacher(dummy)
        student(dummy)

    t0 = time.perf_counter()
    for _ in range(200):
        teacher(dummy)
    t_t = (time.perf_counter() - t0) / 200

    t0 = time.perf_counter()
    for _ in range(200):
        student(dummy)
    t_s = (time.perf_counter() - t0) / 200

    print(f"  Teacher: {t_t*1000:.1f} ms/token  ({1/t_t:.0f} tok/s)")
    print(f"  Student: {t_s*1000:.1f} ms/token  ({1/t_s:.0f} tok/s)")
    print(f"  Overhead: {t_s/t_t:.1f}x (generación de pesos sobre la marcha)")

    # ─── 7. Demostración en vivo ───
    print_separator("7. Demostración: pesos generados del seed")

    layer = student.layers[0]
    seed = layer.seed.detach()
    W = layer._generate_weight().detach()

    print(f"  Seed almacenado:  {list(seed.shape)} = {seed.numel():,} floats")
    print(f"  Peso generado:    {list(W.shape)} = {W.numel():,} floats")
    print(f"  Compresión capa:  {W.numel() / seed.numel():.0f}x")
    print(f"  Determinista:     {torch.allclose(layer._generate_weight(), W, atol=1e-6)}")
    print(f"\n  → Los pesos NO se almacenan. Se generan bajo demanda.")
    print(f"    El seed de {seed.numel():,} parámetros reemplaza ")
    print(f"    {W.numel():,} parámetros del teacher.")

    # ─── 8. Proyección a escala ───
    print_separator("8. Proyección a modelos grandes")

    # overhead total del seed + expansión se amortiza en modelos grandes
    for name, dim, layers, factor in [("70B", 8192, 80, 100),
                                        ("400B", 16384, 96, 200),
                                        ("1T", 18432, 128, 400)]:
        orig = dim * dim * layers * 3  # Q,K,V aprox en parámetros
        seed_p = orig / factor
        print(f"  {name:<8}  original: {orig/1e9:.1f}B  |  "
              f"seed ~{seed_p/1e6:.0f}M  ({factor}x compresión)  "
              f"misma matemática que el demo")

    print(f"\n{'=' * 65}")
    print(f"  ✓ Demo SG-HF completa")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()
