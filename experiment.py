"""
Experiment: curva de tradeoff compresion vs accuracy.
Prueba multiples factores y reporta resultados.
"""

import torch, time
from sg_hf.teacher import TeacherMLP, load_mnist, train_teacher, evaluate_accuracy
from sg_hf.core import FractalMLP
from sg_hf.distill import distill


def run_experiment(compression: float, distill_epochs: int = 25):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # datos
    train_loader, test_loader = load_mnist(batch_size=128, max_samples=30000)

    # teacher (entrenar una vez y reusar)
    teacher = TeacherMLP()
    teacher.load_state_dict(torch.load('data/teacher.pt', weights_only=False))

    # student
    student = FractalMLP([784, 512, 256, 10], compression=compression)

    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    seed_params = sum(l.seed.numel() for l in student.layers)
    ratio_raw = t_params / seed_params
    ratio_total = t_params / s_params

    # destilar
    t0 = time.perf_counter()
    distill(teacher, student, train_loader, test_loader,
            epochs=distill_epochs, lr=5e-4, alpha=0.85,
            temperature=4.0, device=device)
    elapsed = time.perf_counter() - t0

    # evaluar
    student_acc = evaluate_accuracy(student, test_loader, device)
    teacher_acc = evaluate_accuracy(teacher, test_loader, device)

    return {
        'compression': compression,
        'teacher_acc': teacher_acc,
        'student_acc': student_acc,
        'gap': teacher_acc - student_acc,
        'teacher_params': t_params,
        'student_params': s_params,
        'seed_params': seed_params,
        'ratio_seed': ratio_raw,
        'ratio_total': ratio_total,
        'epochs': distill_epochs,
        'elapsed_min': elapsed / 60,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # --- entrena teacher una vez ---
    print("\n>>> Entrenando teacher...")
    train_loader, test_loader = load_mnist(batch_size=128, max_samples=30000)
    teacher = TeacherMLP()
    train_teacher(teacher, train_loader, epochs=5, device=device)
    torch.save(teacher.state_dict(), 'data/teacher.pt')
    teacher_acc = evaluate_accuracy(teacher, test_loader, device)
    print(f"Teacher: {teacher_acc:.2f}%")

    # --- experimentos ---
    factors = [400, 200, 100, 50, 25]
    results = []

    for factor in factors:
        print(f"\n{'='*50}")
        print(f">>> Compression: {factor}x")
        print(f"{'='*50}")
        r = run_experiment(factor, distill_epochs=30)
        results.append(r)
        print(f"\n  Resultado: {r['student_acc']:.2f}% (gap: {r['gap']:.2f}pp, "
              f"seed: {r['ratio_seed']:.0f}x, total: {r['ratio_total']:.0f}x)")

    # --- resumen ---
    print(f"\n\n{'='*60}")
    print(f"  CURVA DE TRADEOFF: COMPRESION vs ACCURACY")
    print(f"{'='*60}")
    print(f"  {'Factor':>8} | {'Seed ratio':>10} | {'Total ratio':>10} | "
          f"{'Student':>8} | {'Teacher':>8} | {'Gap':>6}")
    print(f"  {'-'*58}")
    for r in results:
        print(f"  {r['compression']:>8.0f}x | {r['ratio_seed']:>9.0f}x | "
              f"{r['ratio_total']:>9.0f}x | {r['student_acc']:>7.2f}% | "
              f"{r['teacher_acc']:>7.2f}% | {r['gap']:>+5.2f}pp")


if __name__ == '__main__':
    main()
