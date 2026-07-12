import torch
from torch.utils.data import DataLoader
from sg_hf.transformer import (
    CharDataset, TINY_SHAKESPEARE,
    TeacherTransformer, FractalTransformer,
    train_transformer_teacher, distill_transformer,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = CharDataset(TINY_SHAKESPEARE, block_size=64)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# teacher
teacher = TeacherTransformer(vocab_size=dataset.vocab_size, n_embd=64, n_head=4, n_layer=2, block_size=64)
train_transformer_teacher(teacher, loader, epochs=40, lr=1e-3, device=device)

t_params = sum(p.numel() for p in teacher.parameters())
print(f"Teacher params: {t_params:,}")

teacher.to(device).eval()
with torch.inference_mode():
    seed = torch.zeros((1, 1), dtype=torch.long, device=device)
    gen_ids = teacher.generate(seed, max_new_tokens=100, temperature=0.8)
    gen_text = "".join([dataset.itos[i.item()] for i in gen_ids[0]])
print(f"Teacher: {gen_text[:150]}")

# student
student = FractalTransformer(vocab_size=dataset.vocab_size, n_embd=64, n_head=4, n_layer=2, block_size=64, compression=50.0)
stats = student.compression_stats()
print(f"Student: {stats['total_params']:,} total | {stats['seed_params']:,} seeds | {stats['compression']:.0f}x")

distill_transformer(teacher, student, loader, epochs=40, lr=5e-4, device=device)

student.to(device).eval()
with torch.inference_mode():
    seed = torch.zeros((1, 1), dtype=torch.long, device=device)
    gen_ids = student.generate(seed, max_new_tokens=100, temperature=0.8)
    gen_text = "".join([dataset.itos[i.item()] for i in gen_ids[0]])
print(f"Student: {gen_text[:150]}")
