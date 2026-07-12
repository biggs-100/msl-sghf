import torch
from torch.utils.data import DataLoader
from sg_hf.transformer import (
    CharDataset, SHAKESPEARE_TEXT,
    TeacherTransformer, FractalTransformer,
    train_transformer_teacher, distill_transformer,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = CharDataset(SHAKESPEARE_TEXT, block_size=128)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# teacher
teacher = TeacherTransformer(vocab_size=dataset.vocab_size, n_embd=512, n_head=8, n_layer=2, block_size=128).to(device)
print(f"Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
train_transformer_teacher(teacher, loader, epochs=40, lr=1e-3, device=device)

teacher.eval()
with torch.inference_mode():
    seed = torch.zeros((1, 1), dtype=torch.long, device=device)
    gen_ids = teacher.generate(seed, max_new_tokens=200, temperature=0.8)
    gen_text = "".join([dataset.itos[i.item()] for i in gen_ids[0]])
print(f"Teacher:\n{gen_text[:300]}")

# student
student = FractalTransformer(vocab_size=dataset.vocab_size, n_embd=512, n_head=8, n_layer=2, block_size=128, compression=50.0).to(device)
stats = student.compression_stats()
print(f"Student: {stats['total_params']:,} total | {stats['seed_params']:,} seeds | {stats['compression']:.0f}x")

distill_transformer(teacher, student, loader, epochs=60, lr=5e-4, device=device)

student.eval()
with torch.inference_mode():
    seed = torch.zeros((1, 1), dtype=torch.long, device=device)
    gen_ids = student.generate(seed, max_new_tokens=200, temperature=0.8)
    gen_text = "".join([dataset.itos[i.item()] for i in gen_ids[0]])
print(f"Student:\n{gen_text[:300]}")
