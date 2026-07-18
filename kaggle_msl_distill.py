"""
Kaggle Notebook: MSL Distillation - TinyLlama-1.1B -> MSL Student
===============================================================

INSTRUCCIONES:
1. Crear notebook en kaggle.com
2. Settings -> Accelerator -> GPU P100
3. Settings -> Internet -> ON
4. Copiar TODO este archivo en la primera celda
5. Run All

Tiempo estimado: ~4-6 horas
"""

# %% (1) Instalar dependencias (si hiciera falta)
# Kaggle ya tiene transformers y torch. No instalar nada.

# %% (2) Imports
import os, sys, time, math, gc, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

DEVICE = "cuda"
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu}")
    print(f"VRAM: {vram:.1f} GB")
else:
    print("WARNING: CPU mode (very slow)")

# %% (3) MSL Layer
class MSLinear(nn.Module):
    def __init__(self, in_features, out_features, scale_ranks=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if scale_ranks is None:
            scale_ranks = [8, 16, 32]
        self.scale_ranks = list(scale_ranks)
        self.n_scales = len(scale_ranks)
        self.total_rank = sum(scale_ranks)
        self.cum_ranks = [0]
        for r in self.scale_ranks:
            self.cum_ranks.append(self.cum_ranks[-1] + r)

        self.U = nn.Parameter(torch.randn(out_features, self.total_rank) * 0.02)
        self.s = nn.Parameter(torch.randn(self.total_rank) * 0.05)
        self.V = nn.Parameter(torch.randn(self.total_rank, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

        with torch.no_grad():
            for i in range(self.total_rank):
                self.s.data[i] = math.exp(-i * 0.2) * 2.0

        self._active_scales = self.n_scales
        self.full_params = in_features * out_features
        self.compressed_params = (out_features * self.total_rank + self.total_rank +
                                  self.total_rank * in_features + out_features)

    def set_active_scales(self, k):
        self._active_scales = max(1, min(k, self.n_scales))

    def get_active_rank(self):
        return self.cum_ranks[self._active_scales]

    def forward(self, x):
        k = self._active_scales
        r = self.cum_ranks[k]
        xv = x @ self.V[:r, :].T
        xv = xv * self.s[:r]
        y = xv @ self.U[:, :r].T
        return y + self.bias

class MSLFFN(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate = MSLinear(hidden, intermediate, [16, 32, 64])
        self.up = MSLinear(hidden, intermediate, [16, 32, 64])
        self.down = MSLinear(intermediate, hidden, [32, 64, 128])

    def set_active_scales(self, k):
        self.gate.set_active_scales(k)
        self.up.set_active_scales(k)
        self.down.set_active_scales(k)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class CausalAttention(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        assert hidden % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, hidden, n_heads, intermediate):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = CausalAttention(hidden, n_heads)
        self.ln2 = nn.LayerNorm(hidden)
        self.ffn = MSLFFN(hidden, intermediate)

    def set_active_scales(self, k):
        self.ffn.set_active_scales(k)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class MSLGPT(nn.Module):
    def __init__(self, vocab_size, hidden=1024, n_heads=16, n_layers=22, context=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.context = context
        intermediate = 4 * hidden
        self.token_embed = nn.Embedding(vocab_size, hidden)
        self.pos_embed = nn.Parameter(torch.randn(1, context, hidden) * 0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden, n_heads, intermediate) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.token_embed.weight = self.head.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def set_active_scales(self, k):
        for b in self.blocks:
            b.set_active_scales(k)

    def get_msl_layers(self):
        layers = []
        for b in self.blocks:
            layers.extend([b.ffn.gate, b.ffn.up, b.ffn.down])
        return layers

    @property
    def n_scales(self):
        return self.get_msl_layers()[0].n_scales

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embed(idx) + self.pos_embed[:, :T, :]
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss

# %% (4) Load Teacher
print("Loading TinyLlama-1.1B teacher...")
teacher = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer.pad_token = tokenizer.eos_token
VOCAB = tokenizer.vocab_size
print(f"Teacher: {sum(p.numel() for p in teacher.parameters()):,} params")
print(f"Vocab: {VOCAB}")

# %% (5) Create Student MSL
print("Creating MSL student...")
student = MSLGPT(VOCAB, hidden=1024, n_heads=16, n_layers=22, context=256).to(DEVICE)

# Count params
total_p = sum(p.numel() for p in student.parameters())
msl_p = sum(p.numel() for m in student.get_msl_layers() for p in m.parameters())
equiv = 3 * 4096 * 1024 * 22  # dense FFN params for this size
print(f"Student: {total_p:,} total, {msl_p:,} MSL FFN")
print(f"Dense equiv FFN: {equiv:,} ({equiv/msl_p:.1f}x)")
print(f"Total equiv: ~{total_p - msl_p + equiv:,} (~1B)")

# %% (6) Generate Distillation Data
print("Generating distillation data...")
import random as _random
_random.seed(42)
names = "Alice Bob Charlie Diana Eva Frank Luna Max Bella Oscar".split()
animals = "dog cat bird fish rabbit fox bear frog owl".split()
actions = "found a magical stone|discovered a cave|learned to fly|saved the day".split("|")
places = "forest garden mountain river village castle island".split()

texts = []
for i in range(5000):
    n = _random.choice(names)
    a = _random.choice(animals)
    act = _random.choice(actions)
    pl = _random.choice(places)
    t = (f"Once upon a time, {n} was a little {a} who lived in a {pl}. "
         f"One day, {n} {act}. It was an amazing adventure! The end.")
    texts.append(t)

enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
input_ids = enc["input_ids"]
attn_mask = enc["attention_mask"]

# Generate teacher logits
print("Running teacher forward...")
teacher.eval()
all_logits = []
t0 = time.time()
with torch.no_grad():
    for i in range(0, 5000, 8):
        batch = input_ids[i:i+8].to(DEVICE)
        mask = attn_mask[i:i+8].to(DEVICE)
        out = teacher(batch, attention_mask=mask)
        all_logits.append(out.logits.cpu())
        if i % 200 == 0:
            print(f"  {i}/5000 ({time.time()-t0:.0f}s)")
logits = torch.cat(all_logits, dim=0)
print(f"Done: {logits.shape}, {time.time()-t0:.0f}s")

# %% (7) Distillation Training
print("Starting distillation...")
opt = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)
student.set_active_scales(student.n_scales)

t0 = time.time()
for step in range(1, 2001):
    idx = torch.randint(0, 5000, (4,))
    inp = input_ids[idx].to(DEVICE)
    tgt = logits[idx].to(DEVICE)

    out, ce_loss = student(inp[:, :-1], inp[:, 1:])
    T = 4.0
    s_soft = F.log_softmax(out[:, -1:, :] / T, dim=-1)
    t_soft = F.softmax(tgt[:, -1:, :] / T, dim=-1)
    kd_loss = F.kl_div(s_soft, t_soft, reduction='batchmean') * T * T
    loss = kd_loss + 0.5 * ce_loss

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    opt.step()
    sched.step()

    if step in [1, 500, 1000, 1500, 2000]:
        print(f"Step {step}/2000 | loss={loss.item():.4f} | kd={kd_loss.item():.4f} | ce={ce_loss.item():.4f}")

print(f"Distillation: {time.time()-t0:.0f}s")

# %% (8) Validate Truncation
print("\n" + "="*60)
print("TRUNCATION VALIDATION (no fine-tuning)")
print("="*60)

msl = student.get_msl_layers()
nf = msl[0].n_scales
full = sum(m.full_params for m in msl)

for k in range(1, nf + 1):
    student.set_active_scales(k)
    r = msl[0].get_active_rank()
    comp = 0
    for m in msl:
        ar = m.get_active_rank()
        if ar > 0:
            comp += m.compressed_params * (m.total_rank / ar)
    comp = full / max(comp, 1)
    losses = []
    with torch.no_grad():
        for _ in range(10):
            idx = torch.randint(0, 5000, (8,))
            inp = input_ids[idx].to(DEVICE)
            _, loss = student(inp[:, :-1], inp[:, 1:])
            losses.append(loss.item())
    avg = sum(losses) / len(losses)
    print(f"k={k} | rank={r:3d} | comp={comp:.1f}x | loss={avg:.4f} | ppl={math.exp(avg):.2f}")

# %% (9) Kronecker on U factors
print("\n" + "="*60)
print("KRONECKER ON MSL FACTORS")
print("="*60)
layer = msl[0]
r = 64
U_trunc = layer.U[:, :r]
norms = U_trunc.norm(dim=1, keepdim=True)
U_norm = U_trunc / (norms + 1e-8)

p, q = 32, 8
a, b = U_norm.shape[0] // p, r // q
seed = torch.zeros(p, q, device=DEVICE)
for i in range(p):
    for j in range(q):
        seed[i,j] = U_norm[i*a:(i+1)*a, j*b:(j+1)*b].mean()
row_b = torch.zeros(p, a, device=DEVICE)
col_b = torch.zeros(q, b, device=DEVICE)
for i in range(p):
    Ubi, sbi, _ = torch.linalg.svd(U_norm[i*a:(i+1)*a, :b].float(), full_matrices=False)
    row_b[i] = Ubi[:, 0] * sbi[0].sqrt()
for j in range(q):
    _, sbj, Vbj = torch.linalg.svd(U_norm[:a, j*b:(j+1)*b].float(), full_matrices=False)
    col_b[j] = Vbj[0] * sbj[0].sqrt()
U_k = torch.einsum('ij,ia,jb->iajb', seed, row_b, col_b).reshape(p*a, q*b)
cos_k = (U_norm.flatten() * U_k.flatten()).sum() / (U_norm.norm() * U_k.norm())
print(f"U_norm [{list(U_norm.shape)}] std={U_norm.std():.4f}")
print(f"Kronecker COS: {cos_k:.4f}")
kron_params = seed.numel() + row_b.numel() + col_b.numel()
orig_params = U_trunc.numel()
print(f"Kron params: {kron_params:,} vs original: {orig_params:,} ({orig_params/kron_params:.0f}x)")
print(f"U_norm rows: {U_norm.shape[0]}")

# %% (10) Generate text sample
print("\n" + "="*60)
print("TEXT GENERATION")
print("="*60)
prompt = "Once upon a time"
ctx = torch.tensor([[tokenizer.bos_token_id] + tokenizer.encode(prompt)],
                   dtype=torch.long).to(DEVICE)
ctx = ctx[:, :64]

# Truncate ctx if too long
if ctx.shape[1] > 64:
    ctx = ctx[:, :64]

for label, k in [("ALUMNO(k=1)", 1), ("PROFESOR", student.n_scales)]:
    student.set_active_scales(k)
    student.eval()
    gen = ctx.clone()
    with torch.no_grad():
        for _ in range(64):
            logits, _ = student(gen[:, -student.context:])
            probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
            nxt = torch.multinomial(probs, 1)
            gen = torch.cat([gen, nxt], dim=1)
    out = tokenizer.decode(gen[0])
    print(f"\n[{label}]: {out}")

print("\nDone.")
