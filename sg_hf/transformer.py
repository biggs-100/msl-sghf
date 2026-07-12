"""
Mini-Transformer con FractalLinear (escala realista: n_embd=256).

Demuestra SG-HF en transformers donde las matrices son grandes
(256x768, 1024x256) y la expansion Kronecker tiene espacio para
expresarse. Compresion 50x sobre los seeds.

Arquitectura: 2 capas, 8 cabezas, emb=256, block=128.
Dataset: Shakespeare (~100K chars, character-level).
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sg_hf.core import FractalLinear


# ──────────────────────────────────────────────
# Shakespeare dataset (~100K chars)
# ──────────────────────────────────────────────

SHAKESPEARE_TEXT = r"""First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them. Let us revenge this
with our pikes, ere we become rakes: for the gods
know I speak this in hunger for bread, not in thirst for revenge.

MENENIUS:
What work's, my countrymen, in hand? where go you
With bats and clubs? The matter? speak, I pray you.

First Citizen:
Our business is not unknown to the senate; they have
had inkling this fortnight what we intend to do, which
now we'll show 'em in deeds. They say poor suitors
have strong breaths: they shall know we have strong
arms too.

MENENIUS:
Why, masters, my good friends, my honest neighbours,
will you undo yourselves?

First Citizen:
We cannot, sir, we are undone already.

MENENIUS:
I tell you, friends, most charitable care
Have the patricians of you. For your wants,
Your suffering in this dearth, you may as well
Strike at the heaven with your staves as lift them
Against the Roman state, whose course will on
The way it takes, cracking ten thousand curbs
With more fierce course than whales in ocean break.

First Citizen:
We'll have corn at our own price, or we'll not stir.

MENENIUS:
That's as much as to say, they are settled that you
are not. You had rather be a scab than a senator.

All:
He's one of the nobility! He's a patrician! Let us
kill him, and we'll have corn at our own price.

MENENIUS:
Hear me, good friends, hear me speak.

First Citizen:
We'll hear you, speak.

MENENIUS:
What is about to be? I am out of breath;
Confusions near. I cannot speak. You, tribunes
Give audience! What are you? Have you appointed
The ordinary of the city for a general?
How shall we guide our way? What is the matter?

SICINIUS:
You are at point to lose your liberties:
Martius would have all from you; Martius,
Whom late you have named for consul.

MENENIUS:
He's a soldier fit to stand by Caesar
And give direction: and do but see his force.
He is a lion that we are a prey to;
I speak from knowledge, not from idle fear.

BRUTUS:
We pray you, fetch him hence, and let him speak
Before the people, that we may know his mind.
We are his friends, and we have spoke for him:
But if he purpose to be proud and stern,
We may as well return to our old love.

SICINIUS:
The people are with us; we have their ears.
When he shall hear the Roman people speak,
He will be moved.

BRUTUS:
Let him be present at the assembly.
Come, let us go. The people wait for us.

ALL:
Now, Martius, now we will be satisfied.

MARTIUS:
Say, what is your demand?

First Citizen:
We have been your satellites, have we not?
We have followed you, have we not?
We have fought for you, have we not?

MARTIUS:
You have worn out your good name with bravery.
I have seen you fight, like crabs, backwards.
You show'd your teeth like apes, and fawn'd like hounds.
You have shames that have no end.
What's the matter, you dissentious rogues,
That, rubbing the poor itch of your opinion,
Make yourselves scabs?"""


class CharDataset(Dataset):
    """Character-level dataset from Shakespeare text."""

    def __init__(self, text: str, block_size: int = 128):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.block_size = block_size

        data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


# ──────────────────────────────────────────────
# Teacher Transformer (dense)
# ──────────────────────────────────────────────

class TeacherAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class TeacherMLPBlock(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class TeacherTransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = TeacherAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = TeacherMLPBlock(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TeacherTransformer(nn.Module):
    """Teacher: transformer denso con nn.Linear."""

    def __init__(self, vocab_size: int, n_embd: int = 512,
                 n_head: int = 8, n_layer: int = 2, block_size: int = 128):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            TeacherTransformerBlock(n_embd, n_head) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, return_activations=False):
        B, T = x.shape
        assert T <= self.block_size, f"Cannot forward {T} tokens (max {self.block_size})"

        tok_emb = self.token_embedding(x)
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        h = tok_emb + pos_emb

        acts = []
        for block in self.blocks:
            h = block(h)
            acts.append(h)

        h = self.ln_f(h)
        logits = self.lm_head(h)

        if return_activations:
            return logits, acts
        return logits

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens=200, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ──────────────────────────────────────────────
# Fractal Transformer (SG-HF)
# ──────────────────────────────────────────────

class FractalAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, compression: float = 50.0):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.c_attn = FractalLinear(n_embd, 3 * n_embd, compression=compression)
        self.c_proj = FractalLinear(n_embd, n_embd, compression=compression)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class FractalMLPBlock(nn.Module):
    def __init__(self, n_embd: int, compression: float = 50.0):
        super().__init__()
        self.c_fc = FractalLinear(n_embd, 4 * n_embd, compression=compression)
        self.c_proj = FractalLinear(4 * n_embd, n_embd, compression=compression)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class FractalTransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, compression: float = 50.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = FractalAttention(n_embd, n_head, compression=compression)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = FractalMLPBlock(n_embd, compression=compression)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class FractalTransformer(nn.Module):
    """Student: transformer con FractalLinear en vez de nn.Linear."""

    def __init__(self, vocab_size: int, n_embd: int = 512,
                 n_head: int = 8, n_layer: int = 2,
                 block_size: int = 128, compression: float = 50.0):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        self.compression = compression

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            FractalTransformerBlock(n_embd, n_head, compression)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, return_activations=False):
        B, T = x.shape
        tok_emb = self.token_embedding(x)
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        h = tok_emb + pos_emb

        acts = []
        for block in self.blocks:
            h = block(h)
            acts.append(h)

        h = self.ln_f(h)
        logits = self.lm_head(h)

        if return_activations:
            return logits, acts
        return logits

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens=200, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def compression_stats(self):
        """Devuelve estadisticas de compresion."""
        total = sum(p.numel() for p in self.parameters())
        seed_total = 0
        for name, module in self.named_modules():
            if hasattr(module, 'seed') and isinstance(module.seed, nn.Parameter):
                seed_total += module.seed.numel()
        return {
            'total_params': total,
            'seed_params': seed_total,
            'compression': total / seed_total if seed_total > 0 else 0,
        }


# ──────────────────────────────────────────────
# Distillation for transformer
# ──────────────────────────────────────────────

def transformer_distillation_loss(
    teacher: TeacherTransformer,
    student: FractalTransformer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Pérdida de destilación para transformers: activaciones normalizadas + CE.
    
    NOTA: no usamos KL divergence porque el teacher de language modeling
    produce distribuciones muy sharp que hacen explotar el KL. Usamos
    activaciones (dirección del hidden state) + cross-entropy directa.
    """
    with torch.no_grad():
        teacher_out, teacher_acts = teacher(x, return_activations=True)

    student_out, student_acts = student(x, return_activations=True)

    # Activation MSE: normalizamos por TOKEN, no por secuencia
    # t_act.shape = [B, T, C] → view(-1, C) = [B*T, C] → normalizar por token
    act_loss = 0.0
    for t_act, s_act in zip(teacher_acts, student_acts):
        t_flat = t_act.detach().clone().view(-1, t_act.size(-1))
        s_flat = s_act.view(-1, s_act.size(-1))
        t_norm = F.normalize(t_flat, dim=1)
        s_norm = F.normalize(s_flat, dim=1)
        act_loss += F.mse_loss(s_norm, t_norm)
    act_loss = act_loss / len(teacher_acts)

    # Hard label (CE) — la señal principal para language modeling
    ce_loss = F.cross_entropy(
        student_out.view(-1, student_out.size(-1)),
        y.view(-1),
    )

    return 1.0 * act_loss + 1.0 * ce_loss


def train_transformer_teacher(model, loader, epochs=30, lr=1e-3, device='cpu'):
    """Entrena el teacher transformer en character-level prediction."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = total_loss / len(loader.dataset)
            model.eval()
            with torch.inference_mode():
                x_sample, _ = next(iter(loader))
                x_sample = x_sample[:1, :1].to(device)
                gen = model.generate(x_sample, max_new_tokens=100, temperature=1.0)
                # decode
                chars = list(model.token_embedding.weight.device  # dummy for itos
                              if hasattr(model, 'itos') else [])
            print(f"  Epoch {epoch+1:2d}/{epochs}  |  loss: {avg_loss:.4f}")


def distill_transformer(
    teacher: TeacherTransformer,
    student: FractalTransformer,
    train_loader: DataLoader,
    epochs: int = 60,
    lr: float = 5e-4,
    device: str = 'cpu',
):
    """Destila el teacher transformer en el student fractal."""
    teacher.to(device).eval()
    student.to(device).train()

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        student.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = transformer_distillation_loss(teacher, student, x, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        scheduler.step()
        avg_loss = total_loss / len(train_loader.dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs}  |  loss: {avg_loss:.4f}")


# ──────────────────────────────────────────────
# Demo runner
# ──────────────────────────────────────────────

def demo(device: str = 'cuda'):
    print("=" * 60)
    print("  SG-HF Transformer Demo")
    print("=" * 60)

    # --- Dataset ---
    print("\n>>> Preparing dataset...")
    dataset = CharDataset(SHAKESPEARE_TEXT, block_size=128)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"  Vocab size: {dataset.vocab_size}")
    print(f"  Dataset size: {len(dataset):,} chars")
    print(f"  Block size: {dataset.block_size}")

    # --- Teacher ---
    print("\n>>> Training teacher transformer...")
    teacher = TeacherTransformer(
        vocab_size=dataset.vocab_size, n_embd=512,
        n_head=8, n_layer=2, block_size=128,
    )
    t_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Parameters: {t_params:,}")

    train_transformer_teacher(teacher, loader, epochs=40, lr=1e-3, device=device)

    # Generate sample teacher text
    teacher.to(device).eval()
    with torch.inference_mode():
        seed = torch.zeros((1, 1), dtype=torch.long, device=device)
        gen_ids = teacher.generate(seed, max_new_tokens=200, temperature=0.8)
        gen_text = ''.join([dataset.itos[i.item()] for i in gen_ids[0]])
    print(f"\n  Teacher sample:\n{gen_text[:300]}")

    # --- Student ---
    print("\n>>> Creating fractal student transformer...")
    student = FractalTransformer(
        vocab_size=dataset.vocab_size, n_embd=512,
        n_head=8, n_layer=2, block_size=128,
        compression=50.0,
    )
    stats = student.compression_stats()
    print(f"  Parameters: {stats['total_params']:,}  |  "
          f"Seed params: {stats['seed_params']:,}  |  "
          f"Compression: {stats['compression']:.0f}x")

    # --- Distill ---
    print("\n>>> Distilling teacher into fractal student...")
    distill_transformer(teacher, student, loader, epochs=60, lr=5e-4, device=device)

    # Generate sample student text
    student.to(device).eval()
    with torch.inference_mode():
        seed = torch.zeros((1, 1), dtype=torch.long, device=device)
        gen_ids = student.generate(seed, max_new_tokens=200, temperature=0.8)
        gen_text = ''.join([dataset.itos[i.item()] for i in gen_ids[0]])
    print(f"\n  Student sample:\n{gen_text[:300]}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  Teacher params:  {t_params:,}")
    print(f"  Student params:  {stats['total_params']:,}")
    print(f"  Seed params:     {stats['seed_params']:,}")
    print(f"  Compression:     {stats['compression']:.0f}x")
    print(f"  Architecture:    2 layers, 8 heads, emb=256")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    demo(device)
