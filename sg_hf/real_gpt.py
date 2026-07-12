"""
SG-HF sobre distilgpt2 real.

Pipeline:
  1. Carga distilgpt2 pre-entrenado (teacher, 82M params)
  2. Crea FractalGPT2 (student) con FractalLinear en vez de Conv1D
  3. Inicializa seeds para que generen W ~ teacher (por MSE)
  4. Destilacion: seeds recuperan precision
  5. Compara compresion y texto generado
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sg_hf.core import FractalLinear

device = 'cuda' if torch.cuda.is_available() else 'cpu'
COMPRESSION = 50.0


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ──────────────────────────────────────────────
# FractalGPT2: replica distilgpt2 con FractalLinear
# ──────────────────────────────────────────────

class FractalAttention(nn.Module):
    def __init__(self, n_embd=768, n_head=12):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        # Conv1D(2304, 768) en teacher
        self.c_attn = FractalLinear(n_embd, 3 * n_embd, compression=COMPRESSION)
        self.c_proj = FractalLinear(n_embd, n_embd, compression=COMPRESSION)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class FractalMLP(nn.Module):
    def __init__(self, n_embd=768):
        super().__init__()
        self.c_fc = FractalLinear(n_embd, 4 * n_embd, compression=COMPRESSION)
        self.c_proj = FractalLinear(4 * n_embd, n_embd, compression=COMPRESSION)
        self.act = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class FractalGPT2Block(nn.Module):
    def __init__(self, n_embd=768, n_head=12):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = FractalAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = FractalMLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class FractalGPT2(nn.Module):
    def __init__(self, n_embd=768, n_head=12, n_layer=6, vocab_size=50257, max_pos=1024):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(max_pos, n_embd)
        self.blocks = nn.ModuleList([
            FractalGPT2Block(n_embd, n_head) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.token_embedding(x) + self.pos_embedding(pos)
        for block in self.blocks:
            h = block(h)
        return self.ln_f(h)


# ──────────────────────────────────────────────
# Cargar teacher, crear student, inicializar seeds
# ──────────────────────────────────────────────

def load_teacher():
    print(">>> Cargando distilgpt2...")
    model = AutoModelForCausalLM.from_pretrained('distilgpt2').to(device)
    model.eval()
    print(f"  Teacher params: {count_params(model):,}")
    return model


def create_fractal_student():
    """Crea FractalGPT2 que imita la arquitectura de distilgpt2."""
    print("\n>>> Creando FractalGPT2...")
    student = FractalGPT2(n_embd=768, n_head=12, n_layer=6, vocab_size=50257, max_pos=1024)
    student.to(device)
    # lm_head no incluido en FractalGPT2 (se maneja aparte)
    lm_head = nn.Linear(768, 50257, bias=False).to(device)

    total = count_params(student) + count_params(lm_head)
    seed_only = sum(m.seed.numel() for m in student.modules() if hasattr(m, 'seed'))
    print(f"  Student total: {total:,}")
    print(f"  Seed params: {seed_only:,}")
    print(f"  Compresion seed: ~{count_params(teacher_model):,} / {seed_only:,} = "
          f"{count_params(teacher_model) / seed_only:.0f}x")

    return student, lm_head


def init_seeds_from_teacher(student, teacher, lm_head):
    """
    Inicializa los seeds de cada FractalLinear para que generen
    W ≈ teacher_weights via optimizacion rapida (100 pasos por capa).
    """
    print("\n>>> Inicializando seeds desde teacher...")
    # Mapeo: capas del teacher a capas del student
    # Teacher: transformer.h.0.attn.c_attn → Student: blocks[0].attn.c_attn
    teacher_blocks = teacher.transformer.h
    teacher_lm = teacher.lm_head

    # Inicializar lm_head
    with torch.no_grad():
        lm_head.weight.copy_(teacher_lm.weight)

    # Para cada bloque, inicializar attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj
    pairings = []
    for i in range(6):
        tb = teacher_blocks[i]
        sb = student.blocks[i]
        pairings.extend([
            (tb.attn.c_attn.weight.T, sb.attn.c_attn),    # Conv1D weight is (in, out) → transpose to (out, in)
            (tb.attn.c_proj.weight.T, sb.attn.c_proj),
            (tb.mlp.c_fc.weight.T, sb.mlp.c_fc),
            (tb.mlp.c_proj.weight.T, sb.mlp.c_proj),
        ])
        # biases
        sb.attn.c_attn.bias.data.copy_(tb.attn.c_attn.bias)
        sb.attn.c_proj.bias.data.copy_(tb.attn.c_proj.bias)
        sb.mlp.c_fc.bias.data.copy_(tb.mlp.c_fc.bias)
        sb.mlp.c_proj.bias.data.copy_(tb.mlp.c_proj.bias)

    # Inicializar Embeddings
    with torch.no_grad():
        student.token_embedding.weight.copy_(teacher.transformer.wte.weight)
        student.pos_embedding.weight.copy_(teacher.transformer.wpe.weight)

    # Optimizar seeds para minimizar MSE(seed_generates_W, teacher_W)
    for target_w, fractal_layer in pairings:
        target_w = target_w.detach().to(device)  # (out, in)
        # Optimizar seed de fractal_layer
        params = [fractal_layer.seed, fractal_layer.freq_scale, fractal_layer.freq_shift,
                  fractal_layer.row_basis, fractal_layer.col_basis]
        opt = torch.optim.Adam(params, lr=1e-2)
        for step in range(200):
            opt.zero_grad()
            W_gen = fractal_layer._generate_weight()
            loss = F.mse_loss(W_gen, target_w)
            loss.backward()
            opt.step()
        if hasattr(fractal_layer, 'seed'):
            print(f"  Capa init MSE: {loss.item():.6f}  "
                  f"(seed {list(fractal_layer.seed.shape)})")

    print("  ✓ Inicializacion completa")


def distill_step(student, teacher, lm_head, texts, tokenizer):
    """Un paso de destilacion."""
    student.train()
    lm_head.train()
    opt = torch.optim.AdamW(
        list(student.parameters()) + list(lm_head.parameters()),
        lr=5e-5,
    )

    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                       max_length=128).to(device)
    x = inputs['input_ids']
    y = x[:, 1:].contiguous()

    with torch.no_grad():
        teacher_out = teacher.transformer(x, output_hidden_states=True)
        teacher_hidden = teacher_out.last_hidden_state
        teacher_logits = teacher.lm_head(teacher_hidden)

    student_hidden = student(x)
    student_logits = lm_head(student_hidden)

    # Activation loss (hidden states)
    act_loss = F.mse_loss(
        F.normalize(student_hidden.view(-1, 768), dim=1),
        F.normalize(teacher_hidden.view(-1, 768), dim=1),
    )

    # CE loss
    ce_loss = F.cross_entropy(
        student_logits[:, :-1].contiguous().view(-1, 50257),
        y.view(-1),
    )

    loss = act_loss + ce_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.item(), ce_loss.item()


def main():
    global teacher_model
    teacher = load_teacher()
    teacher_model = teacher  # for seed counting in create_fractal_student

    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token

    student, lm_head = create_fractal_student()

    # init seeds
    init_seeds_from_teacher(student, teacher, lm_head)

    # test generation BEFORE distillation
    student.eval()
    lm_head.eval()
    prompt = "The Roman citizens were angry because"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    print("\n>>> Generacion ANTES de destilacion:")
    with torch.inference_mode():
        h = student(inputs['input_ids'])
        logits = lm_head(h)
        next_token = logits[0, -1].argmax().item()
        text = tokenizer.decode(next_token)
    print(f"  Student raw output (first token only): '{tokenizer.decode(inputs['input_ids'][0].tolist() + [next_token])}'")

    # distill (5 steps)
    print("\n>>> Destilacion rapida (5 pasos)...")
    texts = [
        "The Roman empire fell because of",
        "In the beginning, there was",
        "The citizens gathered to hear",
        "The general spoke of victory",
        "Peace and prosperity were",
    ]
    for step in range(5):
        loss, ce = distill_step(student, teacher, lm_head, texts, tokenizer)
        print(f"  Step {step+1}: loss={loss:.4f}  ce={ce:.4f}")

    # test generation AFTER
    student.eval()
    lm_head.eval()
    print("\n>>> Generacion DESPUES de destilacion:")
    with torch.inference_mode():
        h = student(inputs['input_ids'])
        logits = lm_head(h)
        next_token = logits[0, -1].argmax().item()
        text = tokenizer.decode(next_token)
    print(f"  Student: '{tokenizer.decode(inputs['input_ids'][0].tolist() + [next_token])}'")

    # Teacher generation for comparison
    print("\n>>> Teacher (referencia):")
    teacher.eval()
    with torch.inference_mode():
        out = teacher.generate(
            **inputs, max_new_tokens=20, temperature=0.8, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(f"  Teacher: '{tokenizer.decode(out[0])}'")


if __name__ == '__main__':
    main()
