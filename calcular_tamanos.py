"""
Calcula tamano original vs comprimido de Mistral-7B y proyeccion a MoE.
"""
from safetensors import safe_open

path = r'C:\Users\USER\.cache\huggingface\hub\models--mistralai--Mistral-7B-v0.3\snapshots\caa1feb0e54d415e2df31207e5f4e273e33509b1\consolidated.safetensors'

with safe_open(path, framework='pt', device='cpu') as f:
    keys = list(f.keys())
    shapes = {k: f.get_slice(k).get_shape() for k in keys}

def params_from_shape(shape):
    n = 1
    for d in shape:
        n *= d
    return n

sizes = {'gate': 0, 'up': 0, 'down': 0, 'q':0, 'k':0, 'v':0, 'o':0, 'embed':0, 'norm':0, 'output':0}
for k, shape in shapes.items():
    n = params_from_shape(shape) * 4  # FP32 bytes
    if 'w1' in k: sizes['gate'] += n
    elif 'w3' in k: sizes['up'] += n
    elif 'w2' in k: sizes['down'] += n
    elif 'wq' in k: sizes['q'] += n
    elif 'wk' in k: sizes['k'] += n
    elif 'wv' in k: sizes['v'] += n
    elif 'wo' in k: sizes['o'] += n
    elif 'norm' in k: sizes['norm'] += n
    elif 'embed' in k or 'tok' in k: sizes['embed'] += n
    elif 'output' in k: sizes['output'] += n

total = sum(sizes.values())

def gb(b): return b/1024/1024/1024
def mb(b): return b/1024/1024

print("=" * 65)
print("MISTRAL-7B — ORIGINAL vs COMPRIMIDO (hibrido SG-HF + ternario)")
print("=" * 65)
print("%10s | %14s | %12s | %18s" % ("Componente", "Original FP32", "Comprimido", "Metodo"))
print("-" * 65)

total_comp = 0
for name in ['gate','up','down','q','k','v','o','embed','output','norm']:
    if sizes[name] == 0:
        continue
    orig = sizes[name]
    if name in ('gate', 'up'):
        comp = orig / 4 / 16  # 2-bit ternary
        method = "ternario 16x"
    elif name in ('down', 'q', 'k', 'v', 'o'):
        comp = orig / 4 / 100 * 2  # SG-HF FP16
        method = "SG-HF 100x"
    else:
        comp = orig / 2
        method = "FP16 directo"
    total_comp += comp
    print("%10s | %6.2f GB | %6.0f MB | %18s" % (name, gb(orig), mb(comp), method))

print("-" * 65)
print("%10s | %6.2f GB | %6.0f MB | %d x" % ("TOTAL", gb(total), mb(total_comp), total/total_comp))
print()
print("  Original FP32:  %.2f GB" % gb(total))
print("  Original FP16:  %.2f GB" % gb(total/2))
print("  Comprimido:     %.0f MB" % mb(total_comp))
print("  Ratio total:    %dx vs FP32, %dx vs FP16" % (total/total_comp, total/total_comp/2))

# Proyeccion MoE (GLM-5.2)
print()
print("=" * 65)
print("PROYECCION a MoE tipo GLM-5.2 (8 expertos, ~52B)")
print("=" * 65)
n_layers, n_experts = 32, 8
h, i, v = 4096, 14336, 65000

expert_p = 3 * h * i
attn_p = 4 * h * h
emb_p = v * h
total_p = n_layers * (n_experts * expert_p + attn_p) + emb_p
total_b = total_p * 4

print("  Params: %.1fB" % (total_p/1e9))
print("  FP32:   %.2f GB" % gb(total_b))

gate_up_b = 2 * h * i * n_experts * n_layers * 4
down_b = h * i * n_experts * n_layers * 4
attn_b = attn_p * n_layers * 4
emb_b = emb_p * 4

gc = gate_up_b / 4 / 16
dc = down_b / 4 / 100 * 2
ac = attn_b / 4 / 100 * 2
ec = emb_b / 2
tc = gc + dc + ac + ec

print()
print("  Gate+up (8x):    %s -> %s (ternario 16x)" % (gb(gate_up_b), mb(gc)))
print("  Down (8x):       %s -> %s (SG-HF 100x)" % (gb(down_b), mb(dc)))
print("  Atencion:        %s -> %s (SG-HF 100x)" % (gb(attn_b), mb(ac)))
print("  Embeddings:      %s -> %s (FP16)" % (gb(emb_b), mb(ec)))
print("  %s %s -> %s | %dx" % ("TOTAL".rjust(19), "%.2f GB" % gb(total_b), "%.0f MB" % mb(tc), total_b/tc))
print()
print("  -> Modelo de ~52B comprimido a ~%.0f MB" % mb(tc))
print("  -> Con MLA (KV 16x) + MoE sparsity (2/8 expertos):")
print("     Efectivamente ~1-2 GB por token de inferencia")
print("  -> Corre en laptop con 4-8 GB VRAM")
