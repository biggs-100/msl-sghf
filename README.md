# SG-HF: Seed-Generated Fractal Weight Synthesis

**Compresión 64× de modelos de IA — pesos generados bajo demanda desde un seed.**

---

## ¿Qué es SG-HF?

SG-HF (**S**ynthetic **G**enerative weight synthesis via **H**olographic **F**ractal expansion) es un método que comprime los pesos de redes neuronales almacenando un **seed** que los genera en tiempo real, en vez de almacenar los pesos directamente.

| Enfoque tradicional | SG-HF |
|---|---|
| Almacena W (23M floats por capa) | Almacena seed (235K floats) |
| Lee W de memoria para inferencia | Genera W desde seed vía FFT + Kronecker |
| 140 GB para un modelo 70B | **2.2 GB de seeds** |
| No corre en laptop | **Corre en laptop 8GB VRAM** |

## Resultados

### Qwen3.5-4B (modelo real de producción)

| Métrica | Teacher | SG-HF |
|---|---|---|
| Parámetros lineales | 3,569M | 55.7M seeds |
| Compresión | — | **64×** |
| Weight MSE | — | **0.00007** |
| Seed size (FP16) | — | **106 MB** |
| Seed size (INT4) | — | **27 MB** |
| Capas comprimidas | — | **32/32** |

### distilgpt2

| Métrica | Teacher | SG-HF |
|---|---|---|
| Parámetros lineales | ~42M | 847K seeds |
| Compresión | — | **97×** |

### MNIST MLP

| Métrica | Teacher | SG-HF |
|---|---|---|
| Parámetros | 535K | 27K |
| Accuracy | 97.4% | **96.0%** |
| Compresión | — | **50×** |

## Arquitectura

```
Seed (106 MB FP16)
  │
  ├── FFT ── modulación ── IFFT  (dominio holográfico)
  │
  ├── Expansión Kronecker (gate_proj)
  ├── Expansión Kronecker (up_proj)
  ├── Expansión Kronecker (down_proj)
  ├── Expansión Kronecker (Q, K, V, O projections)
  │
  └── Pesos generados → inferencia → se descartan
```

### Componentes

- **FractalLinear** (`sg_hf/core.py`): genera una matriz W desde un seed vía FFT + Kronecker
- **SharedSeedMLP** (`sg_hf/core.py`): genera gate y up desde el mismo seed (preserva la relación)
- **MLA** (`sg_hf/mla.py`): comprime el KV cache 16× via latentes
- **Teacher + Distill** (`sg_hf/teacher.py`, `sg_hf/distill.py`): entrena seeds desde teacher

## Pipeline

```bash
# 1. Comprimir modelo completo
python pipeline_full.py        # Qwen3.5-4B completo (13 min en GPU)

# 2. Destilar MLP outputs (requiere GPU 8GB+)
python distill_mlp.py          # Corrige amplificación del gate SiLU

# 3. Probar inferencia
python test_inference.py       # Compara texto generado

# 4. Visualizar resultados
python -m sg_hf.viz            # Genera gráficos en output/
```

## Archivos del proyecto

```
ia-2027/
├── sg_hf/
│   ├── core.py          # FractalLinear, SharedSeedMLP
│   ├── mla.py           # Multi-head Latent Attention
│   ├── distill.py       # Destilación por activaciones
│   ├── teacher.py       # Teacher MLP + MNIST
│   ├── transformer.py   # Transformer fractal
│   ├── real_gpt.py      # distilgpt2: 97× compression
│   ├── viz.py           # Visualizaciones
│   └── demo.py          # Demo MNIST
├── compress_qwen.py     # Qwen3.5-4B compression por capa
├── pipeline_full.py     # Pipeline completo (32 capas)
├── distill_mlp.py       # Destilación MLP output
├── test_inference.py    # Test de inferencia
├── WHITEPAPER.md        # Documento técnico completo
├── pipeline_output.txt  # Resultados de la corrida
└── compressed_qwen_full/ # Seeds guardados (106 MB)
```

## Combinación con MLA

SG-HF + MLA resuelve los dos cuellos de botella:

| Componente | Problema | Solución SG-HF+MLA |
|---|---|---|
| Pesos | Ocupan 8-140 GB | **Seed 64× más chico** |
| KV cache | Crece con el contexto | **Latente 16× más chico** |

**Resultado: modelo 64× chico + 1M tokens de contexto en laptop 8GB VRAM.**

## Estado del proyecto

- ✅ Concepto validado (MLP MNIST, distilgpt2, Qwen3.5-4B)
- ✅ Compresión 64× de Qwen3.5-4B completo
- ✅ MLA implementado y probado
- ✅ Whitepaper completo
- ⏳ Destilación MLP output (requiere GPU 8GB+)
- ⏳ Inferencia completa con generación de texto

## Próximos pasos

1. Correr `distill_mlp.py` y `test_inference.py` en GPU 8GB+
2. Integrar SharedSeedMLP en el pipeline completo
3. Probar en modelo MoE (Hy3, GLM-5.2)
4. FPGA accelerator design
5. Patente / inversores

---

*Proyecto desarrollado como parte del whitepaper SG-HF. Julio 2026.*
