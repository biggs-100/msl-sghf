# MSL + SG-HF

**Multi-Scale Linear layer** and **Seed-Generated Fractal Weight compression** for LLMs.

## What is this?

Two complementary inventions for extreme compression of neural networks:

### MSL (Multi-Scale Linear)
A drop-in replacement for `nn.Linear` that factorizes W = U · diag(s) · V^T into hierarchical scales. By truncating the last scales at inference time, you get a smaller model that works **WITHOUT fine-tuning**. The student is born from the teacher.

```
Model trained with MSL → truncate last scales → smaller model, same quality (gap ~0)
```

### SG-HF (Seed-Generated Fractal Weights)
Compression via seeds that generate weights under demand:
- **Kronecker expansion**: 50-100× on weights with std > 0.01
- **Ternary quantization** {-1, 0, +1}: 16× on all modern LLMs (Mistral, Llama, Qwen)

### The full pipeline
```
Teacher (Mistral, Llama) → Distill to MSL → Kronecker on MSL factors → Truncatable student
```

## Repository structure

```
sg_hf/
├── msl_v2.py              # MSL layer implementation
├── msl_transformer.py     # GPT with MSL layers
├── core.py                # FractalLinear (SG-HF Kronecker)

msl_demo_v2.py             # MNIST demo
msl_demo_transformer.py    # GPT demo
kaggle_msl_distill.py      # Kaggle notebook (distill TinyLlama → MSL)
WHITEPAPER.md              # Full documentation
```

## Key results

| Experiment | Teacher | Student (truncated) | Gap |
|-----------|---------|-------------------|-----|
| MNIST MLP | 97.74% | 94.88% @ 21× | 2.86pp |
| GPT (864K) | loss 0.53 | loss 0.53 @ 7× | ~0 |
| Mistral-7B ternary | — | COS 0.77 @ 16× | — |

## Limitations

- **No post-hoc compression >16×** on modern LLMs. Weights have std ~0.003 and flat spectrum.
- **MSL requires training from scratch.** Can't apply to existing models without distillation.
- **Full pipeline (distill → MSL → Kronecker) needs scale.** ~100B tokens for production quality 7B model.

## License

MIT
