"""
SG-HF Ternary: cuantizacion {-1, 0, +1} + escala por fila.

Para pesos que NO se pueden comprimir con Kronecker:
  - Gate/up projections (w1/w3) en SwiGLU MLPs
  - Cualquier peso con std < 0.01

El error independiente por elemento del ternario es mucho menos
danino para SiLU que el error estructurado de Kronecker.

Referencia: TWN (Ternary Weight Networks), 2016.
"""

import torch
import torch.nn.functional as F


def ternary_quantize(W: torch.Tensor, threshold_scale: float = 0.7) -> tuple:
    """
    Cuantizacion ternaria optima por fila (TWN).

    Para cada fila de W:
      1. threshold = threshold_scale * mean(|W_row|)
      2. mask[i] = +1 if W[i] > threshold
                   -1 if W[i] < -threshold
                    0 otherwise
      3. scale = mean(|W_row_i| for active_i)
      4. W_q = mask * scale

    Args:
        W: Peso original (out_features, in_features)
        threshold_scale: Fraccion de mean(|W|) como umbral

    Returns:
        W_q: Peso cuantizado
        mask: Mascara ternaria {-1, 0, +1}
        scale: Escala optima por fila (out_features,)
    """
    th = threshold_scale * W.abs().mean(dim=1, keepdim=True)
    mask = torch.where(W > th, 1.0, torch.where(W < -th, -1.0, 0.0))
    active = (mask != 0).float()
    scale = (W.abs() * active).sum(dim=1) / active.sum(dim=1).clamp(min=1)
    W_q = mask * scale.unsqueeze(1)
    return W_q, mask, scale


def get_compressed_size(mask: torch.Tensor, scale: torch.Tensor) -> tuple:
    """
    Calcula tamanos de almacenamiento.

    Args:
        mask: Mascara ternaria {-1, 0, +1}
        scale: Escala por fila

    Returns:
        (bits_totales, bytes_totales, compresion_vs_fp32)
    """
    n_elements = mask.numel()
    n_rows = mask.shape[0]

    # Almacenamiento: 2 bits por elemento (ternario) + 16 bits por fila (escala)
    bits = n_elements * 2 + n_rows * 16
    bytes_total = bits / 8
    bytes_fp32 = n_elements * 4

    return int(bits), int(bytes_total), bytes_fp32 / bytes_total


def reconstruct(mask: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reconstruye peso desde mascara ternaria + escala."""
    return mask * scale.unsqueeze(1)


def compress_model_weights(model_path: str, output_dir: str,
                           threshold: float = 0.7) -> list:
    """
    Comprime todos los pesos lineales de un modelo safetensors.

    Args:
        model_path: Ruta al archivo .safetensors
        output_dir: Directorio de salida
        threshold: Umbral ternario

    Returns:
        Lista de resultados por peso
    """
    import os
    from safetensors import safe_open

    os.makedirs(output_dir, exist_ok=True)
    results = []

    with safe_open(model_path, framework='pt', device='cpu') as f:
        keys = [k for k in f.keys()
                if k.endswith('.weight') and 'norm' not in k.lower()]

        for key in keys:
            W = f.get_tensor(key).float()
            W_q, mask, scale = ternary_quantize(W, threshold)

            name = key.replace('.', '_')
            torch.save({
                'mask': mask.cpu(),
                'scale': scale.cpu(),
                'shape': W.shape,
                'wmse': F.mse_loss(W_q, W).item(),
                'sparsity': (mask == 0).float().mean().item(),
            }, os.path.join(output_dir, f'{name}.pt'))

            results.append({
                'key': key,
                'shape': list(W.shape),
                'r2': 1 - F.mse_loss(W_q, W).item() / W.var().item(),
                'sparsity': (mask == 0).float().mean().item(),
            })

    return results
