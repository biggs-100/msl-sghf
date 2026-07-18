"""
MSL v2: Multi-Scale Linear — Refinado.

Cambios vs v1:
  1. Sin QR en forward (entrenamiento mas rapido, sin rotacion de subespacios)
  2. Ortogonalidad como loss durante entrenamiento (en vez de QR)
  3. Entrenamiento CONJUNTO de todas las escalas (no progresivo)
  4. Regularizacion fuerte: sorting + orthogonality + L1 espectral
  5. Configuracion de escalas por capa (cada capa puede tener distinta config)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Capa MSL ─────────────────────────────────────────────────────

class MSLinear(nn.Module):
    """
    Multi-Scale Linear v2.

    W = U @ diag(s) @ V  (factorizacion sin ortogonalidad explicita)

    Propiedades:
      - truncar a k escalas = U[:,:r] @ diag(s[:r]) @ V[:r,:]
      - sin fine-tuning: el alumno ES el profesor truncado
      - escalas jerarquicas via regularizacion espectral

    Args:
        in_features: N
        out_features: M
        scale_ranks: lista de ranks por escala
    """

    def __init__(self, in_features: int, out_features: int,
                 scale_ranks: list[int] | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if scale_ranks is None:
            scale_ranks = [8, 16, 32]

        self.scale_ranks = list(scale_ranks)
        self.n_scales = len(scale_ranks)
        self.total_rank = sum(scale_ranks)

        # Limites de escala
        self.cum_ranks = [0]
        for r in self.scale_ranks:
            self.cum_ranks.append(self.cum_ranks[-1] + r)
        self._R = self.total_rank

        # Factores: se almacenan SIN ortogonalidad explicita
        # La regularizacion los mantiene cerca de ortogonales
        self.U = nn.Parameter(torch.randn(out_features, self._R) * 0.02)
        self.s = nn.Parameter(torch.randn(self._R) * 0.05)
        self.V = nn.Parameter(torch.randn(self._R, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Inicializar s con decay exponencial
        with torch.no_grad():
            for i in range(self._R):
                self.s.data[i] = math.exp(-i * 0.2) * 2.0

        # Control de escalas activas
        self._active_scales = self.n_scales

        # Metricas
        self.full_params = in_features * out_features
        self.compressed_params = out_features * self._R + self._R + self._R * in_features + out_features
        self.compression_ratio = self.full_params / self.compressed_params

    # ─── control de escalas ────────────────────────────────────────

    def set_active_scales(self, k: int):
        self._active_scales = max(1, min(k, self.n_scales))

    def get_active_rank(self) -> int:
        return self.cum_ranks[self._active_scales]

    def get_efficiency(self) -> float:
        r = self.get_active_rank()
        msl_ops = r * (self.in_features + self.out_features + 1)
        linear_ops = self.out_features * self.in_features
        return linear_ops / msl_ops if msl_ops > 0 else 0.0

    def get_scales_summary(self) -> list[dict]:
        """Info de cada escala: rank, compression, singular values."""
        summary = []
        for k in range(1, self.n_scales + 1):
            r = self.cum_ranks[k]
            comp = self.compression_ratio * (self._R / r)
            s_range = (self.s[:r].min().item(), self.s[:r].max().item())
            summary.append({
                'k': k, 'rank': r, 'compression': comp,
                's_min': s_range[0], 's_max': s_range[1]
            })
        return summary

    # ─── forward y weight ──────────────────────────────────────────

    def _build_weight(self, k: int | None = None) -> torch.Tensor:
        """Construye W (out, in) a partir de los factores activos."""
        if k is None:
            k = self._active_scales
        r = self.cum_ranks[k]
        return self.U[:, :r] @ torch.diag(self.s[:r]) @ self.V[:r, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward factorizado SIN QR.

        x: (..., N) -> (..., M)
        """
        k = self._active_scales
        r = self.cum_ranks[k]

        # y = x @ V^T @ diag(s) @ U^T + b
        xV = x @ self.V[:r, :].T    # (..., r)
        xV = xV * self.s[:r]        # (..., r)
        y = xV @ self.U[:, :r].T    # (..., M)
        return y + self.bias

    def extra_repr(self) -> str:
        return (f'in={self.in_features}, out={self.out_features}, '
                f'scales={self.scale_ranks}, R={self._R}, '
                f'comp={self.compression_ratio:.1f}x')


# ─── Regularizaciones ─────────────────────────────────────────────

def compute_msl_regularization(
    layers: list[MSLinear],
    alpha_sort: float = 0.5,
    alpha_orth: float = 0.1,
    alpha_l1_s: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """
    Computa todas las regularizaciones para una lista de capas MSL.

    Returns:
        dict con 'sort', 'orth', 'l1_s', 'total'
    """
    sort_loss = 0.0
    orth_loss = 0.0
    l1_s_loss = 0.0

    for layer in layers:
        s = layer.s
        R = layer._R

        # --- Sorting loss: penaliza s[i] < s[i+1] ---
        diff = s[:-1] - s[1:]
        sort_loss = sort_loss + torch.relu(-diff + 0.001).sum()

        # --- Orthogonality loss: ||U^T U - I|| + ||V V^T - I|| ---
        U, V = layer.U, layer.V
        # U: (M, R), queremos U^T U ≈ I  (R×R)
        utu = U.T @ U
        I_R = torch.eye(R, device=U.device)
        orth_loss = orth_loss + (utu - I_R).pow(2).sum()
        # V: (R, N), queremos V V^T ≈ I  (R×R)
        vvt = V @ V.T
        orth_loss = orth_loss + (vvt - I_R).pow(2).sum()

        # --- L1 en s: fomenta esparsidad en componentes tardios ---
        l1_s_loss = l1_s_loss + s.abs().sum()

    n = max(len(layers), 1)
    return {
        'sort': alpha_sort * sort_loss / n,
        'orth': alpha_orth * orth_loss / n,
        'l1_s': alpha_l1_s * l1_s_loss / n,
        'total': (alpha_sort * sort_loss + alpha_orth * orth_loss
                  + alpha_l1_s * l1_s_loss) / n,
    }


# ─── MSMLP v2 ─────────────────────────────────────────────────────

class MSMLP(nn.Module):
    """
    MLP con MSLinear v2, configuracion de escalas por capa.

    Permite definir escalas distintas para cada capa, ej:
      scale_config = {
          'hidden': [8, 16, 32],      # capas ocultas
          'first':  [16, 32, 32],     # primera capa (mas rank)
      }
    """

    def __init__(self, layer_sizes: list[int],
                 scale_config: list[int] | dict = None):
        super().__init__()
        self.layer_sizes = layer_sizes

        # Resolver config de escalas por capa
        n_hidden = len(layer_sizes) - 2
        if scale_config is None:
            scale_config = [8, 16, 32]

        if isinstance(scale_config, list):
            # Misma config para todas las capas ocultas
            per_layer = [scale_config] * n_hidden
        elif isinstance(scale_config, dict):
            per_layer = []
            for i in range(n_hidden):
                if i == 0:
                    per_layer.append(scale_config.get('first', scale_config.get('hidden', [8, 16, 32])))
                else:
                    per_layer.append(scale_config.get('hidden', [8, 16, 32]))

        self.layers = nn.ModuleList()
        self.msl_indices = []

        for i in range(n_hidden):
            in_dim, out_dim = layer_sizes[i], layer_sizes[i + 1]
            self.layers.append(MSLinear(in_dim, out_dim,
                                        scale_ranks=per_layer[i]))
            self.msl_indices.append(len(self.layers) - 1)

        self.output = nn.Linear(layer_sizes[-2], layer_sizes[-1])

    @property
    def n_scales(self) -> int:
        return self.layers[self.msl_indices[0]].n_scales if self.msl_indices else 1

    def set_active_scales(self, k: int):
        for idx in self.msl_indices:
            self.layers[idx].set_active_scales(k)

    def get_msl_layers(self) -> list[MSLinear]:
        return [self.layers[idx] for idx in self.msl_indices]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x)

    def compute_regularization(self, **kwargs) -> dict[str, torch.Tensor]:
        return compute_msl_regularization(self.get_msl_layers(), **kwargs)

    def count_compressed_params(self) -> int:
        return sum(self.layers[idx].compressed_params
                   for idx in self.msl_indices) if hasattr(self, 'msl_indices') else 0

    def count_full_params(self) -> int:
        return sum(self.layers[idx].full_params
                   for idx in self.msl_indices) if hasattr(self, 'msl_indices') else 0

    def compression_ratio(self) -> float:
        full = self.count_full_params()
        comp = self.count_compressed_params()
        return full / comp if comp > 0 else 1.0

    def get_scales_summary(self) -> dict[int, list[dict]]:
        """Resumen por capa de las escalas."""
        return {i: layer.get_scales_summary()
                for i, layer in enumerate(self.get_msl_layers())}
