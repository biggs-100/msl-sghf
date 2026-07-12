"""
SG-HF Core: FractalLinear y FractalMLP.

El corazón del invento: generar pesos desde un seed 400× más chico
usando FFT + expansión Kronecker.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FractalLinear(nn.Module):
    """
    Linear layer cuyos pesos se generan desde un seed.

    En vez de almacenar W ∈ ℝ^{M×N}, almacena:
      - seed S ∈ ℝ^{p×q} (400× más chico que M×N)
      - filtros de modulación en frecuencia (FFT)
      - bases de expansión Kronecker

    Forward:
      1. FFT del seed → modulación → IFFT (dominio holográfico)
      2. Expansión Kronecker: cada elemento S'[i,j] genera
         un bloque (a×b) via producto exterior de vectores base
      3. W se recorta a (M×N) exacto y se aplica F.linear
    """

    def __init__(self, in_features: int, out_features: int,
                 seed_h: int | None = None, seed_w: int | None = None,
                 compression: float = 400.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # --- tamaño del seed ---
        total_params = in_features * out_features
        seed_target = max(4, int(total_params / compression))

        if seed_h is not None and seed_w is not None:
            p, q = seed_h, seed_w
        else:
            # seed aproximadamente cuadrado
            p = max(2, int(math.sqrt(seed_target)))
            q = max(2, seed_target // p)
            # ajustar para que ni p ni q superen M o N
            p = min(p, out_features)
            q = min(q, in_features)

        self.p, self.q = p, q

        # factores de expansión (con techo para cubrir dimensión exacta)
        self.a = math.ceil(out_features / p)   # expansión por fila
        self.b = math.ceil(in_features / q)    # expansión por columna

        # --- parámetros entrenables ---
        # seed: la representación comprimida
        self.seed = nn.Parameter(torch.randn(p, q) * 0.02)

        # modulación en frecuencia (FFT "holográfica")
        fft_h = p
        fft_w = q // 2 + 1   # rfft2 solo guarda mitad simétrica
        self.freq_scale = nn.Parameter(torch.ones(fft_h, fft_w) * 0.1)
        self.freq_shift = nn.Parameter(torch.zeros(fft_h, fft_w))

        # bases de expansión Kronecker
        #  W[i*a:(i+1)*a, j*b:(j+1)*b] ≈ S'[i,j] · outer(row_basis[i], col_basis[j])
        self.row_basis = nn.Parameter(torch.randn(p, self.a) * 0.02)
        self.col_basis = nn.Parameter(torch.randn(q, self.b) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # --- conteo de parámetros ---
        seed_params = p * q
        kernel_params = p * self.a + q * self.b
        freq_params = fft_h * fft_w
        self.compression_ratio = total_params / (seed_params + kernel_params + freq_params)
        self.seed_params = seed_params
        self.total_compressed = seed_params + kernel_params + freq_params

    def _generate_weight(self) -> torch.Tensor:
        """
        Genera la matriz W (out_features × in_features) desde el seed.
        """
        device = self.seed.device
        p, q, a, b = self.p, self.q, self.a, self.b

        # 1. FFT → dominio de frecuencia
        seed_fft = torch.fft.rfft2(self.seed)  # (p, q//2+1) complex

        # 2. Modulación aprendible (el "holograma" se ajusta)
        scale = torch.sigmoid(self.freq_scale)  # [0, 1]
        seed_mod = seed_fft * (1.0 + scale) + self.freq_shift

        # 3. Vuelta al dominio temporal
        seed_time = torch.fft.irfft2(seed_mod, s=(p, q))  # (p, q)

        # 4. Expansión Kronecker vectorizada
        #    W[i*a:(i+1)*a, j*b:(j+1)*b] = S[i,j] · outer(R[i], C[j])
        #    einsum: (p,q) × (p,a) × (q,b) → (p,a,q,b) → reshape (p*a, q*b)
        W_kron = torch.einsum('pq,pa,qb->paqb', seed_time, self.row_basis, self.col_basis)
        W_full = W_kron.reshape(p * a, q * b)  # (out_full, in_full)

        # 5. Recortar al tamaño exacto
        return W_full[:self.out_features, :self.in_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._generate_weight()
        return F.linear(x, W, self.bias)

    def extra_repr(self) -> str:
        return (f'in={self.in_features}, out={self.out_features}, '
                f'seed=({self.p}×{self.q}), '
                f'compression={self.compression_ratio:.1f}x')


class FractalMLP(nn.Module):
    """
    MLP donde cada capa lineal es FractalLinear.

    Útil para demostración y experimentación.
    """

    def __init__(self, layer_sizes: list[int], compression: float = 400.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layer_sizes = layer_sizes

        for i in range(len(layer_sizes) - 2):
            in_dim, out_dim = layer_sizes[i], layer_sizes[i + 1]
            self.layers.append(FractalLinear(in_dim, out_dim, compression=compression))

        # última capa siempre densa (output logits, muy chica)
        self.output = nn.Linear(layer_sizes[-2], layer_sizes[-1])

    def forward(self, x: torch.Tensor, return_activations: bool = False):
        activations = [x]
        # flatten si viene de CNN (MNIST: batch×1×28×28 → batch×784)
        h = x.view(x.size(0), -1) if x.dim() > 2 else x
        for layer in self.layers:
            h = torch.relu(layer(h))
            activations.append(h)
        h = self.output(h)
        if return_activations:
            return h, activations[1:]  # sin la entrada
        return h

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def compressed_params(self) -> int:
        return sum(l.total_compressed for l in self.layers) + sum(
            p.numel() for p in self.output.parameters()
        )

    def compression_ratio(self) -> float:
        t = self.total_params()
        c = self.compressed_params()
        return t / c if c > 0 else 0.0
