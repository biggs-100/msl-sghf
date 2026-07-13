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


class SharedSeedMLP(nn.Module):
    """
    MLP con seed COMPARTIDO entre gate_proj y up_proj.

    En vez de dos seeds independientes (que hace que el error en gate
    se multiplique con up), usa UN seed que genera ambas matrices.

    Forward:
      1. Seed → FFT → modulación → IFFT (holograma compartido)
      2. Expansión Kronecker con heads separados para gate y up:
           W_gate = expand(seed, row_gate, col_gate)
           W_up   = expand(seed, row_up, col_up)
      3. MLP: y = SiLU(x @ W_gate) * (x @ W_up)
      4. down_proj (FractalLinear aparte o Linear normal)
    """

    def __init__(self, hidden_size: int = 2560, intermediate_size: int = 9216,
                 compression: float = 100.0, fractal_down: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Tamaño del seed compartido
        total_params = hidden_size * intermediate_size
        seed_target = max(4, int(total_params / compression))
        p = max(2, min(int(math.sqrt(seed_target)), hidden_size))
        q = max(2, seed_target // p)
        q = min(q, hidden_size)
        self.p, self.q = p, q

        # Factores de expansión
        self.a = math.ceil(intermediate_size / p)
        self.b = math.ceil(hidden_size / q)

        # Seed compartido + modulación FFT
        # Inicializacion corregida: Kronecker triplica varianza
        self.seed = nn.Parameter(torch.randn(p, q) * 0.5)
        fft_h, fft_w = p, q // 2 + 1
        self.freq_scale = nn.Parameter(torch.ones(fft_h, fft_w) * 0.1)
        self.freq_shift = nn.Parameter(torch.zeros(fft_h, fft_w))

        # Bases de expansión — SEPARADAS para gate y up
        # (mismo seed, distinto head)
        self.row_gate = nn.Parameter(torch.randn(p, self.a) * 0.05)
        self.col_gate = nn.Parameter(torch.randn(q, self.b) * 0.05)
        self.row_up = nn.Parameter(torch.randn(p, self.a) * 0.05)
        self.col_up = nn.Parameter(torch.randn(q, self.b) * 0.05)

        # Down projection
        if fractal_down:
            self.down = FractalLinear(intermediate_size, hidden_size,
                                      compression=compression)
        else:
            self.down = nn.Linear(intermediate_size, hidden_size)

    def _generate_gate_up(self):
        """Genera W_gate y W_up desde el seed compartido."""
        p, q, a, b = self.p, self.q, self.a, self.b

        # FFT → modulación → IFFT (seed compartido)
        seed_fft = torch.fft.rfft2(self.seed)
        scale = torch.sigmoid(self.freq_scale)
        seed_mod = seed_fft * (1.0 + scale) + self.freq_shift
        z = torch.fft.irfft2(seed_mod, s=(p, q))  # (p, q)

        # Expansión Kronecker — gate
        W_g = torch.einsum('pq,pa,qb->paqb', z, self.row_gate, self.col_gate)
        W_g = W_g.reshape(p * a, q * b)[:self.intermediate_size, :self.hidden_size]

        # Expansión Kronecker — up (mismo z, distintas bases)
        W_u = torch.einsum('pq,pa,qb->paqb', z, self.row_up, self.col_up)
        W_u = W_u.reshape(p * a, q * b)[:self.intermediate_size, :self.hidden_size]

        return W_g, W_u

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_g, W_u = self._generate_gate_up()
        gate = F.silu(x @ W_g.T)
        up = x @ W_u.T
        hidden = gate * up
        return self.down(hidden)

    def seed_params_count(self) -> int:
        """Parámetros totales del seed compartido + heads."""
        return (self.p * self.q +                    # seed
                self.p * (self.q // 2 + 1) * 2 +     # freq scale+shift
                self.p * self.a * 4 +                # row_gate + col_gate + row_up + col_up
                self.q * self.b * 4)                 # (bases son p×a y q×b)


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
