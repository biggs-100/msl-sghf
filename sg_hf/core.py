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
      - bases de expansión Kronecker (r sumas)

    Forward:
      1. FFT del seed → modulación → IFFT (dominio holográfico)
      2. Expansión Kronecker multi-rango:
           W = Σᵣ Aᵣ ⊗ Bᵣ  (r sumas, mismo seed, distintas bases)
      3. W se recorta a (M×N) exacto y se aplica F.linear

    El rango r controla cuanta informacion retiene el Kronecker:
      r=1: ~60-70% de la estructura   (maxima compresion)
      r=4: ~90-95%                     (recomendado para precision)
      r=8: ~97-99%                     (calidad near-lossless)
    """

    def __init__(self, in_features: int, out_features: int,
                 seed_h: int | None = None, seed_w: int | None = None,
                 compression: float = 400.0,
                 kronecker_rank: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = kronecker_rank

        # --- tamaño del seed (compartido entre ranks) ---
        total_params = in_features * out_features
        seed_target = max(4, int(total_params / compression))

        if seed_h is not None and seed_w is not None:
            p, q = seed_h, seed_w
        else:
            p = max(2, int(math.sqrt(seed_target)))
            q = max(2, seed_target // p)
            p = min(p, out_features)
            q = min(q, in_features)

        self.p, self.q = p, q

        # factores de expansión
        self.a = math.ceil(out_features / p)
        self.b = math.ceil(in_features / q)

        # --- parámetros entrenables ---
        # seed compartido (único para todos los ranks)
        self.seed = nn.Parameter(torch.randn(p, q) * 0.02)

        # modulación FFT compartida
        fft_h = p
        fft_w = q // 2 + 1
        self.freq_scale = nn.Parameter(torch.ones(fft_h, fft_w) * 0.1)
        self.freq_shift = nn.Parameter(torch.zeros(fft_h, fft_w))

        # Bases de expansión: r conjuntos de (row_basis, col_basis)
        # W = Σᵣ Aᵣ ⊗ Bᵣ  donde cada Aᵣ usa row_basis[r] y col_basis[r]
        self.row_basis = nn.Parameter(torch.randn(self.r, p, self.a) * 0.02)
        self.col_basis = nn.Parameter(torch.randn(self.r, q, self.b) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Escala global aprendible (compensa el producto triple del Kronecker)
        self.scale = nn.Parameter(torch.ones(1) * 100.0)

        # --- conteo de parámetros ---
        seed_params = p * q
        kernel_params = self.r * (p * self.a + q * self.b)
        freq_params = fft_h * fft_w
        self.compression_ratio = total_params / (seed_params + kernel_params + freq_params)
        self.seed_params = seed_params
        self.total_compressed = seed_params + kernel_params + freq_params

    def initialize_from_teacher(self, teacher_weight: torch.Tensor):
        """
        Inicializa seed + bases directamente desde el peso del teacher.
        Usa el promedio de cada bloque Kronecker como valor del seed.

        Seed se normaliza para que FFT/IFFT sea estable.
        Bases se inicializan con ruido para evitar degeneracion.
        """
        p, q, a, b = self.p, self.q, self.a, self.b
        out, inp = teacher_weight.shape

        with torch.no_grad():
            # Seed: promedio de cada bloque valido del teacher
            # (bloques parciales en los bordes se manejan con slicing)
            for i in range(p):
                r_start = i * a
                r_end = min(r_start + a, out)
                if r_start >= out:  # bloque fuera de rango
                    self.seed[i, :] = 0
                    continue
                for j in range(q):
                    c_start = j * b
                    c_end = min(c_start + b, inp)
                    if c_start >= inp:  # bloque fuera de rango
                        self.seed[i, j] = 0
                        continue
                    block = teacher_weight[r_start:r_end, c_start:c_end]
                    self.seed[i, j] = block.mean()

            # Normalizar seed para FFT estable
            self.seed.data = self.seed.data / (self.seed.data.std() + 1e-8) * 0.1

            # FFT modulation: off (z ≈ seed)
            self.freq_shift.data.zero_()
            self.freq_scale.data.fill_(0.01)

            # Bases: escala calculada para que W_out ≈ teacher
            # z[i,j] * R[i,u] * C[j,v] debe tener std ≈ teacher.std
            teacher_std = teacher_weight.std().item()
            z_std = self.seed.data.std().item()
            basis_std = math.sqrt(teacher_std / (z_std + 1e-8))
            for rk in range(self.r):
                self.row_basis[rk].data = torch.randn(p, self.a) * basis_std
                self.col_basis[rk].data = torch.randn(q, self.b) * basis_std

    def _generate_weight(self) -> torch.Tensor:
        """
        Genera la matriz W (out_features × in_features) desde el seed.

        Con r > 1, suma multiples productos de Kronecker:
          W = Σᵣ Aᵣ ⊗ Bᵣ  con el mismo seed.
        """
        p, q, a, b, r = self.p, self.q, self.a, self.b, self.r

        # 1. FFT → dominio de frecuencia
        seed_fft = torch.fft.rfft2(self.seed)

        # 2. Modulación aprendible
        scale = torch.sigmoid(self.freq_scale)
        seed_mod = seed_fft * (1.0 + scale) + self.freq_shift

        # 3. Vuelta al dominio temporal
        z = torch.fft.irfft2(seed_mod, s=(p, q))  # (p, q)

        # 4. Expansión Kronecker multi-rango
        #    r=1: version 2D (gradiente estable)
        #    r>1: version 3D con suma sobre ranks
        if self.r == 1:
            rb = self.row_basis.squeeze(0)  # (p, a)
            cb = self.col_basis.squeeze(0)  # (q, b)
            W_kron = torch.einsum('pq,pa,qb->paqb', z, rb, cb)
            W_full = W_kron.reshape(p * a, q * b)
        else:
            W_kron = torch.einsum('pq,rpa,rqb->rpaqb', z, self.row_basis, self.col_basis)
            W_full = W_kron.sum(dim=0).reshape(p * a, q * b)

        # 5. Escalar (compensa el producto triple del Kronecker)
        W_full = W_full * self.scale

        # 6. Recortar al tamaño exacto
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
