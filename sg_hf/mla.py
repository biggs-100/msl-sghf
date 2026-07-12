"""
Multi-head Latent Attention (MLA).

Comprime el KV cache guardando un codigo latente por token
en vez de K y V completos. Inspirado en DeepSeek-V2/V3.

SG-HF comprime los pesos de las proyecciones (W_kv_proj, W_k_expand, etc).
MLA comprime el cache de activaciones (codigo latente vs K,V completo).
Juntos: modelo 64× chico + KV cache 16× chico.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sg_hf.core import FractalLinear


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention.

    Proyecciones:
      W_q: hidden → n_heads * head_dim (Q normal, sin comprimir en cache)
      W_kv_proj: hidden → 2 * n_kv_heads * latent_dim (genera z_k, z_v)
      W_k_expand: latent_dim → head_dim (expande z_k a K)
      W_v_expand: latent_dim → head_dim (expande z_v a V)
      W_o: n_heads * head_dim → hidden

    Cache almacena: (z_k, z_v) = codigos latentes (2 * latent_dim por token)
    En vez de: (k, v) = (2 * n_kv_heads * head_dim por token)

    Ratio de compresion del cache: (n_kv_heads * head_dim) / latent_dim
    """

    def __init__(
        self,
        hidden_size: int = 2560,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        head_dim: int = 256,
        latent_ratio: float = 4.0,      # compresion del cache
        compression: float = 50.0,      # compresion SG-HF de pesos
        use_fractal: bool = True,       # si False, usa nn.Linear normal
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.latent_dim = max(1, int(head_dim / latent_ratio))

        Linear = FractalLinear if use_fractal else nn.Linear

        # Q normal (no pasa por cache)
        self.q_proj = Linear(hidden_size, n_heads * head_dim,
                             compression=compression)

        # Proyeccion conjunta K,V → codigos latentes
        # Entrada: hidden_size
        # Salida: 2 * n_kv_heads * latent_dim (z_k para cada kv head, z_v)
        kv_out = 2 * n_kv_heads * self.latent_dim
        self.kv_proj = Linear(hidden_size, kv_out, compression=compression)

        # Expansores: latente → K o V completo
        # (se comparten entre todos los tokens)
        self.k_expand = nn.Linear(self.latent_dim, head_dim, bias=False)
        self.v_expand = nn.Linear(self.latent_dim, head_dim, bias=False)

        # Output projection
        self.o_proj = Linear(n_heads * head_dim, hidden_size,
                             compression=compression)

        self._latent_ratio = latent_ratio

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple | None = None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Args:
            x: (batch, seq_len, hidden_size)
            past_kv: (z_k, z_v) de tokens anteriores, shapes (B, T_past, n_kv_heads, latent_dim)
            use_cache: si True, devuelve (z_k, z_v) para cachear

        Returns:
            output: (batch, seq_len, hidden_size)
            new_kv: (z_k, z_v) para agregar al cache
        """
        B, T, C = x.shape

        # Q
        q = self.q_proj(x)  # (B, T, n_heads * head_dim)

        # Codigos latentes K,V
        kv = self.kv_proj(x)  # (B, T, 2 * n_kv_heads * latent_dim)
        z_k, z_v = kv.chunk(2, dim=-1)

        # Reorganizar latentes: (B, T, n_kv_heads, latent_dim)
        z_k = z_k.view(B, T, self.n_kv_heads, self.latent_dim)
        z_v = z_v.view(B, T, self.n_kv_heads, self.latent_dim)

        # Cache: concatenar con pasado
        if past_kv is not None:
            z_k = torch.cat([past_kv[0], z_k], dim=1)
            z_v = torch.cat([past_kv[1], z_v], dim=1)

        full_T = z_k.shape[1]

        # Expandir latentes a K,V completos (SOLO cuando se necesita)
        # k_proj: (B, T, n_kv_heads, latent_dim) → (B, T, n_kv_heads, head_dim)
        k = self.k_expand(z_k)
        v = self.v_expand(z_v)

        # Reshape Q: (B, T, n_heads * head_dim) → (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Reshape K,V: (B, T, n_kv_heads, head_dim) → (B, n_kv_heads, T, head_dim)
        k = k.transpose(1, 2)  # (B, n_kv_heads, T, head_dim)
        v = v.transpose(1, 2)

        # GQA: expandir kv heads para matchear n_heads
        if self.n_kv_heads < self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k[:, :, None, :, :].expand(B, self.n_kv_heads, repeat,
                                           full_T, self.head_dim)
            k = k.reshape(B, self.n_heads, full_T, self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.n_kv_heads, repeat,
                                           full_T, self.head_dim)
            v = v.reshape(B, self.n_heads, full_T, self.head_dim)

        # Atencion escalada
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, n_heads, T, head_dim)

        # Concatenar cabezas
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)

        # Output projection
        output = self.o_proj(out)

        new_kv = (z_k[:, -T:], z_v[:, -T:]) if use_cache else None

        return output, new_kv

    def cache_size_per_token(self) -> int:
        """Tamano del cache MLA en bytes (FP16) por token."""
        return 2 * self.n_kv_heads * self.latent_dim * 2  # 2 bytes FP16

    def cache_savings(self) -> float:
        """Ratio de compresion del cache vs atencion estandar."""
        standard = 2 * self.n_kv_heads * self.head_dim * 2  # bytes FP16
        mla = self.cache_size_per_token()
        return standard / mla


class StandardAttention(nn.Module):
    """Atencion estandar (para comparacion)."""

    def __init__(self, hidden_size=2560, n_heads=16, n_kv_heads=4, head_dim=256):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, n_heads * head_dim)
        self.k_proj = nn.Linear(hidden_size, n_kv_heads * head_dim)
        self.v_proj = nn.Linear(hidden_size, n_kv_heads * head_dim)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_size)

    def forward(self, x, past_kv=None, use_cache=True):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=1)
            v = torch.cat([past_kv[1], v], dim=1)

        # GQA repeat
        if self.n_kv_heads < self.n_heads:
            r = self.n_heads // self.n_kv_heads
            k = k.unsqueeze(3).expand(-1, -1, -1, r, -1).reshape(
                B, -1, self.n_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, r, -1).reshape(
                B, -1, self.n_heads, self.head_dim)

        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        output = self.o_proj(out)

        new_kv = (k[:, -T:].transpose(1, 2), v[:, -T:].transpose(1, 2)) if use_cache else None

        return output, new_kv


def test():
    """Verifica que MLA funciona y calcula compresion del cache."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Config Qwen3.5-4B
    cfg = dict(hidden_size=2560, n_heads=16, n_kv_heads=4, head_dim=256)

    mla = MultiHeadLatentAttention(**cfg, latent_ratio=4.0).to(device)
    std = StandardAttention(**cfg).to(device)

    # Forward
    x = torch.randn(2, 128, 2560, device=device)
    out_mla, cache_mla = mla(x)
    out_std, cache_std = std(x)

    print(f"MLA output:  {list(out_mla.shape)}")
    print(f"Std output:  {list(out_std.shape)}")

    # Tamanos del cache
    def cache_bytes(elem):
        return elem.element_size() * elem.nelement()

    # Standard: (k, v) cada uno (B, T, n_kv_heads, head_dim)
    k_std, v_std = cache_std
    std_bytes = cache_bytes(k_std) + cache_bytes(v_std)
    print(f"\nStandard KV cache ({128} tokens): {std_bytes / 1024:.1f} KB")

    # MLA: (z_k, z_v) cada uno (B, T, n_kv_heads, latent_dim)
    z_k, z_v = cache_mla
    mla_bytes = cache_bytes(z_k) + cache_bytes(z_v)
    print(f"MLA KV cache  ({128} tokens): {mla_bytes / 1024:.1f} KB")

    saving = std_bytes / mla_bytes
    print(f"\nCompresion del cache: {saving:.0f}x")

    # Proyeccion a 1M tokens
    t = 1_000_000
    std_1m = t * cfg['n_kv_heads'] * cfg['head_dim'] * 2 * 2  # bytes
    mla_1m = t * 2 * cfg['n_kv_heads'] * mla.latent_dim * 2
    print(f"\nProyeccion a {t:,} tokens:")
    print(f"  Standard: {std_1m / 1e9:.1f} GB")
    print(f"  MLA:      {mla_1m / 1e9:.1f} GB")

    # + SG-HF compression
    print(f"\nSG-HF + MLA en Qwen3.5-4B:")
    print(f"  Seeds:       92 MB")
    print(f"  Cache (1M):  {mla_1m / 1e9:.1f} GB")
    print(f"  Total:       {(92/1024 + mla_1m/1e9):.2f} GB")
    print(f"  -> Corre en laptop 8GB VRAM? {'SI' if (92/1024 + mla_1m/1e9) < 7 else 'NO'}")


if __name__ == '__main__':
    test()
