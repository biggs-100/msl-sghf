"""
MSL Transformer: GPT-like modelo con MSL FFN layers.

Cada FFN (SwiGLU) usa MSLinear en vez de nn.Linear para sus tres proyecciones.
Esto permite truncar escalas y obtener un modelo funcional SIN fine-tuning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sg_hf.msl_v2 import MSLinear, compute_msl_regularization


# ─── Configuracion de escalas para FFN ────────────────────────────

FFN_SCALE_CONFIG = {
    'small':  [8, 16, 32],      # para hidden ~128
    'medium': [16, 32, 64],     # para hidden ~512
    'large':  [32, 64, 128],    # para hidden ~1024+
}

def get_msl_ffn_config(hidden: int, intermediate: int, profile: str = 'auto'):
    """
    Devuelve configs de escalas para gate/up/down de una FFN.

    Args:
        hidden: dimension del modelo
        intermediate: dimension de la FFN intermedia
        profile: 'small', 'medium', 'large', o 'auto'
    """
    if profile == 'auto':
        if hidden <= 128:
            profile = 'small'
        elif hidden <= 512:
            profile = 'medium'
        else:
            profile = 'large'

    base = FFN_SCALE_CONFIG[profile]
    # down_proj necesita mas rank porque comprime intermediate→hidden
    down_scales = [r * 2 for r in base]

    return {
        'gate': base,
        'up': base,
        'down': down_scales,
    }


# ─── MSL FFN ──────────────────────────────────────────────────────

class MSL_FFN(nn.Module):
    """
    Feed-Forward Network con MSL (SwiGLU).

    Reemplaza las 3 proyecciones lineales con MSLinear.
    """

    def __init__(self, hidden: int, intermediate: int,
                 scale_config: dict | None = None):
        super().__init__()
        self.hidden = hidden
        self.intermediate = intermediate

        if scale_config is None:
            scale_config = get_msl_ffn_config(hidden, intermediate, 'auto')

        self.gate = MSLinear(hidden, intermediate,
                             scale_ranks=scale_config['gate'])
        self.up = MSLinear(hidden, intermediate,
                           scale_ranks=scale_config['up'])
        self.down = MSLinear(intermediate, hidden,
                             scale_ranks=scale_config['down'])

    def set_active_scales(self, k: int):
        self.gate.set_active_scales(k)
        self.up.set_active_scales(k)
        self.down.set_active_scales(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down(SiLU(gate(x)) * up(x))
        gate = F.silu(self.gate(x))
        up = self.up(x)
        return self.down(gate * up)


# ─── Attention ────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head causal attention (standard GPT)."""

    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        assert hidden % n_heads == 0
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads

        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.proj(y)


# ─── Transformer Block ────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """GPT-like block: attention + MSL FFN with pre-norm."""

    def __init__(self, hidden: int, n_heads: int, intermediate: int,
                 ffn_scale_config: dict | None = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = CausalSelfAttention(hidden, n_heads)
        self.ln2 = nn.LayerNorm(hidden)
        self.ffn = MSL_FFN(hidden, intermediate, ffn_scale_config)

    def set_active_scales(self, k: int):
        self.ffn.set_active_scales(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ─── MSL GPT ──────────────────────────────────────────────────────

class MSL_GPT(nn.Module):
    """
    GPT-style causal LM con MSL FFN layers.

    Args:
        vocab_size: tamanio del vocabulario
        hidden: dimension del modelo
        n_heads: numero de cabezas de atencion
        n_layers: numero de capas transformer
        context: longitud maxima de contexto
        ffn_scale_config: config de escalas para FFN (o None para auto)
    """

    def __init__(self, vocab_size: int = 96, hidden: int = 128,
                 n_heads: int = 4, n_layers: int = 4, context: int = 128,
                 ffn_scale_config: dict | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.context = context
        self.n_layers = n_layers

        intermediate = 4 * hidden

        self.token_embed = nn.Embedding(vocab_size, hidden)
        self.pos_embed = nn.Parameter(torch.randn(1, context, hidden) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden, n_heads, intermediate, ffn_scale_config)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)

        # Weight tying
        self.token_embed.weight = self.head.weight

        # Init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def set_active_scales(self, k: int):
        """Cambia el nivel de truncamiento de TODOS los MSL FFN."""
        for block in self.blocks:
            block.set_active_scales(k)

    def get_msl_layers(self) -> list[MSLinear]:
        """Devuelve TODAS las capas MSL del modelo."""
        layers = []
        for block in self.blocks:
            layers.extend([block.ffn.gate, block.ffn.up, block.ffn.down])
        return layers

    def compute_regularization(self, **kwargs) -> dict[str, torch.Tensor]:
        """Regularizacion espectral sobre todas las capas MSL."""
        return compute_msl_regularization(self.get_msl_layers(), **kwargs)

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None,
                return_reg: bool = False,
                reg_kwargs: dict | None = None) -> tuple:
        """
        Forward del modelo.

        Args:
            idx: (B, T) tokens de entrada
            targets: (B, T) tokens objetivo (opcional)
            return_reg: si True, incluye losses de regularizacion
            reg_kwargs: kwargs para compute_regularization

        Returns:
            (logits, loss) o (logits, loss, reg_losses)
        """
        B, T = idx.shape
        assert T <= self.context, f"Sequence length {T} > context {self.context}"

        # Embeddings + pos
        x = self.token_embed(idx)
        x = x + self.pos_embed[:, :T, :]

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1)
            )

        if return_reg and reg_kwargs:
            reg = self.compute_regularization(**reg_kwargs)
            return logits, loss, reg

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0) -> torch.Tensor:
        """Genera tokens autoregresivamente."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

    @property
    def n_scales(self) -> int:
        """Numero de escalas (asumiendo que todas las capas MSL tienen el mismo)."""
        msl = self.get_msl_layers()
        return msl[0].n_scales if msl else 1

    def count_params(self) -> dict:
        """Conteo de parametros detallado."""
        total = sum(p.numel() for p in self.parameters())
        msl_mods = self.get_msl_layers()
        msl_params = sum(p.numel() for mod in msl_mods for p in mod.parameters())
        return {
            'total': total,
            'msl': msl_params,
            'non_msl': total - msl_params,
        }


# ─── Util: Crear modelo con config resumida ───────────────────────

def create_msl_gpt(vocab_size: int = 96, hidden: int = 128,
                   n_layers: int = 4, profile: str = 'auto') -> MSL_GPT:
    """Crea un MSL_GPT con config de escalas automatica."""
    ffn_cfg = get_msl_ffn_config(hidden, 4 * hidden, profile)
    return MSL_GPT(
        vocab_size=vocab_size,
        hidden=hidden,
        n_heads=max(2, hidden // 32),
        n_layers=n_layers,
        context=128,
        ffn_scale_config=ffn_cfg,
    )
