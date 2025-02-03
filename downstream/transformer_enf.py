# Adapted from https://github.com/kvfrans/jax-flow/tree/main, by Kevin Frans
import flax.linen as nn
import jax.numpy as jnp


from typing import Any, Callable, Optional, Tuple, Type, Sequence, Union


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    out_dim: Optional[int] = None

    @nn.compact
    def __call__(self, inputs):
        """It's just an MLP, so the input shape is (batch, len, emb)."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
                features=self.mlp_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.normal(stddev=1e-6))(inputs)
        x = nn.gelu(x)
        output = nn.Dense(
                features=actual_out_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.normal(stddev=1e-6))(x)
        return output
    
def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]

class TransformerBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x):
        # Attention Residual.
        x_norm = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        attn_x = nn.MultiHeadDotProductAttention(kernel_init=nn.initializers.xavier_uniform(),
            num_heads=self.num_heads)(x_norm, x_norm)
        x = x + attn_x

        # MLP Residual.
        x_norm2 = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        mlp_x = MlpBlock(mlp_dim=int(self.hidden_size * self.mlp_ratio))(x_norm2)
        x = x + mlp_x
        return x
    
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    out_channels: int
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.out_channels, kernel_init=nn.initializers.constant(0))(x)
        return x

class PosEmb(nn.Module):
    """ RFF positional embedding. """
    embedding_dim: int
    freq: float

    @nn.compact
    def __call__(self, coords: jnp.ndarray) -> jnp.ndarray:
        emb = nn.Dense(self.embedding_dim // 2, kernel_init=nn.initializers.normal(self.freq), use_bias=False)(
            jnp.pi * (coords + 1))  # scale to [0, 2pi]
        return nn.Dense(self.embedding_dim)(jnp.sin(jnp.concatenate([coords, emb, emb + jnp.pi / 2.0], axis=-1)))

class TransformerClassifier(nn.Module):
    """
    Transformer model.
    """
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    num_classes: int

    @nn.compact
    def __call__(self, p_0, c_0, g_0):
        # Embed patched and poses.
        pos_embed = PosEmb(self.hidden_size, freq=1.0)(p_0)
        c = nn.Dense(self.hidden_size, kernel_init=nn.initializers.xavier_uniform())(c_0)
        c = c + pos_embed

        # Run DiT blocks on input and conditioning.
        for _ in range(self.depth):
            c = TransformerBlock(self.hidden_size, self.num_heads, self.mlp_ratio)(c)

        # Final layer.
        return FinalLayer(self.num_classes, self.hidden_size)(jnp.mean(c, axis=1))


class TransformerForecaster(nn.Module):
    """
    Transformer model.
    """
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    learn_pose: bool = True

    @nn.compact
    def __call__(self, p_0, c_0, g_0):
        
        in_channels = c_0.shape[-1]
        if self.learn_pose:
            out_channels = in_channels + 2
        else:
            out_channels = in_channels

        # Embed patched and poses.
        pos_embed = PosEmb(self.hidden_size, freq=1.0)(p_0)
        c = nn.Dense(self.hidden_size, kernel_init=nn.initializers.xavier_uniform())(c_0)
        c = c + pos_embed

        # Run DiT blocks on input and conditioning.
        for _ in range(self.depth):
            c = TransformerBlock(self.hidden_size, self.num_heads, self.mlp_ratio)(c)

        # Final layer.
        c = FinalLayer(out_channels, self.hidden_size)(c)

        # Split into p, c if learning pose
        if self.learn_pose:
            c, p = c[..., :-2], c[..., -2:]
            return p_0 + p, c_0 + c, g_0
        else:
            return p_0, c_0 + c, g_0
