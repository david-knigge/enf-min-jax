# Adapted from https://github.com/kvfrans/jax-flow/tree/main, by Kevin Frans

import functools
import math
from typing import Any, Tuple
import flax.linen as nn
from flax.linen.initializers import xavier_uniform
import jax
from jax import lax
import jax.numpy as jnp
import ml_collections
from einops import rearrange

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

from typing import Any, Callable, Optional, Tuple, Type, Sequence, Union

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    hidden_size: int
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t):
        x = self.timestep_embedding(t)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
        return x

    # t is between [0, 1].
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = jax.lax.convert_element_type(t, jnp.float32)
        t = t * max_period
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp( -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding
    
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    dropout_prob: float
    num_classes: int
    hidden_size: int

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            rng = self.make_rng('label_dropout')
            drop_ids = jax.random.bernoulli(rng, self.dropout_prob, (labels.shape[0],))
        else:
            drop_ids = force_drop_ids == 1
        labels = jnp.where(drop_ids, self.num_classes, labels)
        return labels
    
    @nn.compact
    def __call__(self, labels, train, force_drop_ids=None):
        embedding_table = nn.Embed(self.num_classes + 1, self.hidden_size, embedding_init=nn.initializers.normal(0.02))

        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = embedding_table(labels)
        return embeddings

class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs):
        """It's just an MLP, so the input shape is (batch, len, emb)."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
                features=self.mlp_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(inputs)
        x = nn.gelu(x)
        # x = nn.Dropout(rate=self.dropout_rate)(x)
        output = nn.Dense(
                features=actual_out_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(x)
        # output = nn.Dropout(rate=self.dropout_rate)(output)
        return output
    
def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]

################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, c):
        # Calculate adaLn modulation parameters.
        c = nn.silu(c)
        c = nn.Dense(6 * self.hidden_size, kernel_init=nn.initializers.constant(0.))(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(c, 6, axis=-1)
        
        # Attention Residual.
        x_norm = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        attn_x = nn.MultiHeadDotProductAttention(kernel_init=nn.initializers.xavier_uniform(),
            num_heads=self.num_heads)(x_modulated, x_modulated)
        x = x + (gate_msa[:, None] * attn_x)

        # MLP Residual.
        x_norm2 = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_x = MlpBlock(mlp_dim=int(self.hidden_size * self.mlp_ratio))(x_modulated2)
        x = x + (gate_mlp[:, None] * mlp_x)
        return x
    
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    out_channels: int
    hidden_size: int

    @nn.compact
    def __call__(self, x, c):
        c = nn.silu(c)
        c = nn.Dense(2 * self.hidden_size, kernel_init=nn.initializers.constant(0))(c)
        shift, scale = jnp.split(c, 2, axis=-1)
        x = modulate(nn.LayerNorm(use_bias=False, use_scale=False)(x), shift, scale)
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

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    class_dropout_prob: float
    num_classes: int
    learn_sigma: bool = False

    @nn.compact
    def __call__(self, z, t, y, train=False, force_drop_ids=None):
        p, c, g = z

        # (x = (B, H, W, C) image, t = (B,) timesteps, y = (B,) class labels)
        in_channels = c.shape[-1]
        if self.learn_sigma:
            out_channels = in_channels * 2
        else:
            out_channels = in_channels

        # Embed patched and poses.
        pos_embed = PosEmb(self.hidden_size, freq=1.0)(p)
        c = nn.Dense(self.hidden_size, kernel_init=nn.initializers.xavier_uniform())(c)
        c = c + pos_embed

        # Embed conditional inputs (timesteps and class labels).
        t = TimestepEmbedder(self.hidden_size)(t)
        y = LabelEmbedder(self.class_dropout_prob, self.num_classes, self.hidden_size)(
            y, train=train, force_drop_ids=force_drop_ids)
        con = t + y

        # Run DiT blocks on input and conditioning.
        for _ in range(self.depth):
            c = DiTBlock(self.hidden_size, self.num_heads, self.mlp_ratio)(c, con)

        # Final layer.
        c = FinalLayer(out_channels, self.hidden_size)(c, con)
        return c
