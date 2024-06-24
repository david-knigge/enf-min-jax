from typing import Union

import jax
import jax.numpy as jnp
from jax.nn import gelu, softmax
from flax import linen as nn


# For typing
from enf.bi_invariant._base_bi_invariant import BaseBiInvariant


class RFFEmbedding(nn.Module):
    embedding_dim: int
    learnable_coefficients: bool
    std: float

    def setup(self):
        # Make sure we have an even number of hidden features.
        assert (
            not self.embedding_dim % 2.0
        ), "For the Fourier Features hidden_dim should be even to calculate them correctly."

        # Store pi
        self.pi = 2 * jnp.pi

        # Embedding layer
        self.coefficients = nn.Dense(self.embedding_dim // 2, use_bias=False, kernel_init=nn.initializers.normal(stddev=1))
        self.concat = lambda x: jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)

        if self.learnable_coefficients:
            self.parsed_coefficients = lambda x: self.coefficients(self.pi * x)
        else:
            self.parsed_coefficients = lambda x: jax.lax.stop_gradient(self.coefficients(self.pi * x))

    def __call__(self, x):
        return self.concat(self.std * self.parsed_coefficients(x))


class EquivariantCrossAttention(nn.Module):
    num_hidden: int
    num_heads: int
    bi_invariant: BaseBiInvariant
    embedding_freq_multiplier: tuple

    def setup(self):
        # Bi-invariant embedding for the query and value transforms.
        emb_freq_mult_q, emb_freq_mult_v = self.embedding_freq_multiplier
        self.emb_q = RFFEmbedding(embedding_dim=self.num_hidden, learnable_coefficients=False, std=emb_freq_mult_q)
        self.emb_v = RFFEmbedding(embedding_dim=self.num_hidden, learnable_coefficients=False, std=emb_freq_mult_v)

        # Bi-invariant embedding -> query
        self.emb_to_q = nn.Dense(self.num_heads * self.num_hidden)

        # Context vector -> key, query
        self.c_to_kv = nn.Dense(2 * self.num_heads * self.num_hidden)

        # self.emb_to_v = PointwiseFFN(self.num_hidden, 2 * self.num_heads * self.num_hidden, layer_norm=False)
        # self.v_mixer = PointwiseFFN(self.num_hidden, self.num_hidden)
        self.emb_to_v = nn.Dense(2 * self.num_heads * self.num_hidden)
        self.v_mixer = nn.Dense(self.num_hidden)

        # Output projection
        self.out_proj = nn.Dense(self.num_heads * self.num_hidden)

        # Set the scale factor for the attention weights.
        self.scale = 1.0 / (self.num_hidden ** 0.5)

    def __call__(self, x, p, c, g):
        """ Apply equivariant cross attention.

        Args:
            x (jax.numpy.ndarray): The input coordinates. Shape (batch_size, num_coords, coord_dim).
            p (jax.numpy.ndarray): The latent poses. Shape (batch_size, num_latents, coord_dim).
            c (jax.numpy.ndarray): The latent context vectors. Shape (batch_size, num_latents, latent_dim).
            g (jax.numpy.ndarray): The window size for the gaussian window. Shape (batch_size, num_latents, 1).
        """
        # Get bi-invariants of input coordinates wrt latent coordinates. Depending on the bi-invariant, the shape of the
        # bi-invariants tensor will be different.
        bi_inv = self.bi_invariant(x, p)

        # Apply bi-invariant embedding for the query tranform and conditioning of the value transform.
        emb_q = self.emb_q(bi_inv)
        emb_v = self.emb_v(bi_inv)

        # Calculate the query, key and value.
        q = self.emb_to_q(emb_q)
        k, v = jnp.split(self.c_to_kv(c), 2, axis=-1)

        # Attend the values to the queries and keys.
        # Get gamma, beta conditioning variables for the value transform.
        cond_v_g, cond_v_b = jnp.split(self.emb_to_v(emb_v), 2, axis=-1)

        # Apply conditioning to the value transform, broadcast over the coordinates.
        v = v[:, None, :, :] * (1 + cond_v_g) + cond_v_b

        # Reshape to separate the heads, mix the values.
        v = self.v_mixer(v.reshape(v.shape[:-1] + (self.num_heads, self.num_hidden)))

        # Reshape the query, key and value to separate the heads.
        q = q.reshape(q.shape[:-1] + (self.num_heads, self.num_hidden))
        k = k.reshape(k.shape[:-1] + (self.num_heads, self.num_hidden))

        # For every input coordinate, calculate the attention weights for every latent.
        att = (q * k[:, None, ...]).sum(axis=-1) * self.scale  # 'bczhd,bzhd->bczh'

        # Apply gaussian window if needed.
        gaussian_window = self.bi_invariant.calculate_gaussian_window(x, p, sigma=g)
        att = att + gaussian_window
        att = softmax(att, axis=-2)

        # Apply attention to the values.
        y = (att[..., None] * v).sum(axis=2)  # 'bczh,bczhd->bchd'

        # Reshape y to concatenate the heads.
        y = y.reshape(*y.shape[:2], self.num_heads * self.num_hidden)

        # output projection
        y = self.out_proj(y)
        return y


class PointwiseFFN(nn.Module):
    num_hidden: int
    num_out: int
    num_layers: int = 1
    norm: bool = True

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(self.num_hidden)(x)
            x = gelu(x)
            if self.norm:
                x = nn.LayerNorm()(x)
        x = nn.Dense(self.num_out)(x)
        return x


class EquivariantNeuralField(nn.Module):
    """ Equivariant cross attention network for the latent points, conditioned on the poses.

    Args:
        num_hidden (int): The number of hidden units.
        num_heads (int): The number of attention heads.
        num_out (int): The number of output coordinates.
        latent_dim (int): The dimensionality of the latent code.
        bi_invariant (BaseBiInvariant): The invariant to use for the attention operation.
        embedding_freq_multiplier (Union[float, float]): The frequency multiplier for the embedding.
    """

    num_hidden: int
    num_heads: int
    num_out: int
    latent_dim: int
    bi_invariant: BaseBiInvariant
    embedding_freq_multiplier: Union[float, float]
    cross_attention_blocks = []

    def setup(self):

        # Maps latent to hidden space
        self.latent_stem = nn.Dense(self.num_hidden)

        # Cross attn block
        self.layer_norm_attn = nn.LayerNorm()
        self.attn = EquivariantCrossAttention(
            num_hidden=self.num_hidden,
            num_heads=self.num_heads,
            bi_invariant=self.bi_invariant,
            embedding_freq_multiplier=self.embedding_freq_multiplier
        )
        self.ffn_attn = PointwiseFFN(num_hidden=self.num_hidden, num_out=self.num_hidden)

        # Output ffn
        self.ffn_out = PointwiseFFN(num_hidden=self.num_hidden, num_out=self.num_out, num_layers=2, norm=False)

    def __call__(self, x, p, c, g):
        """ Sample from the model.

        Args:
            x (jnp.Array): The pose of the input points. Shape (batch_size, num_coords, 2).
            p (jnp.Array): The pose of the latent points. Shape (batch_size, num_latents, num_ori (1), 4).
            c (jnp.Arrays): The latent features. Shape (batch_size, num_latents, num_hidden).
            g (float or None): The window size for the gaussian window.
        """
        # Map code to latent space
        c = self.latent_stem(c)

        # Cross attention block, pre-norm on context vectors
        c = self.layer_norm_attn(c)
        f_hat = self.attn(x=x, p=p, c=c, g=g)
        f_hat = self.ffn_attn(f_hat)
        f_hat = jax.nn.gelu(f_hat)

        # Output layers
        return self.ffn_out(f_hat)
