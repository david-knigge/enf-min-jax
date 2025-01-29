import jax
import jax.numpy as jnp
from flax import linen as nn

from typing import Tuple, Callable


class PosEmb(nn.Module):
    embedding_dim: int
    freq: float

    @nn.compact
    def __call__(self, coords):
        emb = nn.Dense(self.embedding_dim // 2, kernel_init=nn.initializers.normal(self.freq), use_bias=False)(
            jnp.pi * (coords + 1))
        return nn.Dense(self.embedding_dim)(jnp.sin(jnp.concatenate([coords, emb, emb + jnp.pi / 2.0], axis=-1)))


class EquivariantNeuralField(nn.Module):
    """ Equivariant cross attention network for the latent points, conditioned on the poses.

    Args:
        num_hidden (int): The number of hidden units.
        num_heads (int): The number of attention heads.
        num_out (int): The number of output coordinates.
        emb_freq (float): The frequency of the positional embedding.
        nearest_k (int): The number of nearest latents to consider.
        bi_invariant (Callable): The bi-invariant function to use.
    """
    num_hidden: int
    att_dim: int
    num_heads: int
    num_out: int
    emb_freq: Tuple[float, float]
    nearest_k: int
    bi_invariant: Callable

    def setup(self):
        # Positional embedding that takes in relative positions.
        self.pos_emb_q = nn.Sequential([PosEmb(self.num_hidden, self.emb_freq[0]), nn.Dense(self.num_heads * self.att_dim)])
        self.pos_emb_v = nn.Sequential([PosEmb(self.num_hidden, self.emb_freq[1]), nn.Dense(2 * self.num_hidden)])

        # Query, key, value functions.
        self.W_k = nn.Dense(self.num_heads * self.att_dim)
        self.W_v = nn.Dense(self.num_hidden)

        # Value bi-linear conditioning function.
        self.v = nn.Sequential([
            nn.Dense(self.num_hidden),
            nn.gelu,
            nn.Dense(self.num_heads * self.num_hidden)])

        # Output layer.
        self.W_out = nn.Dense(self.num_out)

    def __call__(self, x, p, c, g):
        """ Apply equivariant cross attention.

        Args:
            x (jax.numpy.ndarray): The input coordinates. Shape (batch_size, num_coords, coord_dim).
            p (jax.numpy.ndarray): The latent poses. Shape (batch_size, num_latents, coord_dim).
            c (jax.numpy.ndarray): The latent context vectors. Shape (batch_size, num_latents, latent_dim).
            g (jax.numpy.ndarray): The window size for the gaussian window. Shape (batch_size, num_latents, 1).
        """
        # Calculate bi-invariants between input coordinates and latents
        bi_inv = self.bi_invariant(x, p)
        
        # Calculate distances based on bi-invariant magnitude
        zx_mag = jnp.sum(bi_inv ** 2, axis=-1)
        nearest_z = jnp.argsort(zx_mag, axis=-1)[:, :, :self.nearest_k, None]

        # Restrict the bi-invariants and context vectors to the nearest latents
        zx_mag = jnp.take_along_axis(zx_mag[..., None], nearest_z, axis=2)
        bi_inv = jnp.take_along_axis(bi_inv, nearest_z, axis=2)
        k = jnp.take_along_axis(self.W_k(c)[:, None, :, :], nearest_z, axis=2)
        v = jnp.take_along_axis(self.W_v(c)[:, None, :, :], nearest_z, axis=2)
        g = jnp.take_along_axis(g[:, None, :, :], nearest_z, axis=2)

        # Apply bi-invariant embedding for the query transform and conditioning of the value transform
        q = self.pos_emb_q(bi_inv)
        b_v, g_v = jnp.split(self.pos_emb_v(bi_inv), 2, axis=-1)
        v = self.v(v * (1 + b_v) + g_v)

        # Reshape to separate the heads
        q = q.reshape(q.shape[:-1] + (self.num_heads, -1))
        k = k.reshape(k.shape[:-1] + (self.num_heads, -1))
        v = v.reshape(v.shape[:-1] + (self.num_heads, -1))

        # Calculate the attention weights, apply gaussian mask based on bi-invariant magnitude, broadcasting over heads.
        att_logits = (q * k).sum(axis=-1, keepdims=True) - ((1 / g ** 2) * zx_mag)[..., None, :]
        att = jax.nn.softmax(att_logits, axis=2)

        # Attend the values to the queries and keys.
        y = (att * v).sum(axis=2)

        # Combine the heads and apply the output layer.
        y = y.reshape(y.shape[:-2] + (-1,))
        return self.W_out(y)
