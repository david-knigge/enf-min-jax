import jax.numpy as jnp


class TranslationBI:   
    def __call__(self, x: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        """Compute the translation equivariant bi-invariant for ND data.
        
        Args:
            x: The input data. Shape (batch_size, num_coords, coord_dim).
            p: The latent poses. Shape (batch_size, num_latents, coord_dim).
        """
        return x[:, :, None, :] - p[:, None, :, :]


class RotoTranslationBI2D:
    def __call__(self, x: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        """ Compute roto-translation bi-invariant for 2D data. This assumes the input coordinates are
        2D, i.e. do not include the orientation. The latents include orientation angle theta.
        
        Args:
            x: The input data. Shape (batch_size, num_coords, pos_dim (2)).
            p: The latent poses. Shape (batch_size, num_latents, pos_dim (2) + theta (1)).
        """
        # Compute the relative position between the input and the latent poses
        rel_pos = x[:, :, None, :2] - p[:, None, :, :2]

        # Get the orientation angle theta and convert to [cos(θ), sin(θ)]
        theta = p[:, None, :, 2:]
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)

        # Compute the relative orientation between the input and the latent poses
        invariant1 = rel_pos[..., 0] * cos_theta + rel_pos[..., 1] * sin_theta
        invariant2 = -rel_pos[..., 0] * sin_theta + rel_pos[..., 1] * cos_theta
        return jnp.stack([invariant1, invariant2], axis=-1)
