import numpy as np 
import jax.numpy as jnp


class BaseBiInvariant:
    """Base class for bi-invariants. Bi-invariants are functions that are invariant to the simultaneous action of a
    group on the input and latent poses.

    The base class provides a common interface for all bi-invariants. Subclasses must implement the __call__ method.

    The following properties must be set in the subclass:
    - dim: The dimensionality of the bi-invariant.
    - num_x_pos_dims: The number of spatial dimensions for the input coordinates.
    - num_x_ori_dims: The number of orientation dimensions for the input orientations.
    - num_z_pos_dims: The number of spatial dimensions for the latent coordinates.
    - num_z_ori_dims: The number of orientation dimensions for the latent orientations.
    """

    def __init__(self):
        super().__init__()

        # Every invariant has a dimensionality.
        self.dim = None

        # Every invariant can have a different number of dimensions for the input coordinate and latent poses.
        self.num_x_pos_dims = None
        self.num_x_ori_dims = None
        self.num_z_pos_dims = None
        self.num_z_ori_dims = None

    def calculate_gaussian_window(self, x, p, sigma):
        """ Calculate the gaussian window in $R^n$.

        Args:
            x (jax.numpy.ndarray): The pose of the input coordinates. Shape (batch_size, num_coords, num_x_pos_dims).
            p (jax.numpy.ndarray): The pose of the latent points. Shape (batch_size, num_latents, num_z_pos_dims).
            sigma (jax.numpy.ndarray): The standard deviation of the gaussian window. Shape (batch_size, num_latents).

        Returns:
            jax.numpy.ndarray: The gaussian window value. Shape (batch_size, num_coords, num_latents, 1).
        """
        # Extract the positional coordinates.
        p_pos = p[:, :, :self.num_z_pos_dims]
        x_pos = x[:, :, :self.num_x_pos_dims]

        # Calculate squared norm distance between x and p
        norm_rel_dists = jnp.sum((p_pos[:, None, :, :] - x_pos[:, :, None, :]) ** 2, axis=-1, keepdims=True)

        # Calculate the gaussian window
        return - (1 / sigma[:, None, :] ** 2) * norm_rel_dists

    def __call__(self, x, p):
        """Calculate the bi-invariant between the input coordinates and the latent poses.

        Args:
            x (jax.numpy.ndarray): The pose of the input coordinates. Shape (batch_size, num_coords, num_x_pos_dims).
            p (jax.numpy.ndarray): The pose of the latent points. Shape (batch_size, num_latents, num_z_pos_dims).

        Returns:
            jax.numpy.ndarray: The bi-invariant between the input coordinates and the latent poses.
                Shape (batch_size, num_coords, num_latents, dim).
        """
        raise NotImplementedError("Subclasses must implement this method")
