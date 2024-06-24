from enf.bi_invariant._base_bi_invariant import BaseBiInvariant


class AbsolutePositionND(BaseBiInvariant):

    def __init__(self, num_dims: int):
        """ This "bi-invariant" is bi-invariant to absolutely nothing. For when you want to break equivariance.

        Args:
            num_dims (int): The dimensionality of the coordinates, corresponds to the dimensionality of the translation
                group.
        """
        super().__init__()

        # Set the dimensionality of the invariant.
        self.dim = num_dims

        # This invariant is calculated based on two sets of positional coordinates, it doesn't depend on
        # the orientation.
        self.num_x_pos_dims = num_dims
        self.num_x_ori_dims = 0
        self.num_z_pos_dims = num_dims
        self.num_z_ori_dims = 0

    def __call__(self, x, p):
        """
        Returns x, the absolute position of the input coordinate. Another option would be to concatenate x and p, that
        might be more expressive.

        Args:
            x (jax.numpy.ndarray): The pose of the input coordinates. Shape (batch_size, num_coords, num_x_pos_dims).
            p (jax.numpy.ndarray): The pose of the latent points. Shape (batch_size, num_latents, num_z_pos_dims).

        Returns:
            jax.numpy.ndarray: The absolute position of the input.
                Shape (batch_size, num_coords, num_latents, num_x_pos_dims).
        """
        return x[:, :, None, :]
