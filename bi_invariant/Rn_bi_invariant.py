from enf.bi_invariant._base_bi_invariant import BaseBiInvariant


class RnBiInvariant(BaseBiInvariant):

    def __init__(self, num_dims: int):
        """ Calculate the relative position between two sets of coordinates in N dimensions.

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
        Calculate the relative position between two sets of coordinates in N dimensions.

        Args:
            x (jax.numpy.ndarray): The pose of the input coordinates. Shape (batch_size, num_coords, num_x_pos_dims).
            p (jax.numpy.ndarray): The pose of the latent points. Shape (batch_size, num_latents, num_z_pos_dims).

        Returns:
            jax.numpy.ndarray: The relative position between x and p.
                Shape (batch_size, num_coords, num_latents, num_x_pos_dims).
        """
        return x[:, :, None, :self.num_x_pos_dims] - p[:, None, :, :self.num_z_pos_dims]
