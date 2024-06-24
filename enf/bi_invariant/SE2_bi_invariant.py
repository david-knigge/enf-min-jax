import jax.numpy as jnp

from enf.bi_invariant._base_bi_invariant import BaseBiInvariant


class SE2onR2BiInvariant(BaseBiInvariant):
    """Bi-invariant for SE(2) on R^2. This function is bi-invariant to SE(2) transformations on input data defined over
    R2 and latents defined over SE(2).

    Based on the invariants defined in Bekkers, et al. 2024 (https://arxiv.org/abs/2310.02970).
    """

    def __init__(self):
        super().__init__()

        # This invariant is calculated based on two sets of positional coordinates and orientations.
        self.num_x_pos_dims = 2  # This is always 2, the positional coordinates are 2D.
        self.num_x_ori_dims = 0
        self.num_z_pos_dims = 2  # This is always 2, the positional coordinates are 2D.
        self.num_z_ori_dims = 1

        # This invariant is 2D.
        self.dim = 2

    def __call__(self, x, p):
        """ Calculate the Ponita invariants between two sets of coordinates.
        Args:
            x (torch.Tensor): The pose of the input coordinates. Shape (batch_size, num_coords, 2).
            p (torch.Tensor): The pose of the latent points. Shape (batch_size, num_latents, num_ori, 2 (pos) + 1 (ori)).
        Returns:
            invariants (torch.Tensor): The Ponita invariants between the input and latent coordinates.
                Shape (batch_size, num_coords, num_latents, num_x_ori, num_z_ori, 2).
        """
        # Broadcast x over num_latents.
        x_pos = x[:, :, None, :]
        p_pos, p_angles = p[:, None, :, :self.num_z_pos_dims], p[:, None, :, self.num_z_pos_dims:]

        # Embed angle into circle
        p_ori = jnp.concatenate((jnp.cos(p_angles), jnp.sin(p_angles)), axis=-1)

        # Calculate relative positions between x and p.
        rel_pos = x_pos - p_pos

        # Calculate ponita invariants, shapes are [batch_size, num_coords, num_latents, num_x_ori, num_z_ori, 2].
        invariant1 = (rel_pos[..., 0] * p_ori[..., 0] + rel_pos[..., 1] * p_ori[..., 1])
        invariant2 = (-rel_pos[..., 0] * p_ori[..., 1] + rel_pos[..., 1] * p_ori[..., 0])
        invariants = jnp.stack([invariant1, invariant2], axis=-1)

        return invariants


class SE2onSE2BiInvariant(BaseBiInvariant):
    """Bi-invariant for SE(2) on SE(2). This function is bi-invariant to SE(2) transformations on input data defined over
    SE(2) and latents defined over SE(2). This is for when the input data x has orientation.

    Based on the invariants defined in Bekkers, et al. 2024 (https://arxiv.org/abs/2310.02970).
    """

    def __init__(self):
        super().__init__()

        # This invariant is calculated based on two sets of positional coordinates and orientations.
        self.num_x_pos_dims = 2  # This is always 2, the positional coordinates are 2D.
        self.num_x_ori_dims = 1  # Full ponita invariant is calculated if the input points have orientation.
        self.num_z_pos_dims = 2  # This is always 2, the positional coordinates are 2D.
        self.num_z_ori_dims = 1

        # This invariant is 3D
        self.dim = 3

    def __call__(self, x, p):
        """ Calculate the Ponita invariants between two sets of coordinates.
        Args:
            x (torch.Tensor): The pose of the input coordinates. Shape (batch_size, num_coords, num_ori, 2 (pos) + 2 (ori)).
            p (torch.Tensor): The pose of the latent points. Shape (batch_size, num_latents, num_ori, 2 (pos) + 2 (ori)).
        Returns:
            invariants (torch.Tensor): The Ponita invariants between the input and latent coordinates.
                Shape (batch_size, num_coords, num_latents, num_x_ori, num_z_ori, 2).
        """

        # Extract positional and orientation coordinates.
        x_pos, x_angles = x[:, :, None, :2], x[:, :, None, 2:]
        p_pos, p_angles = p[:, None, :, :2], p[:, None, :, 2:]

        # Embed angle into circle
        x_ori = jnp.concatenate((jnp.cos(x_angles), jnp.sin(x_angles)), axis=-1)
        p_ori = jnp.concatenate((jnp.cos(p_angles), jnp.sin(p_angles)), axis=-1)

        # Calculate relative positions between x and p.
        rel_pos = x_pos - p_pos

        # Calculate Ponita invariants, shapes are [batch_size, num_coords, num_latents, num_x_ori, num_z_ori, 3].
        invariant1 = (rel_pos[..., 0] * p_ori[..., 0] + rel_pos[..., 1] * p_ori[..., 1])
        invariant2 = (-rel_pos[..., 0] * p_ori[..., 1] + rel_pos[..., 1] * p_ori[..., 0])
        invariant3 = (x_ori * p_ori).sum(axis=-1)
        invariants = jnp.stack([invariant1, invariant2, invariant3], axis=-1)

        return invariants
