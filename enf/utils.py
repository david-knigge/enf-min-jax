import jax
import jax.numpy as jnp
import optax
import math
from typing import Tuple, Any, Type, Dict
from enf.bi_invariants import TranslationBI, RotoTranslationBI2D


def initialize_grid_positions(batch_size: int, num_latents: int, data_dim: int) -> jnp.ndarray:
    """Initialize a grid of positions in N dimensions."""
    # Calculate grid size per dimension
    grid_size = int(math.ceil(num_latents ** (1/data_dim)))
    lims = 1 - 1 / grid_size
    
    # Create linspace for each dimension
    linspaces = [jnp.linspace(-lims, lims, grid_size) for _ in range(data_dim)]
    
    # Create meshgrid
    grid = jnp.stack(jnp.meshgrid(*linspaces, indexing='ij'), axis=-1)
    
    # Reshape and repeat for batch
    positions = jnp.reshape(grid, (1, -1, data_dim))
    positions = positions.repeat(batch_size, axis=0)
    
    # If we have more points than needed, truncate
    if positions.shape[1] > num_latents:
        positions = positions[:, :num_latents, :]
    
    return positions


def initialize_latents(
    batch_size: int,
    num_latents: int,
    latent_dim: int,
    data_dim: int,
    bi_invariant_cls: Type,
    key: Any,
    window_scale: float = 2.0,
    noise_scale: float = 0.1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Initialize the latent variables based on the bi-invariant type."""
    key, subkey = jax.random.split(key)

    if bi_invariant_cls == TranslationBI:
        # For translation-only, positions are same dimension as data
        pose = initialize_grid_positions(batch_size, num_latents, data_dim)
        
    elif bi_invariant_cls == RotoTranslationBI2D:
        if data_dim != 2:
            raise ValueError("RotoTranslationBI2D requires 2D data")
        
        # Initialize positions in 2D
        positions_2d = initialize_grid_positions(batch_size, num_latents, 2)
        
        # Add orientation angle theta
        key, subkey = jax.random.split(key)
        theta = jax.random.uniform(subkey, (batch_size, num_latents, 1)) * 2 * jnp.pi
        
        # Concatenate positions and theta
        pose = jnp.concatenate([positions_2d, theta], axis=-1)
        
    else:
        raise ValueError(f"Unsupported bi-invariant type: {bi_invariant_cls}")

    # Add random noise to positions
    pose = pose + jax.random.normal(subkey, pose.shape) * noise_scale / jnp.sqrt(num_latents)

    # Initialize context vectors and gaussian window
    context = jnp.ones((batch_size, num_latents, latent_dim)) / latent_dim
    window = jnp.ones((batch_size, num_latents, 1)) * window_scale / jnp.sqrt(num_latents)
    
    return pose, context, window


def create_coordinate_grid(img_shape: Tuple[int, ...], batch_size: int) -> jnp.ndarray:
    """Create a coordinate grid for the input space."""
    x = jnp.stack(jnp.meshgrid(
        jnp.linspace(-1, 1, img_shape[0]),
        jnp.linspace(-1, 1, img_shape[1])), axis=-1)
    x = jnp.reshape(x, (1, -1, 2)).repeat(batch_size, axis=0)
    return x
