import ml_collections
from ml_collections import config_flags
from absl import app

import jax
import jax.numpy as jnp
import optax
import logging

import matplotlib.pyplot as plt

import wandb

# Custom imports
from datasets import get_dataloaders
from enf.model import EquivariantNeuralField
from enf.bi_invariants import TranslationBI
from enf.utils import create_coordinate_grid, initialize_latents


def get_config():

    # Define config
    config = ml_collections.ConfigDict()
    config.seed = 68
    config.debug = False

    # Reconstruction model
    config.recon_enf = ml_collections.ConfigDict()
    config.recon_enf.num_hidden = 256
    config.recon_enf.num_heads = 3
    config.recon_enf.att_dim = 256
    config.recon_enf.num_in = 2  # Images are 2D
    config.recon_enf.num_out = 8  # 8 channels in the image (before/after for S1 (gray) and S2 (RGB))
    config.recon_enf.freq_mult = (15.0, 30.0)
    config.recon_enf.k_nearest = 4

    config.recon_enf.num_latents = 256
    config.recon_enf.latent_dim = 256

    # Segmentation model
    config.seg_enf = ml_collections.ConfigDict()
    config.seg_enf.num_hidden = 128
    config.seg_enf.num_heads = 3
    config.seg_enf.att_dim = 128
    config.seg_enf.num_in = 2  # Images are 2D
    config.seg_enf.num_out = 1  # Mask is single channel
    config.seg_enf.freq_mult = (2.0, 10.0)
    config.seg_enf.k_nearest = 4

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 0

    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_enf = 5e-4
    config.optim.inner_lr = (0., 60., 0.) # (pose, context, window)
    config.optim.inner_steps = 3
    config.optim.first_order_maml = True

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 8
    config.train.noise_scale = 1e-1  # Noise added to latents to prevent overfitting
    config.train.num_epochs_pretrain = 100
    config.train.num_epochs_train_seg = 100
    config.train.log_interval = 200
    logging.getLogger().setLevel(logging.INFO)

    # Set checkpoint path
    config.run_name = "enf"
    return config


# Set config flags
_CONFIG = config_flags.DEFINE_config_dict("config", get_config())


def plot_ombria_comparison(
    original: jnp.ndarray, 
    reconstruction: jnp.ndarray,
    poses: jnp.ndarray = None,
    mask: jnp.ndarray = None, 
    pred_mask: jnp.ndarray = None, 
    save_path: str = None
):
    """Plot original and reconstructed Ombria images side by side.
    
    Args:
        original: Original images with shape (H, W, 8)
        reconstruction: Reconstructed images with shape (H, W, 8)
        mask: Optional ground truth flood mask with shape (H, W)
        pred_mask: Optional predicted flood mask with shape (H, W)
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(3, 6, figsize=(16, 8))
    fig.suptitle('Original (top) vs Reconstruction (bottom)')
    
    # Clip to prevent warnings
    original = jnp.clip(original, 0, 1)
    reconstruction = jnp.clip(reconstruction, 0, 1)

    # Plot originals
    # S1 (IR) images are channels 0 and 4 (before/after)
    axes[0,0].imshow(original[..., 0], cmap='gray')
    axes[0,0].set_title('S1 Before (IR)')
    axes[0,1].imshow(original[..., 4], cmap='gray')
    axes[0,1].set_title('S1 After (IR)')
    
    # S2 (RGB) images are channels 1:4 and 5:8 (before/after)
    axes[0,2].imshow(original[..., [1,2,3]])
    axes[0,2].set_title('S2 Before (RGB)')
    axes[0,3].imshow(original[..., [5,6,7]])
    axes[0,3].set_title('S2 After (RGB)')
    
    # Plot poses
    if poses is not None:
        # Map to 0-W range
        poses = (poses + 1) * original.shape[0] / 2
        axes[1,4].scatter(poses[:, 0], poses[:, 1], c='r', s=10)
        axes[1,4].set_title('Pose')

    # Plot masks
    if mask is not None:
        axes[0,5].imshow(mask, cmap='gray')
        axes[0,5].set_title('Ground Truth Mask')
    else:
        axes[0,5].axis('off')
    
    if pred_mask is not None:
        axes[1,5].imshow(pred_mask > 0.5, cmap='gray')
        axes[1,5].set_title('Predicted Mask')
    else:
        axes[1,5].axis('off')
    
    # Plot reconstructions
    axes[1,0].imshow(reconstruction[..., 0], cmap='gray')
    axes[1,1].imshow(reconstruction[..., 4], cmap='gray')
    axes[1,2].imshow(reconstruction[..., [1,2,3]])
    axes[1,3].imshow(reconstruction[..., [5,6,7]])

    # Remove axes
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        return fig


def main(_):

    # Get config
    config = _CONFIG.value

    # Initialize wandb
    run = wandb.init(project="enf-min", config=config.to_dict(), mode="online" if not config.debug else "dryrun")

    # Load dataset, get sample image, create corresponding coordinates
    train_dloader, test_dloader = get_dataloaders('ombria', config.train.batch_size, config.dataset.num_workers)
    sample_img, _ = next(iter(train_dloader))
    img_shape = sample_img.shape[1:]

    # Random key
    key = jax.random.PRNGKey(55)

    # Create coordinate grid for this dataset
    x = create_coordinate_grid(batch_size=config.train.batch_size, img_shape=img_shape)

    # Define the reconstruction and segmentation models
    recon_enf = EquivariantNeuralField(
        num_hidden=config.recon_enf.num_hidden,
        att_dim=config.recon_enf.att_dim,
        num_heads=config.recon_enf.num_heads,
        num_out=config.recon_enf.num_out,
        emb_freq=config.recon_enf.freq_mult,
        nearest_k=config.recon_enf.k_nearest,
        bi_invariant=TranslationBI(),
    )
    seg_enf = EquivariantNeuralField(
        num_hidden=config.seg_enf.num_hidden,
        att_dim=config.seg_enf.att_dim,
        num_heads=config.seg_enf.num_heads,
        num_out=config.seg_enf.num_out,
        emb_freq=config.seg_enf.freq_mult,
        nearest_k=config.seg_enf.k_nearest,
        bi_invariant=TranslationBI(),
    )

    # Create dummy latents for model init
    key, subkey = jax.random.split(key)
    temp_z = initialize_latents(
        batch_size=1,  # Only need one example for initialization
        num_latents=config.recon_enf.num_latents,
        latent_dim=config.recon_enf.latent_dim,
        data_dim=config.recon_enf.num_in,
        bi_invariant_cls=TranslationBI,
        key=subkey,
        noise_scale=config.train.noise_scale,
    )

    # Init the model
    recon_enf_params = recon_enf.init(key, x, *temp_z)
    seg_enf_params = seg_enf.init(key, x, *temp_z)

    # Define optimizer for the ENF backbone
    enf_opt = optax.adam(learning_rate=config.optim.lr_enf)
    recon_enf_opt_state = enf_opt.init(recon_enf_params)
    seg_enf_opt_state = enf_opt.init(seg_enf_params)

    @jax.jit
    def recon_inner_loop(enf_params, coords, sat_img, key):
        z = initialize_latents(
            batch_size=config.train.batch_size,
            num_latents=config.recon_enf.num_latents,
            latent_dim=config.recon_enf.latent_dim,
            data_dim=config.recon_enf.num_in,
            bi_invariant_cls=TranslationBI,
            key=key,
            noise_scale=config.train.noise_scale,
        )

        def mse_loss(z):
            out = recon_enf.apply(enf_params, coords, *z)
            return jnp.sum(jnp.mean((out - sat_img) ** 2, axis=(1, 2)), axis=0)

        def inner_step(z, _):
            _, grads = jax.value_and_grad(mse_loss)(z)
            # Gradient descent update
            z = jax.tree.map(lambda z, grad, lr: z - lr * grad, z, grads, config.optim.inner_lr)
            return z, None
        
        # Perform inner loop optimization
        z, _ = jax.lax.scan(inner_step, z, None, length=config.optim.inner_steps)

        # Stop gradient if first order MAML
        if config.optim.first_order_maml:
            z = jax.lax.stop_gradient(z)
        return mse_loss(z), z

    @jax.jit
    def recon_outer_step(coords, sat_img, enf_params, enf_opt_state, key):
        # Perform inner loop optimization
        key, subkey = jax.random.split(key)
        (loss, z), grads = jax.value_and_grad(recon_inner_loop, has_aux=True)(enf_params, coords, sat_img, key)

        # Update the ENF backbone
        enf_grads, enf_opt_state = enf_opt.update(grads, enf_opt_state)
        enf_params = optax.apply_updates(enf_params, enf_grads)

        # Sample new key
        return (loss, z), enf_params, enf_opt_state, subkey
    
    @jax.jit
    def segment_outer_step(coords, sat_img, gt_mask, recon_enf_params, seg_enf_params, seg_enf_opt_state, key):        
        # Perform inner loop optimization to obtain latent
        key, subkey = jax.random.split(key)
        loss, z = recon_inner_loop(recon_enf_params, coords, sat_img, key)
        
        # Calculate the segmentation loss
        def mse_loss(seg_enf_params, z, coords, gt_mask):
            pred_mask = seg_enf.apply(seg_enf_params, coords, *z)
            return jnp.sum(jnp.mean((pred_mask - gt_mask) ** 2, axis=(1, 2)), axis=0)

        # Update the segmentation ENF
        loss, grads = jax.value_and_grad(mse_loss)(seg_enf_params, z, coords, gt_mask)
        enf_grads, seg_enf_opt_state = enf_opt.update(grads, seg_enf_opt_state)
        seg_enf_params = optax.apply_updates(seg_enf_params, enf_grads)

        # Sample new key
        return (loss, z), seg_enf_params, seg_enf_opt_state, subkey

    # Pretraining loop for fitting the ENF backbone
    glob_step = 0
    for epoch in range(config.train.num_epochs_pretrain):
        epoch_loss = []
        for i, (sat_img, mask) in enumerate(train_dloader):
            y = jnp.reshape(sat_img, (sat_img.shape[0], -1, sat_img.shape[-1]))

            # Perform outer loop optimization
            (loss, z), recon_enf_params, recon_enf_opt_state, key = recon_outer_step(
                x, y, recon_enf_params, recon_enf_opt_state, key)

            epoch_loss.append(loss)
            glob_step += 1

            if glob_step % config.train.log_interval == 0:
                # Reconstruct and plot the first image in the batch
                sat_img_r = recon_enf.apply(recon_enf_params, x, *z).reshape(sat_img.shape)
                fig = plot_ombria_comparison(sat_img[0], sat_img_r[0], poses=z[0][0], mask=mask[0])
                wandb.log({"recon-mse": sum(epoch_loss) / len(epoch_loss), "reconstruction": fig})
                plt.close('all')
                logging.info(f"RECON ep {epoch} / step {glob_step} || mse: {sum(epoch_loss) / len(epoch_loss)}")

    # Training loop for the segmentation ENF
    for epoch in range(config.train.num_epochs_train_seg):
        epoch_loss = []
        for i, (sat_img, mask) in enumerate(train_dloader):
            y = jnp.reshape(sat_img, (sat_img.shape[0], -1, sat_img.shape[-1]))
            gt_mask = jnp.reshape(mask, (mask.shape[0], -1, mask.shape[-1]))

            # Perform outer loop optimization
            (loss, z), seg_enf_params, seg_enf_opt_state, key = segment_outer_step(
                x, y, gt_mask, recon_enf_params, seg_enf_params, seg_enf_opt_state, key)

            epoch_loss.append(loss)
            glob_step += 1

            if glob_step % config.train.log_interval == 0:
                # Reconstruct and plot the first image in the batch
                sat_img_r = recon_enf.apply(recon_enf_params, x, *z).reshape(sat_img.shape)
                pred_mask = seg_enf.apply(seg_enf_params, x, *z).reshape(mask.shape)
                fig = plot_ombria_comparison(sat_img[0], sat_img_r[0], poses=z[0][0], mask=mask[0], pred_mask=pred_mask[0])
                wandb.log({"seg-mse": sum(epoch_loss) / len(epoch_loss), "reconstruction": fig})
                plt.close('all')
                logging.info(f"SEG ep {epoch} / step {glob_step} || mse: {sum(epoch_loss) / len(epoch_loss)}")

    run.finish()


if __name__ == "__main__":
    app.run(main)
