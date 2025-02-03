import ml_collections
from ml_collections import config_flags
from absl import app
from typing import List
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
from enf.utils import create_coordinate_grid, initialize_latents, initialize_latents_normal

from downstream.diffusion_transformer_enf import DiT


def get_config():

    # Define config
    config = ml_collections.ConfigDict()
    config.seed = 68
    config.debug = False

    # Reconstruction model
    config.recon_enf = ml_collections.ConfigDict()
    config.recon_enf.num_hidden = 128
    config.recon_enf.num_heads = 3
    config.recon_enf.att_dim = 64
    config.recon_enf.num_in = 2  # Images are 2D
    config.recon_enf.num_out = 3  # 3 channels
    config.recon_enf.freq_mult = (3.0, 5.0)
    config.recon_enf.k_nearest = 4

    config.recon_enf.num_latents = 16
    config.recon_enf.latent_dim = 64

    # Diffusion model config
    config.diffusion = ml_collections.ConfigDict()
    config.diffusion.use_cfg = True
    config.diffusion.cfg_val = 1.0
    config.diffusion.t_sampler = 'log-normal'
    config.diffusion.denoise_timesteps = 100
    config.diffusion.num_classes = 10

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 0

    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_enf = 5e-4
    config.optim.lr_transformer = 1e-4
    config.optim.inner_lr = (2., 30., 0.) # (pose, context, window)
    config.optim.inner_steps = 3
    config.optim.first_order_maml = False

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 64
    config.train.noise_scale = 1e-1  # Noise added to latents to prevent overfitting
    config.train.num_epochs_pretrain = 2
    config.train.num_epochs_train_diff = 30
    config.train.log_interval = 200
    logging.getLogger().setLevel(logging.INFO)

    # Set checkpoint path
    config.run_name = "enf"
    return config


# Set config flags
_CONFIG = config_flags.DEFINE_config_dict("config", get_config())


def plot_cifar_comparison(
    original: jnp.ndarray,
    reconstruction: jnp.ndarray,
    poses: jnp.ndarray = None,
):
    """Plot original and reconstructed CIFAR images side by side.
    
    Args:
        original: Original images with shape (H, W, 3)
        reconstruction: Reconstructed images with shape (H, W, 3)
        poses: Optional poses to plot on the image
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(6, 2))
    fig.suptitle('Original (top) vs Reconstruction (bottom)')
    
    # Clip to prevent warnings
    original = jnp.clip(original, 0, 1)
    reconstruction = jnp.clip(reconstruction, 0, 1)

    # Plot original
    axes[0].imshow(original)
    axes[0].set_title('Original')

    # Plot reconstructed
    axes[1].imshow(reconstruction)
    axes[1].set_title('Reconstruction')

    # Plot poses
    if poses is not None:
        # Map to 0-W range
        poses = (poses + 1) * original.shape[0] / 2
        axes[2].imshow(reconstruction)
        axes[2].scatter(poses[:, 0], poses[:, 1], c='r', s=2)
        axes[2].set_title('Poses')

    # Remove axes
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_cifar_generations(generations: List[jnp.ndarray]):
    """Plot CIFAR generations."""
    # Clip
    generations = jnp.clip(generations, 0, 1)
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    for ax, img in zip(axes.flat, generations):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    return fig


# Diffusion model stuff
def get_z_t(z, eps, t):
    z_0 = eps
    z_1 = z
    t = jnp.clip(t, 0, 1-0.01)
    return (1-t) * z_0 + t * z_1


def get_v(z, eps):
    z_0 = eps
    z_1 = z
    return z_1 - z_0


def main(_):

    # Get config
    config = _CONFIG.value

    # Initialize wandb
    run = wandb.init(project="enf-min", config=config.to_dict(), mode="online" if not config.debug else "dryrun")

    # Load dataset, get sample image, create corresponding coordinates
    train_dloader, test_dloader = get_dataloaders('cifar10', config.train.batch_size, config.dataset.num_workers)
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

    # Define optimizer for the ENF backbone
    enf_opt = optax.adam(learning_rate=config.optim.lr_enf)
    recon_enf_opt_state = enf_opt.init(recon_enf_params)

    # Define the transformer model
    diffusion_model = DiT(
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_classes=10,
        class_dropout_prob=0.1,
    )
    temp_t = jnp.ones((config.train.batch_size,))
    temp_labels = jnp.ones((config.train.batch_size,), dtype=jnp.int32)
    diffusion_model_params = diffusion_model.init(key, temp_z, temp_t, temp_labels)
    diffusion_model_params_ema = diffusion_model_params.copy()

    # Define optimizer for the transformer model
    diffusion_model_opt = optax.adam(learning_rate=config.optim.lr_transformer)
    diffusion_model_opt_state = diffusion_model_opt.init(diffusion_model_params)

    @jax.jit
    def recon_inner_loop(enf_params, coords, img, key):
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
            return jnp.sum(jnp.mean((out - img) ** 2, axis=(1, 2)), axis=0)

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
    def recon_outer_step(coords, img, enf_params, enf_opt_state, key):
        # Perform inner loop optimization
        key, subkey = jax.random.split(key)
        (loss, z), grads = jax.value_and_grad(recon_inner_loop, has_aux=True)(enf_params, coords, img, key)

        # Update the ENF backbone
        enf_grads, enf_opt_state = enf_opt.update(grads, enf_opt_state)
        enf_params = optax.apply_updates(enf_params, enf_grads)

        # Sample new key
        return (loss, z), enf_params, enf_opt_state, subkey

    @jax.jit
    def diffusion_train_step(coords, img, labels, diffusion_model_params, diffusion_model_params_ema, diffusion_model_opt_state, key):
        # Perform inner loop optimization to get latents
        key, label_key, time_key, noise_key = jax.random.split(key, 4)
        _, z = recon_inner_loop(recon_enf_params, coords, img, key)

        def diffusion_loss(diffusion_model_params, z, labels):
            # Normalize context vectors
            p, c, g = z
            c = (c - c_mean) / c_std

            # Sample a t for training, log-normal
            t = jax.random.normal(time_key, (c.shape[0],))
            t = ((1 / (1 + jnp.exp(-t))))

            # Setup c_t and v_t
            t_full = t[:, None, None]
            eps_c = jax.random.normal(noise_key, c.shape)
            c_t = get_z_t(c, eps_c, t_full)
            vc_t = get_v(c, eps_c)

            if not config.diffusion.use_cfg:
                labels = jnp.ones(labels.shape, dtype=jnp.int32) * config.diffusion.num_classes

            vc_prime = diffusion_model.apply(
                diffusion_model_params,
                (p, c_t, g),
                t,
                labels,
                train=True,
                rngs={'label_dropout': label_key},
            )
            loss = jnp.mean((vc_prime - vc_t) ** 2)
            
            return loss
        
        # Get gradients
        loss, grads = jax.value_and_grad(diffusion_loss)(diffusion_model_params, z, labels)

        # Update diffusion model parameters
        updates, diffusion_model_opt_state = diffusion_model_opt.update(grads, diffusion_model_opt_state)
        diffusion_model_params = optax.apply_updates(diffusion_model_params, updates)

        # Update exponential moving average
        diffusion_model_params_ema = jax.tree.map(
            lambda a, b: a * 0.999 + b * 0.001,
            diffusion_model_params_ema,
            diffusion_model_params,
        )

        return loss, diffusion_model_params, diffusion_model_params_ema, diffusion_model_opt_state, key

    @jax.jit
    def denoise_step(z_t, t, labels, diffusion_model_params_ema):
        # Denoise z_t
        p_t, c_t, g = z_t

        if not config.diffusion.use_cfg:
            labels = jnp.ones(labels.shape, dtype=jnp.int32) * config.diffusion.num_classes # Null token
            vc = diffusion_model.apply(diffusion_model_params_ema, (p_t, c_t, g), t, labels, train=False, force_drop_ids=False)
        else:
            labels_uncond = jnp.ones(labels.shape, dtype=jnp.int32) * config.diffusion.num_classes # Null token
            z_expanded = jax.tree.map(lambda x: jnp.tile(x, (2, 1, 1)), z_t)
            t_expanded = jnp.tile(t, (2,))
            labels_full = jnp.concatenate([labels, labels_uncond], axis=0)
            vc_pred = diffusion_model.apply(diffusion_model_params_ema, z_expanded, t_expanded, labels_full, train=False, force_drop_ids=False)
            vc_label = vc_pred[:z_t[0].shape[0]]
            vc_uncond = vc_pred[z_t[0].shape[0]:]
            vc = vc_uncond + config.diffusion.cfg_val * (vc_label - vc_uncond)
        return vc

    @jax.jit
    def sample_diffusion_process(labels, diffusion_model_params_ema, key):
        # Sample a batch of latents w/ normal distribution
        key, subkey = jax.random.split(key)
        z = initialize_latents_normal(
            batch_size=config.train.batch_size,
            num_latents=config.recon_enf.num_latents,
            latent_dim=config.recon_enf.latent_dim,
            data_dim=config.recon_enf.num_in,
            bi_invariant_cls=TranslationBI,
            key=subkey,
            noise_scale=config.train.noise_scale,
        )
        delta_t = 1.0 / config.diffusion.denoise_timesteps
        for ti in range(config.diffusion.denoise_timesteps):
            t = ti / config.diffusion.denoise_timesteps # From x_0 (noise) to x_1 (data)
            t_vector = jnp.full((z[0].shape[0],), t)
            vc = denoise_step(z, t_vector, labels, diffusion_model_params_ema)
            z = (z[0], z[1] + vc * delta_t, z[2])
        
        # Unnormalize latents
        z = (z[0], z[1] * c_std + c_mean, z[2])
        return z
    
    # Pretraining loop for fitting the ENF backbone
    glob_step = 0
    for epoch in range(config.train.num_epochs_pretrain):
        epoch_loss = []
        for i, (img, _) in enumerate(train_dloader):
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))

            # Perform outer loop optimization
            (loss, z), recon_enf_params, recon_enf_opt_state, key = recon_outer_step(
                x, y, recon_enf_params, recon_enf_opt_state, key)

            epoch_loss.append(loss)
            glob_step += 1

            if glob_step % config.train.log_interval == 0:
                # Reconstruct and plot the first image in the batch
                img_r = recon_enf.apply(recon_enf_params, x, *z).reshape(img.shape)
                fig = plot_cifar_comparison(img[0], img_r[0], poses=z[0][0])
                wandb.log({"reconstruction": fig}, step=glob_step, commit=False)
                plt.close('all')
                logging.info(f"RECON ep {epoch} / step {glob_step} || mse: {sum(epoch_loss[-10:]) / len(epoch_loss[-10:])}")
            wandb.log({"recon-mse": loss}, step=glob_step)

    # Compute mean and std of the context vectors, used for normalization
    c_list = []
    for i, (img, _) in enumerate(train_dloader):
        key, subkey = jax.random.split(key)
        # Do inner loop, get latents
        y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))
        _, z = recon_inner_loop(recon_enf_params, x, y, subkey)

        # Append context vectors to list
        c_list.append(z[1])

    # Compute mean and std of the context vectors
    c_mean = jnp.mean(jnp.concatenate(c_list, axis=0), axis=(0, 1))
    c_std = jnp.std(jnp.concatenate(c_list, axis=0), axis=(0, 1))

    # Training loop for the diffusion model
    for epoch in range(config.train.num_epochs_train_diff):
        epoch_loss = []
        for i, (img, labels) in enumerate(train_dloader):
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))

            # Optimize the diffusion model
            loss, diffusion_model_params, diffusion_model_params_ema, diffusion_model_opt_state, key = diffusion_train_step(
                x, y, labels, diffusion_model_params, diffusion_model_params_ema, diffusion_model_opt_state, key)

            epoch_loss.append(loss)
            glob_step += 1

            if glob_step % config.train.log_interval == 0:
                # Sample diffusion process
                z = sample_diffusion_process(labels, diffusion_model_params_ema, key)
                img_r = recon_enf.apply(recon_enf_params, x, *z).reshape(img.shape)
                fig = plot_cifar_generations(img_r)
                wandb.log({"diffusion-generations": fig}, step=glob_step, commit=False)
                plt.close('all')
                logging.info(f"DIFFUSION ep {epoch} / step {glob_step} || mse: {sum(epoch_loss) / len(epoch_loss)}")
            wandb.log({"diffusion-mse": loss}, step=glob_step)

if __name__ == "__main__":
    app.run(main)
