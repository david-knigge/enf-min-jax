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
from experiments.datasets import get_dataloaders
from enf.model import EquivariantNeuralField
from enf.bi_invariants import TranslationBI
from enf.utils import create_coordinate_grid, initialize_latents

from experiments.downstream_models.transformer_enf import TransformerClassifier


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
    config.train.batch_size = 32
    config.train.noise_scale = 1e-1  # Noise added to latents to prevent overfitting
    config.train.num_epochs_pretrain = 10
    config.train.num_epochs_train_cls = 10
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
    transformer = TransformerClassifier(
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_classes=10,
    )
    transformer_params = transformer.init(key, *temp_z)

    # Define optimizer for the transformer model
    transformer_opt = optax.adam(learning_rate=config.optim.lr_transformer)
    transformer_opt_state = transformer_opt.init(transformer_params)

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
    def classifier_outer_step(coords, img, labels, transformer_params, transformer_opt_state, key):
        # Perform inner loop optimization to get latents
        key, subkey = jax.random.split(key)
        _, z = recon_inner_loop(recon_enf_params, coords, img, key)

        # Normalize context vectors
        p, c, g = z
        c = (c - c_mean) / c_std
        z = (p, c, g)

        def cross_entropy_loss(params):
            # Get transformer predictions from latents
            logits = transformer.apply(params, *z)
            # Compute cross entropy loss
            one_hot = jax.nn.one_hot(labels, num_classes=10)
            loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1)
            return jnp.mean(loss)

        # Get gradients
        loss, grads = jax.value_and_grad(cross_entropy_loss)(transformer_params)

        # Update transformer parameters
        updates, transformer_opt_state = transformer_opt.update(grads, transformer_opt_state)
        transformer_params = optax.apply_updates(transformer_params, updates)

        return (loss, z), transformer_params, transformer_opt_state, subkey

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
                wandb.log({"recon-mse": sum(epoch_loss) / len(epoch_loss), "reconstruction": fig}, step=glob_step)
                plt.close('all')
                logging.info(f"RECON ep {epoch} / step {glob_step} || mse: {sum(epoch_loss[-10:]) / len(epoch_loss[-10:])}")

    # Training the classifier is much faster when the context vectors are normalized.
    # We determine mean and std of the context vectors and normalize them.
    c_list = []
    for i, (img, _) in enumerate(train_dloader):
        # Do inner loop, get latents
        y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))
        _, z = recon_inner_loop(recon_enf_params, x, y, key)

        # Append context vectors to list
        c_list.append(z[1])

    # Compute mean and std of the context vectors
    c_mean = jnp.mean(jnp.concatenate(c_list, axis=0), axis=(0, 1))
    c_std = jnp.std(jnp.concatenate(c_list, axis=0), axis=(0, 1))

    # Training loop for the transformer classifier
    for epoch in range(config.train.num_epochs_train_cls):
        epoch_loss = []
        for i, (img, labels) in enumerate(train_dloader):
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))

            # Optimize the transformer classifier
            (loss, z), transformer_params, transformer_opt_state, key = classifier_outer_step(
                x, y, labels, transformer_params, transformer_opt_state, key)

            epoch_loss.append(loss)
            glob_step += 1

            if glob_step % config.train.log_interval == 0:
                # Get predictions on a batch
                logits = transformer.apply(transformer_params, *z)
                preds = jnp.argmax(logits, axis=-1)
                accuracy = jnp.mean(preds == labels)

                wandb.log({
                    "classifier-loss": sum(epoch_loss) / len(epoch_loss),
                    "accuracy": accuracy
                }, step=glob_step)

                logging.info(f"CLASSIFIER ep {epoch} / step {glob_step} || " 
                           f"loss: {sum(epoch_loss) / len(epoch_loss):.4f}, "
                           f"accuracy: {accuracy:.4f}")

    # Test loop
    accs = []
    for i, (img, labels) in enumerate(test_dloader):
        y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))
        _, z = recon_inner_loop(recon_enf_params, x, y, key)
        # Normalize context vectors
        p, c, g = z
        c = (c - c_mean) / c_std
        z = (p, c, g)
        logits = transformer.apply(transformer_params, *z)
        preds = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(preds == labels)
        accs.append(accuracy)
    logging.info(f"TEST accuracy: {sum(accs) / len(accs):.4f}")
    run.finish()


if __name__ == "__main__":
    app.run(main)
