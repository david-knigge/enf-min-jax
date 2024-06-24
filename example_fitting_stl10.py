import ml_collections

import jax
import jax.numpy as jnp
import optax
import logging

import matplotlib.pyplot as plt

# Custom modules
from enf.bi_invariant.Rn_bi_invariant import RnBiInvariant
from enf.enf import EquivariantNeuralField

# Custom datasets
from datasets import get_dataloader


def main():

    # Set seed
    seed = 68

    # Define config
    config = ml_collections.ConfigDict()
    config.nef = ml_collections.ConfigDict()

    # Define the model
    config.nef.num_hidden = 64
    config.nef.num_heads = 3
    config.nef.num_in = 2  # Images are 2D
    config.nef.num_out = 3  # RGB images = 3 channels, grayscale = 1

    config.nef.num_latents = 64
    config.nef.latent_dim = 64

    config.nef.emb_freq_mult_q = 0.6
    config.nef.emb_freq_mult_v = 4.0

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.name = "stl10"
    config.dataset.path = "./data"
    config.dataset.num_signals_train = 1000
    config.dataset.num_signals_test = 100
    config.dataset.batch_size = 2
    config.dataset.num_workers = 0

    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_enf = 1e-4
    config.optim.lr_meta_sgd = 1e-3
    config.optim.init_inner_lr_p = 3.0
    config.optim.init_inner_lr_a = 30.0
    config.optim.inner_steps = 3
    config.optim.num_subsampled_points = 3096

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.num_epochs = 300
    config.train.log_interval = 100
    logging.getLogger().setLevel(logging.INFO)

    ##############################
    # Initializing the model
    ##############################

    # Load dataset, get sample image, create corresponding coordinates
    train_dloader, test_dloader = get_dataloader(config.dataset)
    sample_img, _ = next(iter(train_dloader))
    img_shape = sample_img.shape[1:]

    # Create coordinate grid
    x = jnp.stack(jnp.meshgrid(jnp.linspace(-1, 1, sample_img.shape[1]), jnp.linspace(-1, 1, sample_img.shape[2])), axis=-1)
    x = jnp.reshape(x, (-1, 2))
    x = jnp.repeat(x[None, ...], sample_img.shape[0], axis=0)

    # Define the model
    model = EquivariantNeuralField(
        num_hidden=config.nef.num_hidden,
        num_heads=config.nef.num_heads,
        num_out=config.nef.num_out,
        latent_dim=config.nef.latent_dim,
        bi_invariant=RnBiInvariant(2),
        embedding_freq_multiplier=[config.nef.emb_freq_mult_q, config.nef.emb_freq_mult_v],
    )

    # Create dummy latents for model init
    d_p = jnp.ones((config.dataset.batch_size, config.nef.num_latents, 2))  # poses
    d_c = jnp.ones((config.dataset.batch_size, config.nef.num_latents, config.nef.latent_dim))  # context vectors
    d_g = jnp.ones((config.dataset.batch_size, config.nef.num_latents, 1))  # gaussian window parameter

    # Init the model
    enf_params = model.init(jax.random.PRNGKey(0), x[:, :config.optim.num_subsampled_points], d_p, d_c, d_g)

    # Define optimizer for the ENF backbone
    enf_optimizer = optax.adam(learning_rate=config.optim.lr_enf)
    enf_opt_state = enf_optimizer.init(enf_params)

    # Define optimizer for meta SGD
    meta_sgd_params = [config.optim.init_inner_lr_p, jnp.ones(config.nef.latent_dim) * config.optim.init_inner_lr_a]
    meta_sgd_optimizer = optax.adam(learning_rate=config.optim.lr_meta_sgd)
    meta_sgd_opt_state = meta_sgd_optimizer.init(meta_sgd_params)

    ##############################
    # Training logic
    ##############################
    @jax.jit
    def inner_loop(params, x_i, y_i, key):
        # Unpack params
        enf_params, meta_sgd_params = params

        # Initialize values for the poses, note that these depend on the bi-invariant, context and window
        p = jax.random.uniform(key, (x_i[0].shape[0], config.nef.num_latents, 2)) * 2 - 1  # poses, uniform [-1, 1]
        c = jnp.ones((x_i[0].shape[0], config.nef.num_latents, config.nef.latent_dim))  # context vectors
        g = jnp.ones((x_i[0].shape[0], config.nef.num_latents, 1)) * 2 / jnp.sqrt(config.nef.num_latents)  # gaussian window parameter

        def mse_loss(z, x_i, y_i):
            out = model.apply(enf_params, x_i, *z)
            return jnp.sum(jnp.mean((out - y_i) ** 2, axis=(1, 2)), axis=0)

        for i in range(config.optim.inner_steps):
            loss, grads = jax.value_and_grad(mse_loss)((p, c, g), x_i[i], y_i[i])

            # Update the latents, scale gradients by number of points and number of latents
            p = p - meta_sgd_params[0] * grads[0]
            c = c - meta_sgd_params[1] * grads[1]

        # Return loss with resulting latents
        return mse_loss((p, c, g), x_i[i + 1], y_i[i + 1]), (p, c, g)

    @jax.jit
    def outer_step(x_i, y_i, enf_params, meta_sgd_params, enf_opt_state, meta_sgd_opt_state, key):
        # Perform inner loop optimization
        (loss, _), grads = jax.value_and_grad(inner_loop, has_aux=True)([enf_params, meta_sgd_params], x_i, y_i, key)

        # Update the ENF backbone
        enf_grads, enf_opt_state = enf_optimizer.update(grads[0], enf_opt_state)
        enf_params = optax.apply_updates(enf_params, enf_grads)

        # Update the meta SGD parameters
        meta_sgd_grads, meta_sgd_opt_state = meta_sgd_optimizer.update(grads[1], meta_sgd_opt_state)
        meta_sgd_params = optax.apply_updates(meta_sgd_params, meta_sgd_grads)

        # Sample new key
        new_key, key = jax.random.split(key)
        return loss, enf_params, meta_sgd_params, enf_opt_state, meta_sgd_opt_state, new_key

    # Logic for subsampling coordinates and corresponding pixel values
    @jax.jit
    def subsample_xy(key, x_, y_):
        """ Subsample coordinates and pixel values for every inner step, and additionally for the final step that is
        used to calculate gradients for the outer step.

        Args:
             x_: coordinates, shape (batch, num_points, 2)
             y_: pixel values, shape (batch, num_points, num_channels)
        """
        mask = jnp.broadcast_to(jnp.arange(x.shape[1])[jnp.newaxis],(config.optim.inner_steps + 1, x.shape[1]))
        sub_mask = jax.random.permutation(key, mask, independent=True, axis=1)[:, :config.optim.num_subsampled_points]
        x_i = jax.vmap(lambda i: x_[:, i])(sub_mask)
        y_i = jax.vmap(lambda i: y_[:, i])(sub_mask)
        return x_i, y_i

    # Training loop
    key = jax.random.PRNGKey(seed)
    for epoch in range(config.train.num_epochs):
        epoch_loss = []
        for i, batch in enumerate(train_dloader):
            # Unpack batch, flatten img, subsample coordinates
            img, _ = batch
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))

            # Subsample coordinates, separate for every inner step
            x_i, y_i = subsample_xy(key, x, y)

            # Perform outer loop optimization
            loss, enf_params, meta_sgd_params, enf_opt_state, meta_sgd_opt_state, key = outer_step(
                x_i, y_i, enf_params, meta_sgd_params, enf_opt_state, meta_sgd_opt_state, key)

            epoch_loss.append(loss)
        
        logging.info(f"epoch {epoch} -- loss: {sum(epoch_loss) / len(epoch_loss)}")

    # Reconstruct and log an image, perform inner loop
    _, (p, c, g) = inner_loop([enf_params, meta_sgd_params], x_i, y_i, key)

    # Reconstruct image
    img_reconstructed = model.apply(enf_params, x, p, c, g)[0]

    # Plot the original and reconstructed image
    plt.figure()
    plt.subplot(121)
    plt.imshow(jnp.reshape(img[0], (img_shape)))
    plt.title("Original")
    plt.subplot(122)
    plt.imshow(jnp.reshape(img_reconstructed, (img_shape)))
    plt.title("Reconstructed")
    plt.show()

if __name__ == "__main__":
    main()
