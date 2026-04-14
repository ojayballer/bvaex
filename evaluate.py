import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import math

from model.model  import CONVAE

def load_weights(model, path="weights/epoch_100"):
    """Load saved weights into the model"""
    
    # Encoder conv layers
    for i, layer in enumerate(model.encoder.conv_layers):
        layer.weights = jnp.load(f"{path}/encoder_conv_{i}_weights.npy")
        layer.biases = jnp.load(f"{path}/encoder_conv_{i}_biases.npy")
    
    # Encoder mu dense layer
    model.encoder.mu.weight = jnp.load(f"{path}/mu_s_weights.npy")
    model.encoder.mu.bias = jnp.load(f"{path}/mu_s_biases.npy")
    
    # Encoder log_var dense layer
    model.encoder.log_var.weight = jnp.load(f"{path}/log_var_s_weight.npy")
    model.encoder.log_var.bias = jnp.load(f"{path}/log_var_s_bias.npy")
    
    # Decoder dense layer
    model.decoder.dense_layer.weight = jnp.load(f"{path}/decoder_s_dense_layer_weight.npy")
    model.decoder.dense_layer.bias = jnp.load(f"{path}/decoder_s_dense_layer_bias.npy")
    
    # Decoder transposed conv layers
    for i, layer in enumerate(model.decoder.transposed_conv_layers):
        layer.weights = jnp.load(f"{path}/decoder_transposed_conv_{i}_weights.npy")
        layer.biases = jnp.load(f"{path}/decoder_transposed_conv_{i}_biases.npy")
    
    print(f"Weights loaded from {path}")
    return model


def load_images(num_images=8, path="archive/img_align_celeba/img_align_celeba"):
    """Load a few CelebA images for reconstruction"""
    image_paths = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')])
    
    images = []
    for i in range(num_images):
        image = Image.open(image_paths[i])
        image = image.resize((64, 64))
        image = jnp.array(image) / 255.0
        image = jnp.transpose(image, (2, 0, 1))  # (H,W,C) -> (C,H,W)
        images.append(image)
    
    return jnp.array(images)


def reconstruct_images(model, images, key):
    """Pass images through encoder -> z -> decoder"""
    mu, log_var = model.encoder.forward(images)
    z = mu  # Use mu directly for clean evaluation
    reconstructed = model.decoder.forward(z)
    return reconstructed


def generate_from_prior(model, num_images=8, key=None):
    """Sample random z from N(0,I) and decode"""
    if key is None:
        key = jax.random.PRNGKey(0)
    
    z = jax.random.normal(key, (num_images, 128))  # latent_dim=128
    generated = model.decoder.forward(z)
    return generated


def tensor_to_image(tensor):
    """Convert (C,H,W) tensor to (H,W,C) numpy array for plotting"""
    img = np.array(tensor)
    img = np.transpose(img, (1, 2, 0))  # (C,H,W) -> (H,W,C)
    img = np.clip(img, 0, 1)  # clip to valid range
    return img


def plot_reconstruction(originals, reconstructions, cols=8, save_path="results/reconstruction_grid.png"):
    """Plot original vs reconstructed images side by side"""
    num_images = originals.shape[0]
    
    blocks = math.ceil(num_images / cols)
    rows = blocks * 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    
    if rows == 1:
        axes = axes[np.newaxis, :]
    if cols == 1:
        axes = axes[:, np.newaxis]
        
    for i in range(num_images):
        block_idx = i // cols
        col_idx = i % cols
        
        orig_row = block_idx * 2
        recon_row = orig_row + 1
        
        ax_orig = axes[orig_row, col_idx]
        ax_orig.imshow(tensor_to_image(originals[i]))
        ax_orig.axis('off')
        
        if block_idx == 0:
            ax_orig.set_title('Original', fontsize=10)
            
        ax_recon = axes[recon_row, col_idx]
        ax_recon.imshow(tensor_to_image(reconstructions[i]))
        ax_recon.axis('off')
        
        if block_idx == 0:
            ax_recon.set_title('Reconstructed', fontsize=10)
            
    for i in range(num_images, blocks * cols):
        block_idx = i // cols
        col_idx = i % cols
        axes[block_idx * 2, col_idx].axis('off')
        axes[block_idx * 2 + 1, col_idx].axis('off')
        
    plt.suptitle('Original vs Reconstructed', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")


def plot_generated(generated_images, save_path="results/generated_faces.png"):
    """Plot generated faces from random z"""
    num_images = generated_images.shape[0]
    rows = 8
    cols = 8
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    
    for i in range(num_images):
        r = i // cols
        c = i % cols
        axes[r, c].imshow(tensor_to_image(generated_images[i]))
        axes[r, c].axis('off')
    
    plt.suptitle('Generated from Random z ~ N(0,I)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")


def main():
    # Create model with same architecture as training
    model = CONVAE(
        input=(3, 64, 64),
        input_shape=(128, 8, 8),
        kernel_size=4,
        encoder_filters=[32, 64, 128],
        decoder_filters=[64, 32, 3],
        latent_dim=128
    )
    
    # Load trained weights
    model = load_weights(model, path="weights/epoch_100")
    
    # Load some CelebA images
    print("Loading images...")
    images = load_images(num_images=32)
    
    # Reconstruct
    print("Reconstructing images...")
    key = jax.random.PRNGKey(42)
    reconstructed = reconstruct_images(model, images, key)
    print(f"Reconstructed min: {jnp.min(reconstructed)}, max: {jnp.max(reconstructed)}, mean: {jnp.mean(reconstructed)}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Channel 0 (R): min={jnp.min(reconstructed[:,0])}, max={jnp.max(reconstructed[:,0])}, mean={jnp.mean(reconstructed[:,0])}")
    print(f"Channel 1 (G): min={jnp.min(reconstructed[:,1])}, max={jnp.max(reconstructed[:,1])}, mean={jnp.mean(reconstructed[:,1])}")
    print(f"Channel 2 (B): min={jnp.min(reconstructed[:,2])}, max={jnp.max(reconstructed[:,2])}, mean={jnp.mean(reconstructed[:,2])}")
    
    plot_reconstruction(images, reconstructed, cols=8)
    
    # Generate new faces from random z
    print("Generating new faces...")
    key2 = jax.random.PRNGKey(123)
    generated = generate_from_prior(model, num_images=64, key=key2)
    plot_generated(generated)


if __name__ == "__main__":
    main()