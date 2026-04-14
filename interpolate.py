import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

from model.model import CONVAE

def load_weights(model, path="weights/epoch_100"):
    for i, layer in enumerate(model.encoder.conv_layers):
        layer.weights = jnp.load(f"{path}/encoder_conv_{i}_weights.npy")
        layer.biases = jnp.load(f"{path}/encoder_conv_{i}_biases.npy")
    model.encoder.mu.weight = jnp.load(f"{path}/mu_s_weights.npy")
    model.encoder.mu.bias = jnp.load(f"{path}/mu_s_biases.npy")
    model.encoder.log_var.weight = jnp.load(f"{path}/log_var_s_weight.npy")
    model.encoder.log_var.bias = jnp.load(f"{path}/log_var_s_bias.npy")
    model.decoder.dense_layer.weight = jnp.load(f"{path}/decoder_s_dense_layer_weight.npy")
    model.decoder.dense_layer.bias = jnp.load(f"{path}/decoder_s_dense_layer_bias.npy")
    for i, layer in enumerate(model.decoder.transposed_conv_layers):
        layer.weights = jnp.load(f"{path}/decoder_transposed_conv_{i}_weights.npy")
        layer.biases = jnp.load(f"{path}/decoder_transposed_conv_{i}_biases.npy")
    return model

def load_two_images(idx1=0, idx2=5, path="archive/img_align_celeba/img_align_celeba"):
    """Load two specific CelebA images to interpolate between"""
    image_paths = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')])
    images = []
    for idx in [idx1, idx2]:
        image = Image.open(image_paths[idx])
        image = image.resize((64, 64))
        image = jnp.array(image) / 255.0
        image = jnp.transpose(image, (2, 0, 1))  # (H,W,C) -> (C,H,W)
        images.append(image)
    return jnp.array(images)

def tensor_to_image(tensor):
    img = np.array(tensor)
    img = np.transpose(img, (1, 2, 0))  # (C,H,W) -> (H,W,C)
    img = np.clip(img, 0, 1)
    return img

def interpolate(model, images, num_steps=10):
    """Interpolate between two faces in latent space"""
    # Encode both images to get their latent vectors
    mu, log_var = model.encoder.forward(images)
    z1 = mu[0]  # latent vector for face 1
    z2 = mu[1]  # latent vector for face 2
    
    # Create interpolation steps: z = z1 * (1-alpha) + z2 * alpha
    alphas = jnp.linspace(0, 1, num_steps)
    z_interp = jnp.array([z1 * (1 - a) + z2 * a for a in alphas])
    
    # Decode all interpolated latent vectors
    decoded = model.decoder.forward(z_interp)
    return decoded
def main():
    model = CONVAE(
        input=(3, 64, 64),
        input_shape=(128, 8, 8),
        kernel_size=4,
        encoder_filters=[32, 64, 128],
        decoder_filters=[64, 32, 3],
        latent_dim=128
    )
    model = load_weights(model, path="weights/epoch_100")
    
    # four different face pairs
    pairs =[(0, 5), (1, 7), (3, 6), (2, 4)]
    num_steps = 10
    
    fig, axes = plt.subplots(len(pairs), num_steps + 2, figsize=(2 * (num_steps + 2), 2.5 * len(pairs)))
    
    for row, (idx1, idx2) in enumerate(pairs):
        images = load_two_images(idx1=idx1, idx2=idx2)
        interpolated = interpolate(model, images, num_steps=num_steps)
        
        # Original face A
        axes[row, 0].imshow(tensor_to_image(images[0]))
        axes[row, 0].set_title('Face A' if row == 0 else '', fontsize=9)
        axes[row, 0].axis('off')
        
        # Interpolation steps
        for i in range(num_steps):
            axes[row, i + 1].imshow(tensor_to_image(interpolated[i]))
            axes[row, i + 1].set_title(f'α={i/(num_steps-1):.1f}' if row == 0 else '', fontsize=8)
            axes[row, i + 1].axis('off')
        
        # Original face B
        axes[row, -1].imshow(tensor_to_image(images[1]))
        axes[row, -1].set_title('Face B' if row == 0 else '', fontsize=9)
        axes[row, -1].axis('off')
    
    plt.suptitle('Latent Space Interpolation', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/latent_interpolation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to latent_interpolation.png")

main()