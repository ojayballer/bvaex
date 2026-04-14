import jax.numpy as jnp
import jax
import numpy as np
from PIL import Image
import os
from model.model import CONVAE

def load_weights(model, path="saved_weights/epoch_100"):
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

def compute_ssim(img1, img2, window_size=7):
    """Compute SSIM between two images (H,W,C) in range [0,1]"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_vals = []
    for c in range(3):
        a = img1[:, :, c]
        b = img2[:, :, c]
        
        # Simple mean filter (box filter) instead of gaussian for speed
        from jax import lax
        kernel = jnp.ones((window_size, window_size)) / (window_size * window_size)
        
        mu_a = np.mean(a)
        mu_b = np.mean(b)
        sigma_a_sq = np.var(a)
        sigma_b_sq = np.var(b)
        sigma_ab = np.mean((a - mu_a) * (b - mu_b))
        
        numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
        denominator = (mu_a**2 + mu_b**2 + C1) * (sigma_a_sq + sigma_b_sq + C2)
        
        ssim_vals.append(numerator / denominator)
    
    return np.mean(ssim_vals)

def compute_psnr(img1, img2):
    """Compute PSNR between two images (H,W,C) in range [0,1]"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)

def main():
    model = CONVAE(
        input=(3, 64, 64),
        input_shape=(128, 8, 8),
        kernel_size=4,
        encoder_filters=[32, 64, 128],
        decoder_filters=[64, 32, 3],
        latent_dim=128
    )
    model = load_weights(model, path="saved_weights/epoch_100")
    
    # Load images
    path = "archive/img_align_celeba/img_align_celeba"
    image_paths = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')])
    
    num_images = 1000
    batch_size = 32
    
    all_ssim = []
    all_psnr = []
    
    print(f"Computing SSIM and PSNR on {num_images} images...")
    
    for start in range(0, num_images, batch_size):
        # Load batch
        images = []
        for i in range(start, min(start + batch_size, num_images)):
            image = Image.open(image_paths[i]).resize((64, 64))
            image = np.array(image, dtype=np.float32) / 255.0
            images.append(image)
        
        batch_np = np.array(images)
        # Convert to CHW for model
        batch = jnp.transpose(jnp.array(batch_np), (0, 3, 1, 2))
        
        # Reconstruct
        mu, log_var = model.encoder.forward(batch)
        z = mu
        reconstructed = model.decoder.forward(z)
        
        # Convert back to HWC numpy
        recon_np = np.array(jnp.transpose(reconstructed, (0, 2, 3, 1)))
        recon_np = np.clip(recon_np, 0, 1)
        
        # Compute metrics per image
        for i in range(len(images)):
            ssim_val = compute_ssim(batch_np[i], recon_np[i])
            psnr_val = compute_psnr(batch_np[i], recon_np[i])
            all_ssim.append(ssim_val)
            all_psnr.append(psnr_val)
        
        if (start // batch_size) % 10 == 0:
            print(f"Processed {start + len(images)}/{num_images}")
    
    print(f"\n{'='*40}")
    print(f"Results on {num_images} CelebA images:")
    print(f"{'='*40}")
    print(f"Mean SSIM:  {np.mean(all_ssim):.4f}")
    print(f"Mean PSNR:  {np.mean(all_psnr):.2f} dB")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()