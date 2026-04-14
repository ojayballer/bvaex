# Convolutional Beta VAE in JAX

A Convolutional Variational Autoencoder implemented in JAX, with all core components written manually, including forward passes, backward propagation, and parameter updates.

This project focuses on building and training a complete generative model by working directly with `jax.numpy` and `jax.lax`, without relying on automatic differentiation or high level neural network libraries. The model was trained on the full CelebA dataset using GPU acceleration.

---

## Features

Implementation of core components:

- **Conv2D (forward and backward)**  
  [view implementation](model/conv.py)

- **TransposedConv2D (forward and backward)**  
  [view implementation](model/transposed_conv.py)

- **Dense layers with manual gradients**  
  [view implementation](model/dense.py)

- **Activation functions (ReLU, Sigmoid)**  
  [view implementation](model/activations.py)

- **Reparameterization trick (μ, σ, ε)**  
  [view implementation](model/encoder.py)

- **ELBO loss (Reconstruction + KL Divergence)**  
  [view implementation](model/loss.py)

- **Custom Adam optimizer (AdamW style)**  
  [view implementation](model/optimizer.py)

All gradients and parameter updates are explicitly derived and implemented without autograd.

---

## Dataset

- CelebA dataset  
- 202,599 images (full dataset)  
- Resolution: 64 × 64  
- Format: CHW tensors  

The model was trained on the complete CelebA dataset using a custom data loader that streams batches directly from disk.

Dataset location:

```
archive/img_align_celeba/img_align_celeba/
```

---

## Setup and Installation

```bash
git clone https://github.com/ojayballer/ConvVAE-from-scratch-using-JAX.git
cd ConvVAE-from-scratch-using-JAX
pip install jax jaxlib numpy matplotlib pillow
```

---

## Training

Run training:

```bash
python train.py
```

**Training details**

- GPU: NVIDIA Tesla P100  
- Epochs: 100  
- Batch size: 512  
- Training time: ~2 hours on full dataset  

JAX was used to compile all numerical operations through XLA, allowing the manually implemented layers and gradients to execute efficiently on GPU.

Trained weights are stored in:

```
weights/epoch_100/
```

---

## Results

**Evaluation on 1,000 hold out images**

- SSIM: 0.9364  
- PSNR: 22.49 dB  

---

## Visualizations

**Generated Faces**
![Generated Faces](results/generated_faces.png)

**Reconstructions**
![Reconstructed Faces](results/reconstruction_grid.png)

**Latent Interpolation**
![Latent Interpolation](results/latent_interpolation.png)

**Training Loss**
![Loss Curves](results/loss_curves.png)

Outputs are stored in:

```
results/
```

---

## References

- [Auto Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)  
- [Adam A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)  
- [Beta VAE Learning Basic Visual Concepts](https://openreview.net/forum?id=Sy2fzU9gl)  