# 💫 enf-min-jax: Minimal implementation of Equivariant Neural Fields in JAX with usage examples

**Authors**: David R. Wessels*, David M. Knigge*, Riccardo Valperga, Samuele Papa, Sharvaree Vadgama, Efstratios Gavves, Erik J. Bekkers 

**equal contribution*
___
In this repo, we provide a minimal representation of the Equivariant Neural Field (ENF) architecture introduced in the paper: ["Grounding Continuous Representations in Geometry: Equivariant Neural Fields"](https://arxiv.org/abs/2406.05753).

The code is written in JAX and is designed to be as simple as possible to understand the core concepts of the ENF architecture, and be maximally extensible for your own shenanigans.

As such, this code isn't enough to reproduce the experimental results in the paper, we provide that code in a [separate repo](https://github.com/dafidofff/enf-jax).

Any questions, requests, comments, or suggestions, please feel free to open an issue or PR.

# 🌊 Equivariant Neural Fields (ENF) Experiments

We'll be expanding on the list of example use-cases, currently this repo consists of:

1. Ombria Flood Segmentation. See [this script](experiments/ombria_segmentation.py).
2. CIFAR-10 Classification. See [this script](experiments/cifar10_classification.py).
3. CIFAR-10 Generative Modelling w/ Diffusion. See [this script](experiments/cifar10_diffusion.py).

## 🛠️ Installation
The following lines will install all required dependencies, assuming CUDA 12 is installed:
```bash
conda create -n enf python=3.11
conda activate enf
pip install -U "jax[cuda12]" flax optax matplotlib ml-collections pillow wandb
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```


## 🧪 Experiments

### 1. Ombria Flood Segmentation

A multi-modal segmentation task using the OMBRIA flood map dataset. The experiment uses both Sentinel-1 (S1) and Sentinel-2 (S2) satellite imagery to detect flooded areas.

The pipeline features two ENF architectures; one for reconstruction and one for segmentation. A single input sample consists of two S1 and two S2 images, from before and after the flood event. All these images are stacked into 8 channels and reconstructed using a single set of ENF latents. After pretraining, this set of ENF latents is used as input for a second ENF that learns to predict the segmentation mask from these latents.

**Usage:**
```bash
export PYTHONPATH="."; python experiments/ombria_segmentation.py
```

### 2. CIFAR-10 Classification

An image classification experiment on CIFAR-10 using a two-stage approach:
1. ENF for feature extraction
2. Transformer classifier for classification

**Usage:**
```bash
export PYTHONPATH="."; python experiments/cifar10_classification.py
```

### 3. CIFAR-10 Diffusion

A diffusion model experiment on CIFAR-10 using ENFs for feature extraction and a Diffusion Transformer (DiT) for generation. We use a v-diffusion model with a Gaussian diffusion process. Code was adapted from this [JAX repo](https://github.com/kvfrans/jax-flow) and modified for the ENF architecture.

**Usage:**
```bash
export PYTHONPATH="."; python experiments/cifar10_diffusion.py
```

## 📊 Monitoring

All experiments use Weights & Biases (wandb) for experiment tracking.

## 🎯 Training Pipeline

Each experiment follows a similar two-stage training approach:

1. **Pretraining Stage**:
   - Train the ENF backbone to reconstruct input images
   - Uses inner-loop optimization for latent variables
   - Monitors reconstruction quality

2. **Task-Specific Stage**:
   - Freezes ENF backbone
   - Trains task-specific head (classifier/segmentation/diffusion)
   - Uses normalized latent representations

## 📝 Citation

If you find this code useful, please cite the original paper:

```bibtex
@article{wessels2024enf,
  title={Grounding Continuous Representations in Geometry: Equivariant Neural Fields},
  author={Wessels, David R. and Knigge, David M. and Valperga, Riccardo and Papa, Samuele and Vadgama, Sharvaree and Gavves, Efstratios and Bekkers, Erik J.},
  journal={arXiv preprint arXiv:2406.05753},
  year={2024}
}
```

## 🤝 Contributing

Feel free to open issues or submit pull requests. We welcome:
- Bug fixes
- New experiments
- Documentation improvements
- Performance optimizations
