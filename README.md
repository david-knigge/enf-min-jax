# ✴️ enf-min-jax: Minimal implementation of Equivariant Neural Fields in JAX

**Authors**: David M. Knigge*, David R. Wessels*, Riccardo Valperga, Samuele Papa, Sharvaree Vadgama, Efstratios Gavves, Erik J. Bekkers 

**equal contribution*
___
In this repo, we provide a minimal representation of the Equivariant Neural Field (ENF) architecture introduced in the paper: ["Grounding Continuous Representations in Geometry: Equivariant Neural Fields"](https://arxiv.org/abs/2406.05753).

The code is written in JAX and is designed to be as simple as possible to understand the core concepts of the ENF architecture, and be maximally extensible for your own shenanigans.

As such, this code isn't enough to reproduce the experimental results in the paper, we provide that code in a [separate repo]().

Any questions, requests, comments, or suggestions, please feel free to open an issue or PR.

## Installation
The following lines will install all required dependencies, assuming CUDA 12 is installed:
```bash
conda create -n enf python=3.11
conda activate enf
pip install -U "jax[cuda12]" flax optax matplotlib ml-collections
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Usage
To fit an ENF to STL10 images, run the following script:
```bash
python example_fitting_stl10.py
```

## Resources
* See this [notebook](https://colab.research.google.com/gist/david-knigge/8e38ace480e2fe19cfe52e2570e639dc/explainer_enf.ipynb) for a step-by-step guided implementation of ENFs for classification.
* See the [project page](https://dafidofff.github.io/enf-page).

## Citation
If you find this code useful, please consider citing the original paper:
```
@article{wessels2024enf,
  title={Grounding Continuous Representations in Geometry: Equivariant Neural Fields},
  author={Wessels, David R. and Knigge, David M. and Valperga, Riccardo and Papa, Samuele and Vadgama, Sharvaree and Gavves, Efstratios and Bekkers, Erik J.},
  journal={arXiv preprint arXiv:2406.05753},
  year={2024}
}
```
