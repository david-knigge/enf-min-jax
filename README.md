# ✴️ enf-min-jax: Minimal implementation of Equivariant Neural Fields in JAX

**Authors**: David M. Knigge*, David R. Wessels*, Riccardo Valperga, Samuele Papa, Sharvaree Vadgama, Efstratios Gavves, Erik J. Bekkers 

**equal contribution*
___
In this repo, we provide a minimal representation of the Equivariant Neural Field (ENF) architecture introduced in the paper: ["Grounding Continuous Representations in Geometry: Equivariant Neural Fields"](https://arxiv.org/abs/2406.05753).

The code is written in JAX and is designed to be as simple as possible to understand the core concepts of the ENF architecture, and be maximally extensible for your own shenanigans.

As such, this code isn't enough to reproduce the experimental results in the paper, we provide that code in a [separate repo]().

Any questions, requests, comments, or suggestions, please feel free to open an issue or PR.
___
## Installation
The following lines will install all required dependencies, assuming CUDA 12 is installed:
```bash
conda create -n enf python=3.11
conda activate enf
pip install -U "jax[cuda12]" flax optax matplotlib ml-collections
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```


Optionally, install jupyter for running the explainer notebook:
```bash
pip install jupyter
```
___
## Usage
To fit an ENF to STL10 images, run the following script:
```bash
python example_fitting_stl10.py
```
___
## Resources
* See the [project page](https://davidmknigge.nl/enf-page).
___
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