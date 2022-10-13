# Learnable Polyphase Sampling for Shift Invariant and Equivariant Convolutional Networks

### Thirty-sixth Conference on Neural Information Processing Systems (NeurIPS 2022)

[Renán A. Rojas-Gómez](http://renanar2.web.illinois.edu/)\*,
[Teck-Yian Lim](https://teckyianlim.me/)\*,
[Alexander G. Schwing](http://www.alexander-schwing.de/),
[Minh N. Do](https://minhdo.ece.illinois.edu/),
[Raymond A. Yeh](https://www.raymond-yeh.com/)<sup>1</sup><br>
(*Equal Contribution) <br>
University of Illinois at Urbana-Champaign, Purdue University<sup>1</sup><br/>


# Overview
This is the official implementation of "Learnable Polyphase Sampling for Shift Invariant and Equivariant Convolutional Networks" accepted at NeurIPS 2022. If you use this code or found it helpful, please consider citing:
<pre>
@inproceedings{rojas-neurips2022-learnable,
  title = {Learnable Polyphase Sampling for Shift Invariant and Equivariant Convolutional Networks},
  author = {Rojas-G{\'o}mez$^*$, Ren{\'a}n and Lim$^*$, Teck Yian and Schwing, Alexander G and Do, Minh N and Yeh, Raymond A}
  booktitle = {Neural Information Processing Systems (NeurIPS)},
  year = {2022},
  note = {($^*$ Equal Contribution)},
}
</pre>

## Setup Dependencies
All our experiments were executed using:
- python v3.8.10
- pytorch-lightning v1.4.0
- torchvision v0.10.0
- pytorch-lightning v1.4.0
- cudatoolkit v10.2.89
- opencv-python v4.5.4.60

For a full list of requirements, please refer to ***learn_poly_sampling/env_reqs.yml***. To install the dependencies, please first install [mini-conda](https://docs.conda.io/en/latest/miniconda.html) and execute:

```bash
conda env create -f env_reqs.yml
```

## Run Tests
Our paper results can be reproduced using the scripts included in ***learn_poly_sampling/scripts***.

1. #### Experiments on Image Classification (CIFAR-10)

- Download the CIFAR-10 dataset from its [official repository](https://www.cs.toronto.edu/~kriz/cifar.html) and uncompress it into the dataset folder (e.g. `/learn_poly_sampling/datasets/cifar-10-batches-py`).

- Classification accuracy or shift consistency can be computed by setting the `--eval_mode` flag as either `class_accuracy` or `shift_consistency`, respectively.

2. #### Experiments on Image Classification (ImageNet)

- Download the ILSVRC2012 dataset from its [official repository](https://image-net.org/challenges/LSVRC/2012/), uncompress it into the dataset folder (e.g. `/learn_poly_sampling/datasets/ILSVRC2012`) and split it into train and test partitions using [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).

- Classification accuracy or shift consistency can be computed by setting the `--eval_mode` flag as either `class_accuracy` or `shift_consistency`, respectively.

## Demo
For notebook demonstrations of our proposed LPS (LPD and LPU) layers, please refer to the ***demo*** directory.

To run the notebook, please execute:
```
conda install -c conda-forge notebook
conda install -c conda-forge nb_conda_kernels
conda install -c conda-forge matplotlib
jupyter-notebook demo
```

## Results and Pre-trained models
### [Pre-trained Classification and Segmentation Models](https://uofi.box.com/s/mb2ygdziztjxxkjyybvpkumkkzgezofz)
Please refer to the link above to download all our pre-trained classification and semantic segmentation models. Note that our evaluation scripts assume the checkpoints are stored at ***learn_poly_sampling/checkpoints/{classification, segmentation}***.
