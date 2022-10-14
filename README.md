# Learnable Polyphase Sampling for Shift Invariant and Equivariant Convolutional Networks

### [[Project Page]](https://raymondyeh07.github.io/learnable_polyphase_sampling) [[Paper (NeurIPS 2022)]]()

[Renan A. Rojas-Gomez](http://renanar2.web.illinois.edu/)\*,
[Teck-Yian Lim](https://teckyianlim.me/)\*,
[Alexander G. Schwing](http://www.alexander-schwing.de/),
[Minh N. Do](https://minhdo.ece.illinois.edu/),
[Raymond A. Yeh](https://www.raymond-yeh.com/)<sup>1</sup><br>
(*Equal Contribution) <br>
University of Illinois at Urbana-Champaign, Purdue University<sup>1</sup><br/>

<p align="center">
<img src='https://raymondyeh07.github.io/learnable_polyphase_sampling/resource/pipeline.png' width=800>
</p>


# Overview
This is the official implementation of "Learnable Polyphase Sampling for Shift Invariant and Equivariant Convolutional Networks" accepted at NeurIPS 2022. If you use this code or found it helpful, please consider citing:
<pre>
@inproceedings{rojas-neurips2022-learnable,
  title = {Learnable Polyphase Sampling for Shift Invariant and Equivariant Convolutional Networks},
  author = {Rojas-Gomez$^*$, Renan A and Lim$^*$, Teck Yian and Schwing, Alexander G and Do, Minh N and Yeh, Raymond A}
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
cd learn_poly_sampling/
conda env create -f env_reqs.yml
```

To check the installation, execute the following:
```
make
```

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

#### Experiments Setup for Image Classification (ImageNet)

- Download the ILSVRC2012 dataset from its [official repository](https://image-net.org/challenges/LSVRC/2012/), uncompress it into the dataset folder (e.g. `/learn_poly_sampling/datasets/ILSVRC2012`) and split it into train and val partitions using [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).

- Classification accuracy or shift consistency can be computed by setting the `--eval_mode` flag as either `class_accuracy` or `shift_consistency`, respectively.

#### Results for Image Classification (ImageNet)
To reproduce our results in Tab. 2 \& 3, run the scripts included in ```learn_poly_sampling/scripts``` with our pre-trained models.

### [Pre-trained Classification Models](https://uofi.box.com/s/pql7u3c7x8zifp0m46xhe2uduwwazcad)
Please refer to the link above to download all our pre-trained classification models. Note that our evaluation scripts assume the checkpoints are stored at ```learn_poly_sampling/checkpoints/classification```.


## Acknowledgements
We thank the authors of [antialiased-cnns](https://github.com/adobe/antialiased-cnns), [Adaptive-anti-Aliasing
](https://github.com/MaureenZOU/Adaptive-anti-Aliasing), and [truly_shift_invariant_cnns
](https://github.com/achaman2/truly_shift_invariant_cnns) for open-sourcing their code, which we refer to and used during the development of this project.
