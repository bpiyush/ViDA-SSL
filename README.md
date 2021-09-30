# ViDA-SSL
Video Dataset Analysis for Self-Supervised Learning


## Setup 

Note: Tested on Mac M1 (CPU-only) with `conda` installed and managed via `miniforge3`. See [this](https://bpiyush.github.io/ml-engg-docs/mac_m1.html) for more information on how to setup `conda` with `miniforge3` so that it works on the new Apple Silicon chip.

1. Create conda environment
   ```bash
   conda create -y -n vida-ssl
   conda activate vida-ssl
   ```
2. Install `python3.9`
   ```bash
   conda install -y python=3.9
   ```
3. Install packages
   ```bash
   pip install numpy matplotlib jupyter jupyterlab ipdb tqdm natsort
   pip install torch torchvision torchaudio torchtext
   conda install transformers
   pip install fast-pytorch-kmeans
   pip install bokeh
   ```