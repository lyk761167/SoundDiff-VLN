# SoundDiff-VLN
A Task-Conditioned Diffusion Acoustic Prior Framework for Instruction-Level Audio-Visual-Language Navigation
<a id="top"></a>

## Installation

Create the required environment through the following steps:


1.`git clone https://github.com/lyk761167/SoundDiff-VLN && cd SoundDiff-VLN`

2.Create a virtual env with python=3.9, this will be used throughout:
`conda create -n sounddiff python=3.9 cmake=3.14.0 -y && conda activate sounddiff`


3.Install [habitat-sim v0.1.7](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7)

Check further instructions from [here](https://github.com/facebookresearch/habitat-sim/blob/v0.1.7/BUILD_FROM_SOURCE.md) — `--headless --with-cuda`

4.Install habitat-lab-dialog (modified version of habitat-lab v0.1.7).
```bash
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout v0.1.7
pip install -e .
```
##  Data Preparation
1. Scene Dataset

- Download the MP3D scenes from the [official project page]([https://niessner.github.io/Matterport/]), and place them under `data/scene_datasets/mp3d/`.

2. R2R-CE Episodes
   Download the VLN-CE episodes and extract them into the data directory:

- [r2r] ([https://YOUR_LINK_HERE](https://niessner.github.io/Matterport/))(Rename R2R-CE)
3. Soundspace dataset
  Follow instructions on the dataset page to download the rendered audio data and datasets and put them under project/data/ folder.

conda install habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat

git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
cd ..

# CUDA 12.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org

pip install -r requirements.txt
# Install JanusVLN
pip install -e .
