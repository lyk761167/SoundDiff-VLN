# SoundDiff-VLN
A Task-Conditioned Diffusion Acoustic Prior Framework for Instruction-Level Audio-Visual-Language Navigation
<a id="top"></a>

## 🛠️ Installation

Create the required environment through the following steps:

```bash
git clone https://github.com/lyk761167/SoundDiff-VLN && cd SoundDiff-VLN

conda create -n ss python=3.9 cmake=3.14.0 -y && conda activate janusvln
Install [habitat-sim v0.1.7](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7)

Check further instructions from [here](https://github.com/facebookresearch/habitat-sim/blob/v0.1.7/BUILD_FROM_SOURCE.md) — `--headless --with-cuda`

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
