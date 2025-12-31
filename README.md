# SoundDiff-VLN
A Task-Conditioned Diffusion Acoustic Prior Framework for Instruction-Level Audio-Visual-Language Navigation
<a id="top"></a>

## 🛠️ Installation

Create the required environment through the following steps:

```bash
git clone https://github.com/MIV-XJTU/JanusVLN.git && cd JanusVLN

conda create -n janusvln python=3.9 -y && conda activate janusvln

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
