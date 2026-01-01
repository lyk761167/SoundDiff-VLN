# SoundSpaces Dataset

## Overview
The SoundSpaces dataset includes audio renderings (room impulse responses) for two datasets, metadata of each scene, episode datasets and mono sound files. 


## Download
0. Create a folder named "data" under root directory
1. Run the commands below in the **data** directory to download partial binaural RIRs (867G), metadata (1M), datasets (77M) and sound files (13M). Note that this partial binaural RIRs only contain renderings for nodes accessible by the agent on the navigation graph. 
```
wget http://dl.fbaipublicfiles.com/SoundSpaces/binaural_rirs.tar && tar xvf binaural_rirs.tar
wget http://dl.fbaipublicfiles.com/SoundSpaces/metadata.tar.xz && tar xvf metadata.tar.xz
wget http://dl.fbaipublicfiles.com/SoundSpaces/sounds.tar.xz && tar xvf sounds.tar.xz
wget http://dl.fbaipublicfiles.com/SoundSpaces/datasets.tar.xz && tar xvf datasets.tar.xz
wget http://dl.fbaipublicfiles.com/SoundSpaces/pretrained_weights.tar.xz && tar xvf pretrained_weights.tar.xz
```
2. Download [Replica-Dataset](https://github.com/facebookresearch/Replica-Dataset) and [Matterport3D](https://niessner.github.io/Matterport).
3. Run the command below in the root directory to cache observations for two datasets (**with habitat-sim and habitat-lab versions being v0.1.7**)
```
python scripts/cache_observations.py
```
4. (Optional) Download the full ambisonic (3.6T for Matterport) and binaural (682G for Matterport and 81G for Replica) RIRs data by running the following script in the root directory. Remember to first back up the downloaded bianural RIR data.
```
python scripts/download_data.py --dataset mp3d --rir-type binaural_rirs
python scripts/download_data.py --dataset replica --rir-type binaural_rirs
```


## Data Folder Structure
```
data/
├── datasets/
│   ├── audionav/
│   └── semantic_audionav/
├── scene_datasets/
│   └── mp3d/
│       ├── 17DRP5sb8fy/
│       ├── 1LXtFkjw3qL/
│       └── ..../
├── scene_observations/
│   └── mp3d/
│       ├── 17DRP5sb8fy.pkl/
│       ├── 1LXtFkjw3qL.pkl/
│       └── ..../
├── hf_models/
│   ├── Qwen2-Audio-7B-audio_tower/
│   │   ├── model.safetensors/
│   │   └── config.json/
│   └── Qwen2-Audio-7B/
│       ├── tokenizer.json/
│       ├── model-00001-of-00005.safetensors/
│       └── ..../
├── metadata/
│   └── mp3d/
├── r2r-audio/
│   ├── train/
│   │   ├── train.json/
│   │   └── ..../
│   ├── val_seen/
│   │   ├── val_seen.json/
│   │   └── ..../
│   └── val_unseen/
│       ├── val_unseen.json/
│       └── ..../
├── r2r/
├── binaural_rirs/
├── pretrained_weights/
├── sounds/
└── models
    ├──output/
    └── mp3d/
```
