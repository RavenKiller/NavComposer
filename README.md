## Introduction
We’re excited to release the code soon! Stay tuned for updates.

## Setup
### Requirements
Ensure that [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is installed and this repository is cloned before proceeding.

Create and activate a Conda environment, then install dependencies:
```bash
conda create -n navc python=3.9
conda activate navc
pip install -r requirements.txt
```
All necessary packages will be installed after running these commands.
The program has been tested on Linux.

### Trajectory Data Preparation

This project organizes trajectory datasets using the following three-level folder structure:
`<dataset_folder>/<episode_folder>/<modality_folder>`.

For example, a pre-processed VLN-CE val-unseen dataset follows this format:
```
val_unseen/
    15/
        rgb/
            0_0.jpg
            1_0.jpg
            2_0.jpg
            ...
        inst/
            0.txt
            1.txt
            2.txt
        ...
    21/
    42/
    57/
    ...
```
Each episode folder can contain multiple modality subfolders:
+ `rgb/` (required) contains trajectory images in temporal order. Image filenames can use any naming convention, provided they remain in chronological order when sorted using [natsort](https://github.com/SethMMorton/natsort).
+ `depth/` (optional) contains corresponding depth images.
+ `action/` (optional) contains a single JSON file (`0.json`) with ground truth actions. The action labels are: 0: stop, 1: move forward, 2: turn left, 3: turn right.
+ `inst/` (optional) contains original instructions (if available) in text files.
+ `inst_<suffix>/` (generated) contains generated instructions and any intermediate information (if available). Instructions are stored in text files (e.g., `0.txt`), and semantic entities are stored in JSON files (e.g., `0.info`).
+ `action_<suffix>/` (generated) contains generated actions obtained by running action inference independently.
+ `key_frame/` (generated) contains key frame indices in `0.json` after key frame downsampling. This folder appears only when the run mode is set to keyframe.

Dataset and episode folder names are arbitrary—you can name them as you prefer.
To use customized trajectory datasets, ensure they follow the same structure.

### Pre-processed Data
Several pre-processed datasets used in our experiments are available for download from BaiduNetdisk:

[VLN-CE Validation] [Diverse Sources]

We also provide generation results for the VLN-CE validation sets using two main variants: 

[Navcomposer Instructions]

By default, all datasets and model weights are placed in the `data/` folder.
Therefore, it's recommended to download them into `data/` and then unzip them.
If extracted to a different location, update the corresponding path variables in the code.


### Extra Preparation
Running variants other than vo-qwn-qwn-qwn and contrastive matching evaluation requires additional setup:

+ ResNet50 action module (rn) needs `data/checkpoints/actionclassifier/best_microsoft_resnet-50.pth`.
+ DINOv2 action module (dn) needs `data/checkpoints/actionclassifier/best_dinov2_base.pth`.
+ MAE scene module (mae) needs `data/checkpoints/mae/mae_tune_vit_base.pth`.
+ SWAG scene and object modules (swg) need `data/places365_cls_idx.json`, `data/in_cls_idx.json`.
+ gpt, llm, gmm modules need authentication. Change corresponding api_key or hf_token in `tools/config.py`.
+ **Important**: Contrastive matcher needs `data/checkpoints/cm/cm.pth`, `data/episodes_orders.json` and `data/episodes_insts_orders.json`. These two JSON files ensure consistent batch ordering; otherwise, evaluation results may vary.

These files are also available:

[[Model Weights]]()
