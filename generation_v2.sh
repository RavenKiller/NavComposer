#!/bin/bash
# ================================================
# Script Name   : generation_v2.sh
# Description   : Run NavComposerV2
# Author        : Zongtao He
# Email         : raven025@outlook.com
# Last Modified : 2025-06-22
# Usage         : `bash generation_v2.sh`
#                 or `sbatch generation_v2.sh` (on a SLURM system)
# ================================================

#SBATCH --job-name=composer
#SBATCH --partition A800
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a800:1
#SBATCH --output=data/composer.out

# Uncomment proxy setting as needed
# cd ~/hzt/clash
# nohup ./clash-linux-amd64-v1.2.0 -f 1725849302984.yml -d . &
# export http_proxy=http://127.0.0.1:7890
# export https_proxy=http://127.0.0.1:7890
# cd -

cd ~
source .bashrc
cd -

# Change the `RUN_FOLDER` as needed
RUN_FOLDER=$1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

conda activate navc

PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()');

torchrun --master_port ${PORT} main.py --config_file main.yaml run_folder=${RUN_FOLDER} generate_alias=inst_navcomposer_v2 speaker_config.name=NavComposerV2