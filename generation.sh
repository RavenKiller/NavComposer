#!/bin/bash
# ================================================
# Script Name   : generation.sh
# Description   : Run NavComposer
# Author        : Zongtao He
# Email         : raven025@outlook.com
# Last Modified : 2025-03-24
# Usage         : `bash generation.sh`
#                 or `sbatch generation.sh` (on a SLURM system)
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
# nohup ./clash-linux-amd64-v1.2.0 -f v2ray_forclash_9892.yaml -d . &
# export http_proxy=http://127.0.0.1:9892
# export https_proxy=http://127.0.0.1:9892
# cd -

cd ~
source .bashrc
cd -

# Change the `RUN_FOLDER` as needed
RUN_FOLDER=$1

conda activate navc

PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()');
torchrun --master_port ${PORT} main.py --config_file main.yaml run_folder=${RUN_FOLDER}