#!/bin/bash
# ================================================
# Script Name   : evaluation.sh
# Description   : Run NavInstrCritic evaluation
# Author        : Zongtao He
# Email         : raven025@outlook.com
# Last Modified : 2025-03-27
# Usage         : `bash evaluation.sh all inst_<suffix>`
#                 or `sbatch evaluation.sh all inst_<suffix>` (on a SLURM system)
#                 Before running, prepare val_seen and val_unseen splits of VLN-CE,
#                 and organize them in the data folder. Results will be stored in 
#                 `data/speaker_evaluation`. For more guidance, please refer to README.md
# ================================================

#SBATCH --job-name=critic
#SBATCH --partition A800
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a800:1
#SBATCH --output=data/critic.out

########### Proxy ###########

# Uncomment proxy setting as needed
cd ~/hzt/clash
nohup ./clash-linux-amd64-v1.2.0 -f v2ray_forclash_9892.yaml -d . &
export http_proxy=http://127.0.0.1:9892
export https_proxy=http://127.0.0.1:9892
cd -


########### Variable ###########
cd ~
source .bashrc
conda activate navc
cd -

# python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('punkt_tab')"

dataset_folder=data/vlnce_traj_action_clean
output_folder=data/speaker_evaluation


default_inst_aliases=inst_navcomposer
if [ $# -eq 1 ]; then
    run_types=$1
    inst_aliases=${default_inst_aliases}
elif [ $# -eq 2 ]; then
    run_types=$1
    inst_aliases=$2
else
    run_types=all
    inst_aliases=${default_inst_aliases}
fi
eval_splits="val_seen val_unseen"
eval_splits_list="["
for eval_split in $eval_splits; do
    eval_splits_list+="\"${eval_split}\","
done
L=$(expr ${#eval_splits_list} - 1)
eval_splits_list=${eval_splits_list:0:$L} # list style
eval_splits_list+="]"
eval_splits_cat=""
for eval_split in $eval_splits; do
    eval_splits_cat+="${eval_split}_"
done
L=$(expr ${#eval_splits_cat} - 1)
eval_splits_cat=${eval_splits_cat:0:$L} # underline seperated
consistency_backend=qwenlargebatch
mkdir ${output_folder}
python_cmd="from pathlib import Path; p=Path(\"${dataset_folder}\"); print(p.parent)"
dataset_folder_parent=$(python -c "${python_cmd}")
python_cmd="from pathlib import Path; p=Path(\"${dataset_folder}\"); print(p.name)"
dataset_folder_source=$(python -c "${python_cmd}")

for run_type in ${run_types}
do
    for inst_alias in ${inst_aliases}
    do
        ########### Log ###########

        date "+%Y-%m-%d %H:%M:%S"
        printf "Target instruction: %s\n" ${inst_alias}
        printf "Run type: %s\n" ${run_type}
        printf "Dataset folder: %s\n" ${dataset_folder}
        printf "Evaluation splits: %s\n" ${eval_splits}
        printf "Output folder: %s\n" ${output_folder}

        ########### Evaluation ###########
        if [[ $run_type = all || $run_type = matching ]]; then
            printf "\nRun contrastive matching evaluation ...\n"
            python tools/nav_instr_critic/matching_evaluation.py \
                --mode eval \
                data_config.eval_splits=${eval_splits_list} \
                data_config.text_alias=${inst_alias} \
                data_config.data_path=${dataset_folder_parent} \
                data_config.sources=["${dataset_folder_source}"] \
                trainer_config.batch_size=32 \
                trainer_config.eval_dir_alias=${inst_alias} \
                trainer_config.eval_output_folder=${output_folder}
            # mv data/checkpoints/clip-matcher-final/${inst_alias}/${eval_splits_cat}_ckpt.CLIPMatcher.11.json ${output_folder}/${inst_alias}_clipmatcher_${eval_splits_cat}.json
            # rm -r data/checkpoints/clip-matcher-final/${inst_alias}
        fi

        if [[ $run_type = all || $run_type = consistency ]]; then
            printf "\nRun semantic consistency evaluation ...\n"
            python tools/nav_instr_critic/consistency_evaluation.py \
                --mode matcher \
                --folder ${dataset_folder} \
                --splits ${eval_splits} \
                --inst_alias1 ${inst_alias} \
                --output_folder ${output_folder} \
                --backend ${consistency_backend} \
                --bfloat16
        fi

        if [[ $run_type = all || $run_type = diversity ]]; then
            printf "\nRun diversity evaluation ...\n"
            python tools/nav_instr_critic/diversity_evaluation.py \
                --folder ${dataset_folder} \
                --splits ${eval_splits} \
                --inst_alias ${inst_alias} \
                --output_folder ${output_folder}
        fi

        if [[ $run_type = all || $run_type = matching ]]; then
            cat ${output_folder}/${inst_alias}_clipmatcher_${eval_splits_cat}.json
        fi
        if [[ $run_type = all || $run_type = consistency ]]; then
            cat ${output_folder}/${inst_alias}_llmmatcher_${eval_splits_cat}_${consistency_backend}.json
        fi
        if [[ $run_type = all || $run_type = diversity ]]; then
            cat ${output_folder}/${inst_alias}_diversity_${eval_splits_cat}.json
        fi

    done
done