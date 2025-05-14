#!/bin/bash
# ================================================
# Script Name   : generation_full.sh
# Description   : Major experiments in our paper
# Author        : Zongtao He
# Email         : raven025@outlook.com
# Last Modified : 2025-03-24
# Usage         : `bash generation_full.sh`
#                 or `sbatch generation_full.sh` (on a SLURM system)
#                 Uncomment desired lines to run corresponding experiments.
# ================================================

#SBATCH --job-name=composer 
#SBATCH --partition A800
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a800:1
#SBATCH --output=data/composer.out

cd ~/hzt/clash
nohup ./clash-linux-amd64-v1.2.0 -f v2ray_forclash_9892.yaml -d . &
export http_proxy=http://127.0.0.1:9892
export https_proxy=http://127.0.0.1:9892
cd -

cd ~
source .bashrc
cd -
conda activate navc
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()');

########################################## 
# Module Implementations
# Note: Remember to change `RUN_FOLDER` and re-run the script if experiment on both `val_unseen` and `val_seen` splits
##########################################

# RUN_FOLDER=data/vlnce_traj_action_clean/val_unseen

# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_vo_llv_llv_llm scene_config.name=LlavaScene object_config.name=LlavaObject summary_config.name=Llama3Summary run_folder=${RUN_FOLDER}

# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_vo_mae_llv_llm scene_config.name=MAEScene object_config.name=LlavaObject summary_config.name=Llama3Summary run_folder=${RUN_FOLDER}
# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_vo_swg_llv_llm scene_config.name=SWAGScene object_config.name=LlavaObject summary_config.name=Llama3Summary run_folder=${RUN_FOLDER}
# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_vo_blp_llv_llm scene_config.name=BLIP2Scene object_config.name=LlavaObject summary_config.name=Llama3Summary run_folder=${RUN_FOLDER}
# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_vo_qwn_llv_llm scene_config.name=Qwen2Scene object_config.name=LlavaObject summary_config.name=Llama3Summary run_folder=${RUN_FOLDER}
# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_vo_gpt_llv_llm scene_config.name=GPTScene object_config.name=LlavaObject summary_config.name=Llama3Summary run_folder=${RUN_FOLDER}

# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_vo_llv_dtr_llm scene_config.name=LlavaScene object_config.name=DETRObject summary_config.name=Llama3Summary run_folder=${RUN_FOLDER}
# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_vo_llv_swg_llm scene_config.name=LlavaScene object_config.name=SWAGObject summary_config.name=Llama3Summary run_folder=${RUN_FOLDER}
# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_vo_llv_blp_llm scene_config.name=LlavaScene object_config.name=BLIP2Object summary_config.name=Llama3Summary run_folder=${RUN_FOLDER}
# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_vo_llv_qwn_llm scene_config.name=LlavaScene object_config.name=Qwen2Object summary_config.name=Llama3Summary run_folder=${RUN_FOLDER}
# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_vo_llv_gpt_llm scene_config.name=LlavaScene object_config.name=GPTObject summary_config.name=Llama3Summary run_folder=${RUN_FOLDER}

# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_vo_llv_llv_gmm scene_config.name=LlavaScene object_config.name=LlavaObject summary_config.name=Gemma2Summary run_folder=${RUN_FOLDER}
# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_vo_llv_llv_qwn scene_config.name=LlavaScene object_config.name=LlavaObject summary_config.name=Qwen25Summary run_folder=${RUN_FOLDER}
# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_vo_llv_llv_gpt scene_config.name=LlavaScene object_config.name=LlavaObject summary_config.name=GPTSummary run_folder=${RUN_FOLDER}

# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_dn_llv_llv_llm scene_config.name=LlavaScene object_config.name=LlavaObject summary_config.name=Llama3Summary run_folder=${RUN_FOLDER} action_config.name=Dinov2Action
# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_rn_llv_llv_llm scene_config.name=LlavaScene object_config.name=LlavaObject summary_config.name=Llama3Summary run_folder=${RUN_FOLDER} action_config.name=RNAction


########################################## 
# Method Comparison
##########################################

# RUN_FOLDER=data/vlnce_traj_action_clean/val_unseen

# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_vo_qwn_qwn_qwn action_config.name=VOAction scene_config.name=Qwen2Scene object_config.name=Qwen2Object summary_config.name=Qwen25Summary run_folder=${RUN_FOLDER}

# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_vo_gpt_gpt_qwn action_config.name=VOAction scene_config.name=GPTScene object_config.name=GPTObject summary_config.name=Qwen25Summary run_folder=${RUN_FOLDER}

# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_llavanextvideo speaker_config.name=LlavaNextVideoSpeaker speaker_config.downsample=4 run_folder=${RUN_FOLDER}

# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_qwen25vl speaker_config.name=Qwen25VLSpeaker speaker_config.downsample=1 run_folder=${RUN_FOLDER}

# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_internvideo speaker_config.name=InternVideoSpeaker speaker_config.downsample=1 run_folder=${RUN_FOLDER}


########################################## 
# Ablation Study
# Note: To reduce cost, `reference_alias` is used to set the reference information
##########################################

# RUN_FOLDER=data/vlnce_traj_action_clean/val_unseen

# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_xx_gpt_gpt_qwn action_config.name=VOAction scene_config.name=GPTScene object_config.name=GPTObject summary_config.name=Qwen25Summary run_folder=${RUN_FOLDER} speaker_config.name=AblateNavComposer reference_alias=inst_vo_gpt_gpt_qwn_mid_update ablate_action=True

# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_vo_xxx_gpt_qwn action_config.name=VOAction scene_config.name=GPTScene object_config.name=GPTObject summary_config.name=Qwen25Summary run_folder=${RUN_FOLDER} speaker_config.name=AblateNavComposer reference_alias=inst_vo_gpt_gpt_qwn_mid_update ablate_scene=True

# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_vo_gpt_xxx_qwn action_config.name=VOAction scene_config.name=GPTScene object_config.name=GPTObject summary_config.name=Qwen25Summary run_folder=${RUN_FOLDER} speaker_config.name=AblateNavComposer reference_alias=inst_vo_gpt_gpt_qwn_mid_update ablate_object=True


########################################## 
# Diverse Generation
# Note 1: Add a short sentence to the prompt, reducing redundancy in instructions
# Note 2: In order to comply with computing resources, use bfloat16 precision and resized images for some sources.
#   If the computation is sufficient, `image_resize` and `camera_params` settings can be remove.
##########################################


# PROMPT="As an expert in robotics and vision-and-language navigation research, condense a list of observations and navigation actions into a brief instruction for robot navigation. Focus solely on essential information, omitting unnecessary details. The instruction must be as short as possible while remaining clear and fluent. Rephrase or restructure the content as needed, and do not use a list format. Omit secondary and repeated details to ensure navigation instructions are clear. Avoid any introductory phrases and extra explanations."
# ACTION=VOAction
# SCENE=Qwen2Scene
# OBJECT=Qwen2Object
# SUMMARY=Qwen25Summary
# PRECISION=bfloat16
# GEN_NUM=3

# RUN_FOLDER=data/various_traj_clean/hm3d
# torchrun --master_port ${PORT} main.py --config_file main.yaml \
#     generate_alias=inst_navcomposer \
#     action_config.name=${ACTION} \
#     scene_config.name=${SCENE} \
#     object_config.name=${OBJECT} \
#     summary_config.name=${SUMMARY} \
#     summary_config.qwen25_config.system_prompt="${PROMPT}" \
#     generate_num=${GEN_NUM} \
#     skip_exist=True \
#     run_precision=${PRECISION} \
#     run_folder=${RUN_FOLDER}


# RUN_FOLDER=data/various_traj_clean/sun3d
# torchrun --master_port ${PORT} main.py --config_file main.yaml \
#     generate_alias=inst_navcomposer \
#     action_config.name=${ACTION} \
#     scene_config.name=${SCENE} \
#     object_config.name=${OBJECT} \
#     summary_config.name=${SUMMARY} \
#     summary_config.qwen25_config.system_prompt="${PROMPT}" \
#     generate_num=${GEN_NUM} \
#     skip_exist=True \
#     run_precision=${PRECISION} \
#     run_folder=${RUN_FOLDER}

# RUN_FOLDER=data/various_traj_clean/scenenet
# torchrun --master_port ${PORT} main.py --config_file main.yaml \
#     generate_alias=inst_navcomposer \
#     action_config.name=${ACTION} \
#     scene_config.name=${SCENE} \
#     object_config.name=${OBJECT} \
#     summary_config.name=${SUMMARY} \
#     summary_config.qwen25_config.system_prompt="${PROMPT}" \
#     generate_num=${GEN_NUM} \
#     skip_exist=True \
#     run_precision=${PRECISION} \
#     run_folder=${RUN_FOLDER}

# RUN_FOLDER=data/various_traj_clean/tumrgbd
# torchrun --master_port ${PORT} main.py --config_file main.yaml \
#     generate_alias=inst_navcomposer \
#     action_config.name=${ACTION} \
#     scene_config.name=${SCENE} \
#     object_config.name=${OBJECT} \
#     summary_config.name=${SUMMARY} \
#     summary_config.qwen25_config.system_prompt="${PROMPT}" \
#     speaker_config.image_resize=480 \
#     action_config.vo_config.camera_params=[640.0,480.0,320.0,240.0,320.0,240.0] \
#     generate_num=${GEN_NUM} \
#     run_precision=${PRECISION} \
#     run_folder=${RUN_FOLDER}


# RUN_FOLDER=data/various_traj_clean/scannet
# torchrun --master_port ${PORT} main.py --config_file main.yaml \
#     generate_alias=inst_navcomposer \
#     action_config.name=${ACTION} \
#     scene_config.name=${SCENE} \
#     object_config.name=${OBJECT} \
#     summary_config.name=${SUMMARY} \
#     summary_config.qwen25_config.system_prompt="${PROMPT}" \
#     speaker_config.image_resize=968 \
#     action_config.vo_config.camera_params=[1296.0,968.0,648.0,484.0,648.0,484.0] \
#     generate_num=${GEN_NUM} \
#     run_precision=${PRECISION} \
#     run_folder=${RUN_FOLDER}


# RUN_FOLDER=data/various_traj_clean/diode
# torchrun --master_port ${PORT} main.py --config_file main.yaml \
#     generate_alias=inst_navcomposer \
#     action_config.name=${ACTION} \
#     scene_config.name=${SCENE} \
#     object_config.name=${OBJECT} \
#     summary_config.name=${SUMMARY} \
#     summary_config.qwen25_config.system_prompt="${PROMPT}" \
#     speaker_config.image_resize=300 \
#     action_config.vo_config.camera_params=[400.0,300.0,200.0,150.0,200.0,150.0] \
#     action_config.vo_config.angle_threshold=5 \
#     generate_num=${GEN_NUM} \
#     run_precision=${PRECISION} \
#     run_folder=${RUN_FOLDER}


# RUN_FOLDER=data/various_traj_clean/kitti
# torchrun --master_port ${PORT} main.py --config_file main.yaml \
#     generate_alias=inst_navcomposer \
#     action_config.name=${ACTION} \
#     scene_config.name=${SCENE} \
#     object_config.name=${OBJECT} \
#     summary_config.name=${SUMMARY} \
#     summary_config.qwen25_config.system_prompt="${PROMPT}" \
#     speaker_config.image_resize=375 \
#     action_config.vo_config.camera_params=[1240.0,375.0,620.0,187.0,620.0,187.0] \
#     action_config.vo_config.angle_threshold=5 \
#     generate_num=${GEN_NUM} \
#     run_precision=${PRECISION} \
#     run_folder=${RUN_FOLDER}

# RUN_FOLDER=data/various_traj_clean/12scenes
# torchrun --master_port ${PORT} main.py --config_file main.yaml \
#     generate_alias=inst_navcomposer \
#     action_config.name=${ACTION} \
#     scene_config.name=${SCENE} \
#     object_config.name=${OBJECT} \
#     summary_config.name=${SUMMARY} \
#     summary_config.qwen25_config.system_prompt="${PROMPT}" \
#     speaker_config.image_resize=968 \
#     action_config.vo_config.camera_params=[1296.0,968.0,648.0,484.0,648.0,484.0] \
#     generate_num=${GEN_NUM} \
#     run_precision=${PRECISION} \
#     run_folder=${RUN_FOLDER}

# RUN_FOLDER=data/various_traj_clean/grutopia
# torchrun --master_port ${PORT} main.py --config_file main.yaml \
#     generate_alias=inst_navcomposer \
#     action_config.name=${ACTION} \
#     scene_config.name=${SCENE} \
#     object_config.name=${OBJECT} \
#     summary_config.name=${SUMMARY} \
#     summary_config.qwen25_config.system_prompt="${PROMPT}" \
#     speaker_config.image_resize=256 \
#     action_config.vo_config.camera_params=[256.0,256.0,128.0,128.0,128.0,128.0] \
#     generate_num=${GEN_NUM} \
#     run_precision=${PRECISION} \
#     run_folder=${RUN_FOLDER}


# RUN_FOLDER=data/various_traj_clean/navtj
# torchrun --master_port ${PORT} main.py --config_file main.yaml \
#     generate_alias=inst_navcomposer \
#     action_config.name=${ACTION} \
#     scene_config.name=${SCENE} \
#     object_config.name=${OBJECT} \
#     summary_config.name=${SUMMARY} \
#     summary_config.qwen25_config.system_prompt="${PROMPT}" \
#     speaker_config.image_resize=480 \
#     action_config.vo_config.camera_params=[640.0,480.0,320.0,240.0,320.0,240.0] \
#     generate_num=${GEN_NUM} \
#     run_precision=${PRECISION} \
#     run_folder=${RUN_FOLDER}



########################################## 
# Style Control
##########################################

# ACTION=VOAction
# SCENE=GPTScene
# OBJECT=GPTObject
# SUMMARY=Qwen25Summary
# PRECISION=float32
# GEN_NUM=3
# RUN_FOLDER=data/style_test

# ALIAS=inst_style_diversified
# torchrun --master_port ${PORT} main.py --config_file main.yaml \
#     generate_alias=${ALIAS} \
#     action_config.name=${ACTION} \
#     scene_config.name=${SCENE} \
#     object_config.name=${OBJECT} \
#     summary_config.name=${SUMMARY} \
#     generate_num=${GEN_NUM} \
#     run_precision=${PRECISION} \
#     speaker_config.diversify_level_action=1 \
#     speaker_config.diversify_level_element=1 \
#     run_folder=${RUN_FOLDER}

# ALIAS=inst_style_undiversified
# ACTION=VOAction
# SCENE=Qwen2Scene
# OBJECT=Qwen2Object
# SUMMARY=Qwen25Summary
# torchrun --master_port ${PORT} main.py --config_file main.yaml \
#     generate_alias=${ALIAS} \
#     action_config.name=${ACTION} \
#     scene_config.name=${SCENE} \
#     object_config.name=${OBJECT} \
#     summary_config.name=${SUMMARY} \
#     summary_config.qwen25_config.temperature=0.001 \
#     generate_num=${GEN_NUM} \
#     run_precision=${PRECISION} \
#     speaker_config.diversify_level_action=0 \
#     speaker_config.use_enter_leave=False \
#     speaker_config.diversify_level_element=0 \
#     run_folder=${RUN_FOLDER}

# ALIAS=inst_style_30words
# ACTION=VOAction
# SCENE=Qwen2Scene
# OBJECT=Qwen2Object
# SUMMARY=Qwen25Summary
# PROMPT="As an expert in robotics and vision-and-language navigation research, condense a list of observations and navigation actions into a brief instruction for robot navigation. Focus solely on essential information, omitting unnecessary details. The instruction must be as short as possible while remaining clear and fluent. Rephrase or restructure the content as needed, and do not use a list format. Omit secondary and repeated details to ensure the navigation instruction is exactly 30 words. Avoid any introductory phrases and extra explanations."
# torchrun --master_port ${PORT} main.py --config_file main.yaml \
#     generate_alias=${ALIAS} \
#     action_config.name=${ACTION} \
#     scene_config.name=${SCENE} \
#     object_config.name=${OBJECT} \
#     summary_config.name=${SUMMARY} \
#     summary_config.qwen25_config.system_prompt="${PROMPT}" \
#     generate_num=${GEN_NUM} \
#     run_precision=${PRECISION} \
#     random_seed=44 \
#     run_folder=${RUN_FOLDER}

# ALIAS=inst_style_polite
# PROMPT="As a very polite user, summarize a list of observations and navigation actions into a brief instruction for robot navigation. Focus solely on essential information, omitting unnecessary details. The tune of the instruction must be polite, formal, indirect, and collaborative while remaining clear and fluent. Rephrase or restructure the content as needed, and do not use a list format. Avoid any introductory phrases and extra explanations."
# torchrun --master_port ${PORT} main.py --config_file main.yaml \
#     generate_alias=${ALIAS} \
#     action_config.name=${ACTION} \
#     scene_config.name=${SCENE} \
#     object_config.name=${OBJECT} \
#     summary_config.name=${SUMMARY} \
#     summary_config.qwen25_config.system_prompt="${PROMPT}" \
#     generate_num=${GEN_NUM} \
#     run_precision=${PRECISION} \
#     run_folder=${RUN_FOLDER}

# ALIAS=inst_style_reverie
# PROMPT="As a lazy user, imagine a one-sentence robot task according to a list of observations and navigation actions. Focus solely on the object and location where the robot stops, omitting unnecessary details. Avoid any introductory phrases and extra explanations. Please use a clear verb to instruct the robot to engage with the final object and serve the user (do not directly use the word \"interact\" or \"serve\")."
# torchrun --master_port ${PORT} main.py --config_file main.yaml \
#     generate_alias=${ALIAS} \
#     action_config.name=${ACTION} \
#     scene_config.name=${SCENE} \
#     object_config.name=${OBJECT} \
#     summary_config.name=${SUMMARY} \
#     summary_config.qwen25_config.system_prompt="${PROMPT}" \
#     generate_num=${GEN_NUM} \
#     run_precision=${PRECISION} \
#     run_folder=${RUN_FOLDER}



########################################## 
# Time Cost
##########################################

# RUN_FOLDER=data/vlnce_traj_action_exp/val_seen
# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_vo_qwn_qwn_qwn_timetest action_config.name=VOAction scene_config.name=Qwen2Scene object_config.name=Qwen2Object summary_config.name=Qwen25Summary run_folder=${RUN_FOLDER} action_config.vo_config.feature=ORB run_precision=bloat16 speaker_config.time_file=time_info_vo_orb_bf16.jsonl
# RUN_FOLDER=data/vlnce_traj_action_exp/val_unseen
# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_vo_qwn_qwn_qwn_timetest action_config.name=VOAction scene_config.name=Qwen2Scene object_config.name=Qwen2Object summary_config.name=Qwen25Summary run_folder=${RUN_FOLDER} action_config.vo_config.feature=ORB run_precision=bloat16 speaker_config.time_file=time_info_vo_orb_bf16.jsonl

# RUN_FOLDER=data/vlnce_traj_action_exp/val_seen
# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_rn_llv_llv_llm_timetest scene_config.name=LlavaScene object_config.name=LlavaObject summary_config.name=Llama3Summary run_folder=${RUN_FOLDER} action_config.name=RNAction speaker_config.name=NavComposerWithTime speaker_config.time_file=time_info_rn.jsonl
# RUN_FOLDER=data/vlnce_traj_action_exp/val_unseen
# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_rn_llv_llv_llm_timetest scene_config.name=LlavaScene object_config.name=LlavaObject summary_config.name=Llama3Summary run_folder=${RUN_FOLDER} action_config.name=RNAction speaker_config.name=NavComposerWithTime speaker_config.time_file=time_info_rn.jsonl

# RUN_FOLDER=data/vlnce_traj_action_exp/val_seen
# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_dn_llv_llv_llm_timetest scene_config.name=LlavaScene object_config.name=LlavaObject summary_config.name=Llama3Summary run_folder=${RUN_FOLDER} action_config.name=Dinov2Action speaker_config.name=NavComposerWithTime speaker_config.time_file=time_info_dn.jsonl
# RUN_FOLDER=data/vlnce_traj_action_exp/val_unseen
# torchrun --master_port ${PORT} main.py --config_file main.yaml generate_alias=inst_dn_llv_llv_llm_timetest scene_config.name=LlavaScene object_config.name=LlavaObject summary_config.name=Llama3Summary run_folder=${RUN_FOLDER} action_config.name=Dinov2Action speaker_config.name=NavComposerWithTime speaker_config.time_file=time_info_dn.jsonl