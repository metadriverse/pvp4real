#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)


filename=$(basename "$0")
extension="${filename##*.}"

EXP_NAME="${filename%.*}"

# Loop over each GPU
for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_eil_metadrive_fakehuman.py \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --exp_name=${EXP_NAME} \
    --seed=${seeds[$i]} \
    > ${EXP_NAME}_seed${seeds[$i]}.log 2>&1 &
done
