#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)

# Loop over each GPU
for i in {0..3}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_pvp_metadrive_fakehuman.py \
    --exp_name=0706_bc_fulltakeover \
    --bc_loss_weight=1.0 \
    --only_bc_loss="True" \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=-10000.0 \
    --adaptive_batch_size=True \
    --save_freq=10000 \
    > "seed${seeds[$i]}.log" 2>&1 &
done
