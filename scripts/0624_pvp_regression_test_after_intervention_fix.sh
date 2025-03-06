#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)





# Loop over each GPU
for i in {0..1}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_pvp_metadrive_fakehuman.py \
    --exp_name=0624_pvp_bcloss0.1 \
    --bc_loss_weight=0.1 \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --adaptive_batch_size=True \
    --save_freq=10000 \
    > "seed${seeds[$i]}.log" 2>&1 &
done



# Loop over each GPU
for i in {2..3}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_pvp_metadrive_fakehuman.py \
    --exp_name=0624_pvp_bcloss0.5 \
    --bc_loss_weight=0.5 \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --adaptive_batch_size=True \
    --save_freq=10000 \
    > "seed11${seeds[$i]}.log" 2>&1 &
done




# Loop over each GPU
for i in {4..5}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_pvp_metadrive_fakehuman.py \
    --exp_name=0624_pvp_bcloss1.0 \
    --bc_loss_weight=1.0 \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --adaptive_batch_size=True \
    --save_freq=10000 \
    > "seed22${seeds[$i]}.log" 2>&1 &
done


