#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)





# Loop over each GPU
for i in {0..3}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_pvp_metadrive_fakehuman_transformer.py \
    --exp_name=0711_pvp_bcloss1_nogru_larger_res_sanitycheck \
    --bc_loss_weight=1.0 \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --adaptive_batch_size=True \
    --save_freq=10000 \
    > "seed${seeds[$i]}.log" 2>&1 &
done

