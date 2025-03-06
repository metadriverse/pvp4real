#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)



## Loop over each GPU
#for i in {0..7}
#do
#    CUDA_VISIBLE_DEVICES=$i \
#    nohup python pvp/experiments/metadrive/train_pvp_metadrive_fakehuman.py \
#    --exp_name=0910_pvporig_noadap_withbcloss_bs1024_onlineupdate \
#    --bc_loss_weight=0.0 \
#    --batch_size=1024 \
#    --adaptive_batch_size=False \
#    --wandb \
#    --wandb_project=pvp2024 \
#    --wandb_team=drivingforce \
#    --seed=${seeds[$i]} \
#    --free_level=0.95 \
#    --save_freq=10000 \
#    > "seed${seeds[$i]}.log" 2>&1 &
#done


## Loop over each GPU
#for i in {0..7}
#do
#    CUDA_VISIBLE_DEVICES=$i \
#    nohup python pvp/experiments/metadrive/train_pvp_metadrive_fakehuman.py \
#    --exp_name=0915_hgdagger \
#    --wandb \
#    --wandb_project=pvp2024 \
#    --wandb_team=drivingforce \
#    --only_bc_loss=True \
#    --bc_loss_weight=1.0 \
#    --seed=${seeds[$i]} \
#    > "0915-1_seed${seeds[$i]}.log" 2>&1 &
#done
#
#
#
## Loop over each GPU
#for i in {0..7}
#do
#    CUDA_VISIBLE_DEVICES=$i \
#    nohup python pvp/experiments/metadrive/train_pvp_metadrive_fakehuman.py \
#    --exp_name=0915_bcandploss \
#    --wandb \
#    --wandb_project=pvp2024 \
#    --wandb_team=drivingforce \
#    --only_bc_loss=False \
#    --bc_loss_weight=1.0 \
#    --seed=${seeds[$i]} \
#    > "0915-2_seed${seeds[$i]}.log" 2>&1 &
#done


# Loop over each GPU
for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_pvp_metadrive_fakehuman.py \
    --exp_name=0915_BCOnPureExpertData \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --only_bc_loss=False \
    --bc_loss_weight=1.0 \
    --free_level=-10000 \
    --seed=${seeds[$i]} \
    > "0915-2_seed${seeds[$i]}.log" 2>&1 &
done
