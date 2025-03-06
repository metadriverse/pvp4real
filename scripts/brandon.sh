#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)


#for i in {0..2}
#do
#    CUDA_VISIBLE_DEVICES=$i \
#    nohup python pvp/experiments/metadrive/train_metadrive_multigoal_sac_RealMultigoalEnv.py \
#    --ckpt=runs/0710_multigoal_sac_TRAINED_EXPERT/rl_model_3000000_steps \
#    --seed=${seeds[$i]} \
#    --exp_name="0711_multigoal_sac_RealMultigoalEnv" \
#    --wandb \
#    > "brandon-seed${seeds[$i]}.log" 2>&1 &
#done


for i in {6..7}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_metadrive_multigoal_sac_RealMultigoalEnv.py \
    --ckpt=runs/0711_multigoal_sac_RealMultigoalEnv/0711_multigoal_sac_RealMultigoalEnv_seed100_2024-07-11_17-22-18/models/rl_model_3000000_steps \
    --seed=${seeds[$i]} \
    --exp_name="0806_multigoal_sac_RealMultigoalEnv_2ndfinetune" \
    --wandb \
    > "brandon-seed${seeds[$i]}.log" 2>&1 &
done



## Loop over each GPU
#for i in {4..5}
#do
#    CUDA_VISIBLE_DEVICES=$i \
#    nohup python pvp/experiments/metadrive/train_metadrive_multigoal_ppo.py \
#    --seed=${seeds[$i]} \
#    --exp_name="0708_ppo_traffic=0.06_safemdenv" \
#    --traffic_density=0.06 \
#    --wandb \
#    --lane_line_detector=0 \
#    > "brandon-seed${seeds[$i]}.log" 2>&1 &
#done

