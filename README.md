# [Proxy Value Propagation for Real (PVP4Real)](https://metadriverse.github.io/pvp4real/)

<h3 align="center"><b>ICRA 2025</b></h3>

<p align="center">
  <img src="PVP4Real_Teaser.png" alt="PVP4Real" width="100%">
</p>

Official release for the code used in paper: *Data-Efficient Learning from Human Interventions for Mobile Robots*. Compared to [PVP repo](https://metadriverse.github.io/pvp/), 
we include the simulated human experiments in this repo.

[**Webpage**](https://metadriverse.github.io/pvp4real/) | 
[**Code**](https://github.com/metadriverse/pvp4real) |
[**Paper** (Coming)](https://arxiv.org/pdf/2503.04969)




## Installation

```bash
# Clone the code to local machine
git clone https://github.com/pengzhenghao/pvp4real
cd pvp4real

# Create Conda environment
conda create -n pvp4real python=3.7
conda activate pvp4real

# Install dependencies
pip install setuptools==65.5.0 pip==21  # Fix gym installation issue
pip install wheel==0.38.0  # Fix gym installation issue
pip install -r requirements.txt
pip install -e .
pip install torch

# Install MetaDrive
# In case you need it, the MetaDrive commit we ran on is: c29cc37d30158fe70d963647b6c80dc814248f60
# Using latest MetaDrive should work:
pip install git+https://github.com/metadriverse/metadrive.git

```


## Launch Experiments

You can find the scripts to launch a batch of experiments in the [`./scripts`](./scripts) folder. Call them by:

```bash
bash scripts/metadrive_simhuman_pvp4real.sh
bash scripts/metadrive_simhuman_pvp.sh
bash scripts/metadrive_simhuman_haco.sh
bash scripts/metadrive_simhuman_ppo.sh
bash scripts/metadrive_simhuman_td3.sh
bash scripts/metadrive_simhuman_bc.sh
bash scripts/metadrive_simhuman_eil.sh
bash scripts/metadrive_simhuman_hgdagger.sh
```


Here is the example on how to launch PVP4Real:

```bash
python pvp/experiments/metadrive/train_pvp_metadrive_fakehuman.py \
--exp_name="pvp4real" \
--bc_loss_weight=1.0
```



---

Tips:

We evaluate frequently to get beautiful learning curves. You can change the evaluation frequency when debugging:

```python
model.learn(
    ...
    eval_freq=150,  # <<< Change this
    n_eval_episodes=50,  # <<< Change this
)
```

You can use wandb to monitor the training process:

```bash
nohup python pvp/experiments/metadrive/train_pvp_metadrive_fakehuman.py \
--exp_name=${EXP_NAME} \
--wandb \
--wandb_project=pvp2024 \
--wandb_team=drivingforce
```

> [!NOTE]  
> The BC, HGDagger and EIL experiments are not strictly following the paper.
> The BC data is not collected before training but instead collected and is expanding
> during the training process. 

## 📎 References

```latex
Coming soon...
```

