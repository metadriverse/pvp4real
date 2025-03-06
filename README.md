# [Proxy Value Propagation for Real (PVP4Real)](https://metadriverse.github.io/pvp4real/)

<h3><b>ICLR</b></h3>

Official release for the code used in paper: *Learning from Active Human Involvement through Proxy Value Propagation*

Compared to [PVP repo](https://metadriverse.github.io/pvp/), we include the simulated human experiments in this repo.

[**Webpage**](https://metadriverse.github.io/pvp4real/) | 
[**Code**](https://github.com/metadriverse/pvp4real) |
[**Paper** (Coming)](#)

TODO: Teaser figure.

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




Tips:

We evaluate frequently to get beautiful learning curves. You can change the evaluation frequency:

```python
model.learn(
    ...
    eval_freq=150,  # <<< Change this
    n_eval_episodes=50,  # <<< Change this
)
```

## 📎 References

```latex
Coming soon...
```

