# Differentially Private Decentralized Deep Learning with Consensus Algorithms

Jasmine Bayrooti, Zhan Gao, Amanda Prorok

## 0) Background

Fully decentralized learning can enable greater-scale applications via local communication over an underlying graph. However, most algorithms share parameters directly with other agents, thus presenting a potential privacy risk if agents are untrustworthy. In this project, we present differentially private algorithms for fully decentralized learning that secures agents' private datasets from others in the system. Our algorithms achieve good performance on standard image classification tasks even with low privacy budgets (i.e., strong privacy guarantees).

## 1) Install Dependencies

We used the following PyTorch libraries for CUDA 12; you may have to adapt for your own CUDA version:

```console
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install dependencies:
```console
conda install scipy
pip install -r requirements.txt
```

## 2) Running experiments

Start by running
```console
source init_env.sh
```

Run experiments for the different datasets as follows:

```console
python run.py config/mnist/dsgd/dp_e_100.yaml --gpu-device 0
```

This command trains `N` agents using DP-DSGD on the MNIST classification task using GPU 0. The `config` directory holds configuration files for the different experiments, specifying the hyperparameters used for each experiment. The first field in every config file is `exp_base`, which specifies the base directory to save experiment outputs. You should change this for your own setup and also update the dataset paths in `src/datasets/dataset_name.py`. The experiments include standard central SGD, DSGD, DSGT, and DiNNO training as well as DP-SGD and our algorithms DP-DSGD, DP-DSGT, and DP-DiNNO.

Training curves and other metrics are logged using [wandb.ai](wandb.ai).
