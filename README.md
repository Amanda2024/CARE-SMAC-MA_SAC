# Multi-task Multi-agent Soft Actor Critic for SMAC

## Overview

The CARE formulti-task: [Multi-Task Reinforcement Learning with Context-based Representations](https://arxiv.org/pdf/2102.06177.pdf)
The SAC for multi-agent: [Soft actor-critic algorithms and applications](https://arxiv.org/pdf/1812.05905.pdf)


## Installation

```shell
conda create -n py3 python==3.6 -y
conda activate py3

cd .

pip install -r requirements.txt
```

## Run the experiments :

```shell
cd sc2/
python run_madt_sc2.py
```