"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""
import copy
import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from models.care_encoder_model import CARE_ENCODER

logger = logging.getLogger(__name__)



def _gaussian_logprob(noise, log_std):
    """Compute the gaussian log probability.

    Args:
        noise (TensorType):
        log_std (TensorType): [description]

    Returns:
        TensorType: log-probaility of the sample.
    """
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def _squash(mu, pi, log_pi):
    """Apply squashing function.
        See appendix C from https://arxiv.org/pdf/1812.05905.pdf.

    Args:
        mu ([TensorType]): mean of the gaussian distribution.
        pi ([TensorType]): sample from the gaussian distribution.
        log_pi ([TensorType]): log probability.

    Returns:
        Tuple[TensorType, TensorType, TensorType]: tuple of
            (squashed mean of the gaussian, squashed sample from the gaussian,
                squashed  log-probability of the sample)
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi

class GPTConfig:
    """base GPT config, params common to all GPT versions"""

    def __init__(self, state_size, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.state_size = state_size
        for k, v in kwargs.items():
            setattr(self, k, v)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class GPT_MLP(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config, model_type="actor"):
        super().__init__()

        self.config = config

        # self.block_size = config.block_size
        self.model_type = model_type
        self.state_size = config.local_obs_dim
        self.act_dim = config.action_dim
        self.task_emb_dim = config.task_emb_dim
        self.input_dim = self.state_size + self.task_emb_dim

        self.care_encoder = CARE_ENCODER(self.state_size, self.state_size, self.task_emb_dim)

        self._layer_N = 2
        self.feature_norm = nn.LayerNorm(self.input_dim)
        self.feature_norm_critic = nn.LayerNorm(self.input_dim + self.act_dim)
        self.log_std_bounds = [-20, 2]

        active_func = nn.ReLU()
        init_method = nn.init.orthogonal_
        gain = nn.init.calculate_gain("relu")

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(self.input_dim, 64)), active_func, nn.LayerNorm(64)
        )
        self.fc1_critic = nn.Sequential(
            init_(nn.Linear(self.input_dim + self.act_dim, 64)), active_func, nn.LayerNorm(64)
        )
        self.fc_h = nn.Sequential(
            init_(nn.Linear(64, 64)), active_func, nn.LayerNorm(64)
        )
        self.fc2 = get_clones(self.fc_h, self._layer_N)

        if model_type == "actor":
            self.head = nn.Linear(64, config.action_dim*2, bias=False)
        elif model_type == "critic":
            self.head = nn.Linear(64, 1, bias=False)
        else:
            raise NotImplementedError

    # def get_block_size(self):
    #     return self.block_size

    def configure_optimizers(self, train_config, lr):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer


    def encode(self, states, pre_actions, rtgs=None, task_id=0, timesteps=None):
        # (n_threads*n_agent, context_lengrh, dim)
        state_care = self.care_encoder(states, task_id)  # (48.79)  (48.28)  --> (48, 107)
        x = state_care.view(-1, self.input_dim)  # torch.Size([48, 79])
        x = self.feature_norm(x)   # torch.Size([48, 79])
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        x = self.head(x)
        mu_and_log_std = x
        mu, log_std = mu_and_log_std.chunk(2, dim=-1)  # 48,10 48  10
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds  # [-20, 2]
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()
        noise = torch.randn_like(mu)
        pi = mu + noise * std
        log_pi = _gaussian_logprob(noise, log_std)

        mu, pi, log_pi = _squash(mu, pi, log_pi)

        pi = pi.view(-1, 1, pi.size(-1))
        logits_tuple = (mu, log_pi, log_std)

        return pi, logits_tuple

    # state, action, and return
    def forward(self, states, policy_actions, rtgs=None, timesteps=None, task_id=0):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, block_size, 1)
        state_care = self.care_encoder(states, task_id)  # (48.79)  (48.28)  --> (48, 107)
        state_care_action = torch.cat((state_care, policy_actions), dim=-1)
        x = state_care_action.view(-1, self.input_dim+self.act_dim)
        x = self.feature_norm_critic(x)
        x = self.fc1_critic(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        x = self.head(x)
        logits = x.view(-1, 1, x.size(-1))

        return logits
