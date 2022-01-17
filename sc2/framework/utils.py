import copy
import random
import numpy as np
import torch
from torch.nn import functional as F
from gym.spaces.discrete import Discrete

# from sc2.toy_data import toy_example


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample(
    m3,
    critic_model,
    critic_model_2,
    state,
    obs,
    sample=False,
    actions=None,
    rtgs=None,
    timesteps=None,
    available_actions=None,
    task_ids=None,
    use_task_embedding=False,
):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    # if torch.cuda.is_available():
    #     block_size = model.module.module.get_block_size()
    # else:
    #     block_size = model.get_block_size()

    m3 = m3.module if hasattr(m3.module, "module") else m3

    critic_model_2.eval()
    m3.eval()
    critic_model.eval()

    # x: (batch_size, context_length, dim)
    # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
    obs_cond = obs[:, -1]
    state_cond = obs[:, -1]
    if actions is not None:
        actions = actions[:, -1]
    rtgs = rtgs[:, -1]
    timesteps = timesteps[:, -1]
    if task_ids is not None:
        task_ids = task_ids[:, -1]

    # obs_cond = (
    #     obs if obs.size(1) <= block_size // 3 else obs[:, -block_size // 3 :]
    # )  # crop context if needed
    # state_cond = (
    #     state if state.size(1) <= block_size // 3 else state[:, -block_size // 3 :]
    # )  # crop context if needed
    # if actions is not None:
    #     actions = (
    #         actions
    #         if actions.size(1) <= block_size // 3
    #         else actions[:, -block_size // 3 :]
    #     )  # crop context if needed
    # rtgs = (
    #     rtgs if rtgs.size(1) <= block_size // 3 else rtgs[:, -block_size // 3 :]
    # )  # crop context if needed
    # timesteps = (
    #     timesteps
    #     if timesteps.size(1) <= block_size // 3
    #     else timesteps[:, -block_size // 3 :]
    # )  # crop context if needed
    # if task_ids is not None:
    #     task_ids = (
    #         task_ids
    #         if task_ids.size(1) <= block_size // 3
    #         else task_ids[:, -block_size // 3 :]
    #     )
    # logits, _, _ = m3.module.encode(
    #     obs_cond, pre_actions=actions, rtgs=rtgs, timesteps=timesteps, task_id=task_ids
    # )
    logits, _ = m3.module.encode(
        obs_cond, pre_actions=actions, rtgs=rtgs, timesteps=timesteps, task_id=task_ids
    )
    # pluck the logits at the final step and scale by temperature
    logits = logits[:, -1, :]
    # apply softmax to convert to probabilities
    if available_actions is not None:
        logits[available_actions == 0] = -1e10
    probs = F.softmax(logits, dim=-1) ##

    if sample:
        a = torch.multinomial(probs, num_samples=1)   ## torch.Size([48, 1])
    else:
        _, a = torch.topk(probs, k=1, dim=-1)

    # v = critic_model.module.module(state_cond, actions, rtgs, timesteps, task_ids).detach()  # 48.1.1  ###TODO: RJQ 此处应该改为q function
    # v = v[:, -1, :]  # 48.1
    policy_actions = logits
    q1 = critic_model.module.module(state_cond, policy_actions, rtgs, timesteps, task_ids).detach() # 48.1.1
    q2 = critic_model_2.module.module(state_cond, policy_actions, rtgs, timesteps, task_ids).detach()
    q1 = q1[:, -1, :]
    q2 = q2[:, -1, :]

    return a, (q1, q2, logits)


def get_dim_from_space(space):
    if isinstance(space[0], Discrete):
        return space[0].n
    elif isinstance(space[0], list):
        return space[0][0]


def padding_obs(obs, target_dim):
    len_obs = np.shape(obs)[-1]
    if len_obs > target_dim:
        print("target_dim (%s) too small, obs dim is %s." % (target_dim, len(obs)))
        raise NotImplementedError
    elif len_obs < target_dim:
        padding_size = target_dim - len_obs
        if isinstance(obs, list):
            obs = np.array(copy.deepcopy(obs))
            padding = np.zeros(padding_size)
            obs = np.concatenate((obs, padding), axis=-1).tolist()
        elif isinstance(obs, np.ndarray):
            obs = copy.deepcopy(obs)
            shape = np.shape(obs)
            padding = np.zeros((shape[0], shape[1], padding_size))
            obs = np.concatenate((obs, padding), axis=-1)
        else:
            print("unknwon type %s." % type(obs))
            raise NotImplementedError
    return obs


def padding_ava(ava, target_dim):
    len_ava = np.shape(ava)[-1]
    if len_ava > target_dim:
        print("target_dim (%s) too small, ava dim is %s." % (target_dim, len(ava)))
        raise NotImplementedError
    elif len_ava < target_dim:
        padding_size = target_dim - len_ava
        if isinstance(ava, list):
            ava = np.array(copy.deepcopy(ava), dtype=np.long)
            padding = np.zeros(padding_size, dtype=np.long)
            ava = np.concatenate((ava, padding), axis=-1).tolist()
        elif isinstance(ava, np.ndarray):
            ava = copy.deepcopy(ava)
            shape = np.shape(ava)
            padding = np.zeros((shape[0], shape[1], padding_size), dtype=np.long)
            ava = np.concatenate((ava, padding), axis=-1)
        else:
            print("unknwon type %s." % type(ava))
            raise NotImplementedError
    return ava
