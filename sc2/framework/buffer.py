import torch
import numpy as np
import copy, glob
from torch.utils.data import Dataset
from .utils import padding_obs, padding_ava


class StateActionReturnDataset(Dataset):
    def __init__(
        self,
        global_state,
        local_obs,
        block_size,
        actions,
        done_idxs,
        rewards,
        avas,
        q_values_1,
        q_values_2,
        logits,
        rtgs,
        rets,
        advs,
        timesteps,
        task_ids,
    ):
        self.block_size = block_size
        self.global_state = global_state
        self.local_obs = local_obs
        self.actions = actions
        self.done_idxs = done_idxs
        self.rewards = rewards
        self.avas = avas
        self.q_values_1 = q_values_1
        self.q_values_2 = q_values_2
        self.logits = logits
        self.rtgs = rtgs
        self.rets = rets
        self.advs = advs
        self.timesteps = timesteps
        self.task_ids = task_ids

    def __len__(self):
        # return len(self.global_state) - self.block_size
        return len(self.global_state)

    def stats(self):
        print(
            "max episode length: ",
            max(np.array(self.done_idxs[1:]) - np.array(self.done_idxs[:-1])),
        )
        print(
            "min episode length: ",
            min(np.array(self.done_idxs[1:]) - np.array(self.done_idxs[:-1])),
        )
        print("max rtgs: ", max(self.rtgs))
        print(
            "aver episode rtgs: ", np.mean([self.rtgs[i] for i in self.done_idxs[:-1]])
        )

    @property
    def max_rtgs(self):
        return max(self.rtgs)[0]

    def __getitem__(self, idx):
        context_length = self.block_size // 3
        done_idx = idx + context_length
        for i in self.done_idxs:
            if i > idx:  # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - context_length
        states = torch.tensor(
            np.array(self.global_state[idx:done_idx]), dtype=torch.float32
        )
        obss = torch.tensor(np.array(self.local_obs[idx:done_idx]), dtype=torch.float32)

        if done_idx in self.done_idxs:
            next_states = (
                [np.zeros_like(self.global_state[idx]).tolist()]
                + self.global_state[idx + 1 : done_idx]
                + [np.zeros_like(self.global_state[idx]).tolist()]
            )
            next_states.pop(0)
            next_obs = (
                [np.zeros_like(self.local_obs[idx]).tolist()]
                + self.local_obs[idx + 1 : done_idx]
                + [np.zeros_like(self.local_obs[idx]).tolist()]
            )
            next_obs.pop(0)
            next_rtgs = (
                [np.zeros_like(self.rtgs[idx]).tolist()]
                + self.rtgs[idx + 1 : done_idx]
                + [np.zeros_like(self.rtgs[idx]).tolist()]
            )
            next_rtgs.pop(0)
        else:
            next_states = self.global_state[idx + 1 : done_idx + 1]
            next_obs = self.local_obs[idx + 1 : done_idx + 1]
            next_rtgs = self.rtgs[idx + 1 : done_idx + 1]
        next_states = torch.tensor(next_states, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        next_rtgs = torch.tensor(next_rtgs, dtype=torch.float32)

        if idx == 0 or idx in self.done_idxs:
            pre_actions = [[0]] + self.actions[idx : done_idx - 1]
        else:
            pre_actions = self.actions[idx - 1 : done_idx - 1]
        pre_actions = torch.tensor(pre_actions, dtype=torch.long)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long)

        rewards = torch.tensor(self.rewards[idx:done_idx], dtype=torch.float32)
        avas = torch.tensor(self.avas[idx:done_idx], dtype=torch.long)
        q_values_1 = torch.tensor(self.q_values_1[idx:done_idx], dtype=torch.float32)
        q_values_2 = torch.tensor(self.q_values_2[idx:done_idx], dtype=torch.float32)
        logits = torch.tensor(self.logits[idx:done_idx], dtype=torch.float32)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32)
        rets = torch.tensor(self.rets[idx:done_idx], dtype=torch.float32)
        advs = torch.tensor(self.advs[idx:done_idx], dtype=torch.float32)
        # timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64)
        timesteps = torch.tensor(self.timesteps[idx:done_idx], dtype=torch.int64)
        # for the multi-task setting
        task_ids = torch.tensor(self.task_ids[idx:done_idx], dtype=torch.int64)

        dones = torch.zeros_like(rewards)
        if done_idx in self.done_idxs:
            dones[-1][0] = 1

        return (
            states,
            obss,
            actions,
            rewards,
            avas,
            q_values_1,
            q_values_2,
            logits,
            rtgs,
            rets,
            advs,
            timesteps,
            pre_actions,
            next_states,
            next_obs,
            next_rtgs,
            dones,
            task_ids,
        )


class ReplayBuffer:
    def __init__(self, block_size, global_obs_dim, local_obs_dim, action_dim):
        self.block_size = block_size
        self.buffer_size = 5000
        self.global_obs_dim = global_obs_dim
        self.local_obs_dim = local_obs_dim
        self.action_dim = action_dim
        self.data = []
        self.episodes = []
        self.episode_dones = []
        self.gamma = 0.99
        self.gae_lambda = 0.95

    @property
    def size(self):
        return len(self.data)

    def insert(
        self, global_obs, local_obs, action, reward, done, available_actions, q_value_1, q_value_2, logits, task_ids
    ):
        n_threads, n_agents = np.shape(reward)[0], np.shape(reward)[1]
        for n in range(n_threads):
            if len(self.episodes) < n + 1:
                self.episodes.append([])
                self.episode_dones.append(False)
            if not self.episode_dones[n]:
                for i in range(n_agents):
                    if len(self.episodes[n]) < i + 1:
                        self.episodes[n].append([])
                    step = [
                        global_obs[n][i].tolist(),
                        local_obs[n][i].tolist(),
                        action[n][i].tolist(),
                        reward[n][i].tolist(),
                        done[n][i],
                        available_actions[n][i].tolist(),
                        task_ids[n][i].tolist(),
                        q_value_1[n][i].tolist(),
                        q_value_2[n][i].tolist(),
                        logits[n][i].tolist(),
                    ]
                    self.episodes[n][i].append(step)
                if np.all(done[n]):
                    self.episode_dones[n] = True
                    if self.size > self.buffer_size:
                        raise NotImplementedError
                    if self.size == self.buffer_size:
                        del self.data[0]
                    self.data.append(copy.deepcopy(self.episodes[n]))
        if np.all(self.episode_dones):
            self.episodes = []
            self.episode_dones = []

    def reset(self, num_keep=0, buffer_size=5000):
        self.buffer_size = buffer_size
        if num_keep == 0:
            self.data = []
        elif self.size >= num_keep:
            keep_idx = np.random.randint(0, self.size, num_keep)
            self.data = [self.data[idx] for idx in keep_idx]

    # offline data size could be large than buffer size
    def load_offline_data(self, data_dir, offline_episode_num, max_epi_length=400):
        for j in range(len(data_dir)):
            path_files = glob.glob(pathname=data_dir[j] + "*")
            # for file in sorted(path_files):
            for i in range(offline_episode_num[j]):
                episode = torch.load(path_files[i])

                # if len(episode[0]) > max_epi_length:
                #     print("file: ", path_files[i])
                #     print("invalid episode length: ", len(episode[0]))
                #     continue

                # padding obs
                for agent_trajectory in episode:
                    for step in agent_trajectory:
                        step[0] = padding_obs(step[0], self.global_obs_dim)
                        step[1] = padding_obs(step[1], self.local_obs_dim)
                        step[5] = padding_ava(step[5], self.action_dim)
                        # append task id for the multi-task setting
                        step.append(j)

                self.data.append(episode)

    def sample(self):
        # adding elements with list will be faster
        global_states = []
        local_obss = []
        actions = []
        rewards = []
        avas = []
        q_values_1 = []
        q_values_2 = []
        logits = []
        rtgs = []
        rets = []
        done_idxs = []
        time_steps = []
        advs = []
        task_ids = []

        for episode_idx in range(self.size):
            episode = self.get_episode(episode_idx)
            # episode = self.get_episode(episode_idx, min_return)
            if episode is None:
                continue
            for agent_trajectory in episode:
                time_step = 0
                for step in agent_trajectory:
                    # add task_id in the datasets
                    g, o, a, r, d, ava, task_id, q1, q2, pi, rtg, ret, adv = step
                    global_states.append(g)
                    local_obss.append(o)
                    actions.append(a)
                    rewards.append(r)
                    avas.append(ava)
                    q_values_1.append(q1)
                    q_values_2.append(q2)
                    logits.append(pi)
                    rtgs.append(rtg)
                    rets.append(ret)
                    advs.append(adv)
                    time_steps.append([time_step])
                    task_ids.append(task_id)
                    time_step += 1
                # done_idx - 1 equals the last step's position
                done_idxs.append(len(global_states))

        # or we can separate it as well
        # states = np.concatenate((global_states, local_obss), axis=1)
        dataset = StateActionReturnDataset(
            global_states,
            local_obss,
            self.block_size,
            actions,
            done_idxs,
            rewards,
            avas,
            q_values_1,
            q_values_2,
            logits,
            rtgs,
            rets,
            advs,
            time_steps,
            task_ids,
        )
        return dataset

    # from [g, o, a, r, d, ava]/[g, o, a, r, d, ava, v] to [g, o, a, r, d, ava, v, rtg, ret, adv]


    ### g, o, a, r, d, ava, task_id, q1, q2, rtg, ret, adv = step  8+3
    def get_episode(self, index):
        episode = copy.deepcopy(self.data[index])

        # cal rtg and ret
        for agent_trajectory in episode:
            rtg = 0.0
            ret = 0.0
            adv = 0.0
            for i in reversed(range(len(agent_trajectory))):
                if (
                    len(agent_trajectory[i]) == 7
                ):  # offline, give a fake v_value, unused
                    agent_trajectory[i].append([0.0])
                elif len(agent_trajectory[i]) == 10:
                    pass  # online nothing to do
                else:
                    raise NotImplementedError

                reward = agent_trajectory[i][3][0]
                rtg += reward
                agent_trajectory[i].append([rtg])

                # todo: check ret and adv calculation
                if i == len(agent_trajectory) - 1:
                    next_v = 0.0
                else:
                    next_v = agent_trajectory[i + 1][7][0]
                v = agent_trajectory[i][7][0]
                # adv with gae
                delta = reward + self.gamma * next_v - v
                adv = delta + self.gamma * self.gae_lambda * adv

                # adv without gae
                # adv = reward + self.gamma * next_v - v

                # ret = adv + v
                ret = reward + self.gamma * ret
                # ret = reward + self.gamma * next_v
                # print("reward: %s, v: %s, next_v: %s, adv: %s, ret: %s " % (reward, v, next_v, adv, ret))

                agent_trajectory[i].append([ret])
                agent_trajectory[i].append([adv])

        # prune dead steps
        for i in range(len(episode)):
            end_idx = 0
            for step in episode[i]:
                if step[4]:
                    break
                else:
                    end_idx += 1
            episode[i] = episode[i][0 : end_idx + 1]
        return episode
