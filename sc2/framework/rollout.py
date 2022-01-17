import copy

import torch
import numpy as np
from .utils import sample, padding_obs, padding_ava
from .feature_translation import gen_task_embedding


class RolloutWorker:
    def __init__(
        self,
        m3,
        critic_model,
        critic_model_2,
        buffer,
        global_obs_dim,
        local_obs_dim,
        action_dim,
        args,
    ):
        self.args = args
        self.buffer = buffer
        self.critic_model_2 = critic_model_2
        self.m3 = m3
        self.critic_model = critic_model
        self.global_obs_dim = global_obs_dim
        self.local_obs_dim = local_obs_dim
        self.action_dim = action_dim
        self.device = "cpu"
        if torch.cuda.is_available():
            # if torch.cuda.is_available() and not isinstance(self.model, torch.nn.DataParallel):
            self.device = torch.cuda.current_device()
            self.critic_model_2 = torch.nn.DataParallel(critic_model_2).to(self.device)
            self.m3 = torch.nn.DataParallel(m3).to(self.device)
            self.critic_model = torch.nn.DataParallel(critic_model).to(self.device)

    def rollout(self, env, ret, train=True, random_rate=0.0, task_id=0):
        self.m3.train(False)
        self.critic_model_2.train(False)
        self.critic_model.train(False)

        T_rewards, T_wins, steps, episode_dones = 0.0, 0.0, 0, np.zeros(env.n_threads)

        obs, share_obs, available_actions = env.real_env.reset()  # (16, 3, 79)  (16, 3, 99)  (16, 3, 10)
        obs = padding_obs(obs, self.local_obs_dim)                # # (16, 3, 79)
        share_obs = padding_obs(share_obs, self.global_obs_dim)   # (16, 3, 99)
        available_actions = padding_ava(available_actions, self.action_dim)  # (16, 3, 10)

        # x: (n_threads, n_agent, context_lengrh, dim)
        global_states = torch.from_numpy(share_obs).to(self.device).unsqueeze(2)  # torch.Size([16, 3, 1, 99])
        local_obss = torch.from_numpy(obs).to(self.device).unsqueeze(2)
        rtgs = np.ones((env.n_threads, env.num_agents, 1, 1)) * ret  #   (16, 3, 1, 1)
        actions = np.zeros((env.n_threads, env.num_agents, 1, 1))
        timesteps = torch.zeros(
            (env.n_threads * env.num_agents, 1, 1), dtype=torch.int64
        )
        if self.args.use_task_embedding:
            if self.args.use_one_hot_task_embedding:
                task_ids = task_id * torch.ones(
                    (env.n_threads * env.num_agents, 1, 1), dtype=torch.int64
                )
            else:
                task_ids = torch.tensor(
                    np.repeat(
                        np.expand_dims(task_id, axis=0),
                        env.n_threads * env.num_agents,
                        axis=0,
                    ),
                    dtype=torch.int64,
                ).unsqueeze(1)  # torch.Size([48, 1, 28])
        t = 0

        while True:
            if self.args.use_task_embedding:
                sampled_action, v_value = sample(
                    self.m3,
                    self.critic_model,
                    self.critic_model_2,
                    state=global_states.view(-1, np.shape(global_states)[2], np.shape(global_states)[3]).to(self.device),  # torch.Size([48, 1, 99])
                    obs=local_obss.view(-1, np.shape(local_obss)[2], np.shape(local_obss)[3]).to(self.device),
                    sample=train,
                    actions=torch.tensor(actions, dtype=torch.int64).to(self.device).view(-1, np.shape(actions)[2], np.shape(actions)[3]).to(self.device),
                    rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).view(-1, np.shape(rtgs)[2], np.shape(rtgs)[3]),
                    timesteps=timesteps.to(self.device),
                    available_actions=torch.from_numpy(available_actions).view(-1, np.shape(available_actions)[-1]).to(self.device),
                    task_ids=task_ids.to(self.device),
                    use_task_embedding=self.args.use_task_embedding,
                )
            # else:
            #     sampled_action, v_value = sample(
            #         self.m3,
            #         self.model,
            #         self.critic_model,
            #         state=global_states.view(
            #             -1, np.shape(global_states)[2], np.shape(global_states)[3]
            #         ).to(self.device),
            #         obs=local_obss.view(
            #             -1, np.shape(local_obss)[2], np.shape(local_obss)[3]
            #         ).to(self.device),
            #         sample=train,
            #         actions=torch.tensor(actions, dtype=torch.int64)
            #         .to(self.device)
            #         .view(-1, np.shape(actions)[2], np.shape(actions)[3])
            #         .to(self.device),
            #         rtgs=torch.tensor(rtgs, dtype=torch.float32)
            #         .to(self.device)
            #         .view(-1, np.shape(rtgs)[2], np.shape(rtgs)[3]),
            #         timesteps=timesteps.to(self.device),
            #         available_actions=torch.from_numpy(available_actions)
            #         .view(-1, np.shape(available_actions)[-1])
            #         .to(self.device),
            #         task_ids=None,
            #         use_task_embedding=False,
            #     )

            action = (
                sampled_action.view((env.n_threads, env.num_agents, -1)).cpu().numpy()
            )

            cur_global_obs = share_obs
            cur_local_obs = obs
            cur_ava = available_actions

            (
                obs,
                share_obs,
                rewards,
                dones,
                infos,
                available_actions,
            ) = env.real_env.step(action)
            obs = padding_obs(obs, self.local_obs_dim)
            share_obs = padding_obs(share_obs, self.global_obs_dim)
            available_actions = padding_ava(available_actions, self.action_dim)
            t += 1

            if train:
                q_value_1 = (
                    v_value[0].view((env.n_threads, env.num_agents, -1)).cpu().numpy()
                )
                q_value_2 = (
                    v_value[1].view((env.n_threads, env.num_agents, -1)).cpu().numpy()
                )
                logits = (
                    v_value[2].view((env.n_threads, env.num_agents, -1)).cpu().detach().numpy()
                )
                # q_values = (q_value_1, q_value_2)
                # Todo: Check the task ids shape!
                insert_task_ids = torch.tensor(
                    np.repeat(
                        np.expand_dims(task_id, axis=0),
                        env.n_threads * env.num_agents,
                        axis=0,
                    ),
                    dtype=torch.int64,
                ).unsqueeze(1).view((env.n_threads, env.num_agents, -1)).cpu().numpy()
                self.buffer.insert(
                    cur_global_obs,
                    cur_local_obs,
                    action,
                    rewards,
                    dones,
                    cur_ava,
                    q_value_1,
                    q_value_2,
                    logits,
                    insert_task_ids,
                )

            for n in range(env.n_threads):
                if not episode_dones[n]:
                    steps += 1
                    T_rewards += np.mean(rewards[n])
                    if np.all(dones[n]):
                        episode_dones[n] = 1
                        if infos[n][0]["won"]:
                            T_wins += 1.0
            if np.all(episode_dones):
                break

            rtgs = np.concatenate(
                [rtgs, np.expand_dims(rtgs[:, :, -1, :] - rewards, axis=2)], axis=2
            )
            global_state = torch.from_numpy(share_obs).to(self.device).unsqueeze(2)
            global_states = torch.cat([global_states, global_state], dim=2)
            local_obs = torch.from_numpy(obs).to(self.device).unsqueeze(2)
            local_obss = torch.cat([local_obss, local_obs], dim=2)
            actions = np.concatenate([actions, np.expand_dims(action, axis=2)], axis=2)
            timestep = t * torch.ones(
                (env.n_threads * env.num_agents, 1, 1), dtype=torch.int64
            )
            timesteps = torch.cat([timesteps, timestep], dim=1)
            if self.args.use_task_embedding:
                if self.args.use_one_hot_task_embedding:
                    task_ids = torch.cat(
                        [
                            task_ids,
                            task_id
                            * torch.ones(
                                (env.n_threads * env.num_agents, 1, 1),
                                dtype=torch.int64,
                            ),
                        ],
                        dim=1,
                    )
                else:
                    task_ids = torch.cat(
                        [
                            task_ids,
                            torch.tensor(
                                np.repeat(
                                    np.expand_dims(task_id, axis=0),
                                    env.n_threads * env.num_agents,
                                    axis=0,
                                ),
                                dtype=torch.int64,
                            ).unsqueeze(1),
                        ],
                        dim=1,
                    )

        aver_return = T_rewards / env.n_threads
        aver_win_rate = T_wins / env.n_threads
        self.critic_model_2.train(True)
        self.m3.train(True)
        self.critic_model.train(True)
        return aver_return, aver_win_rate, steps
