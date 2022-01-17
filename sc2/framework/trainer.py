"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
from .feature_translation import gen_task_embedding
import copy
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.distributions import Categorical
import sys
from .utils import sample

sys.path.append("../../")


def get_weights_copy(model):
    weights_path = "weights_temp.pt"
    torch.save(model.state_dict(), weights_path)
    return torch.load(weights_path)


class TrainerConfig:
    # optimization parameters
    max_epochs = 1000
    batch_size = 128
    learning_rate = 5e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 0.5
    weight_decay = 0.1  # only applied on matmul weights
    num_workers = 0  # for DataLoader
    lam = 0.3  # for awac term

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, model, critic_models, config, args):
        self.model = model
        self.critic_model = critic_models[0]
        self.critic_model_2 = critic_models[1]

        # self.log_alpha = torch.nn.Parameter(
        #     torch.tensor(
        #         [
        #             np.log(0.01, dtype=np.float32)  ### init_temperature:0.01
        #             for _ in range(1)
        #         ]
        #     )
        # )
        # self.log_alpha_optimizer = torch.optim.Adam(self.log_alpha, lr=1e-6)
        # self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        # # We optimize log(alpha), instead of alpha.
        # self.log_alpha = torch.zeros(
        #     1, requires_grad=True, device=self.device)
        # self.alpha = self.log_alpha.exp()
        # self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=1e-6)
        self.alpha = 0.05

        self.config = config
        self.args = args
        self.warmup_epochs = 0

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        self.raw_model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        self.optimizer = self.raw_model.configure_optimizers(
            config, config.learning_rate
        )

        self.raw_critic_model = (
            self.critic_model.module
            if hasattr(self.critic_model, "module")
            else self.critic_model
        )

        self.raw_critic_model_2 = (
            self.critic_model_2.module
            if hasattr(self.critic_model_2, "module")
            else self.critic_model_2
        )

        self.critic_optimizer = torch.optim.Adam(list(self.raw_critic_model.parameters()) + list(self.raw_critic_model_2.parameters()), lr=config.learning_rate * 10)

        # self.critic_optimizer = self.raw_critic_model.configure_optimizers(
        #     config, config.learning_rate * 10
        # )
        #
        # self.critic_2_optimizer = self.raw_critic_model_2.configure_optimizers(
        #     config, config.learning_rate * 10
        # )
        # self.target_model = None
        # self.weights_path = "weights_temp.pt"

    def train(self, dataset, train_critic=True, use_task_embedding=True):
        model, critic_model, config = self.raw_model, self.raw_critic_model, self.config
        critic_model_2 = self.raw_critic_model_2

        # if self.target_model is None:
        #     target_model = model
        #     self.target_model = target_model
        # else:
        #     target_model = self.target_model
        #     target_model.load_state_dict(torch.load(self.weights_path))
        #     self.target_model = target_model
        target_model = copy.deepcopy(model)
        target_model.train(False)
        target_critic = copy.deepcopy(critic_model)
        target_critic.train(False)
        target_critic_2 = copy.deepcopy(critic_model_2)
        target_critic_2.train(False)

        def run_epoch():
            if self.warmup_epochs < self.args.online_warmup:
                model.train(False)
            else:
                model.train(True)
            critic_model.train(True)
            critic_model_2.train(True)
            if self.config.mode == "offline":
                loader = DataLoader(
                    dataset,
                    shuffle=True,
                    pin_memory=True,
                    drop_last=True,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                )
            elif self.config.mode == "online":
                loader = DataLoader(
                    dataset,
                    shuffle=True,
                    pin_memory=True,
                    drop_last=True,
                    batch_size=dataset.__len__(),
                    num_workers=config.num_workers,
                )
            else:
                raise NotImplementedError

            loss_info = 0
            pbar = tqdm(enumerate(loader), total=len(loader))

            # todo: check these inputs
            for (
                it,
                (
                    s,
                    o,
                    a,
                    r,
                    ava,
                    q1,
                    q2,
                    lgts, #logits
                    rtg,
                    ret,
                    adv,
                    t,
                    pre_a,
                    next_s,
                    next_o,
                    next_rtg,
                    done,
                    task_id,
                ),
            ) in pbar:
                s = s.to(self.device)
                o = o.to(self.device)
                a = a.to(self.device)
                r = r.to(self.device)
                ava = ava.to(self.device)
                q1 = q1.to(self.device)
                q2 = q2.to(self.device)
                lgts = lgts.to(self.device)
                rtg = rtg.to(self.device)
                ret = ret.to(self.device)
                adv = adv.to(self.device)
                t = t.to(self.device)
                pre_a = pre_a.to(self.device)
                next_s = next_s.to(self.device)
                next_o = next_o.to(self.device)
                next_rtg = next_rtg.to(self.device)
                done = done.to(self.device)
                if use_task_embedding:
                    if self.args.use_one_hot_task_embedding:
                        task_id = task_id.to(self.device)
                    else:
                        if self.config.mode == "offline":
                            print("task emneddings ... ")
                            task_embedding = np.stack(
                                [
                                    gen_task_embedding(self.args.offline_map_lists[i])
                                    for i in range(len(self.args.offline_map_lists))
                                ]
                            )
                            task_embedding = np.repeat(
                                np.expand_dims(task_embedding, axis=0),
                                task_id.shape[0],
                                axis=0,
                            )
                            task_id = torch.stack(
                                [
                                    torch.tensor(task_embedding, dtype=torch.float32)[i][
                                        task_id[i]
                                    ]
                                    for i in range(task_id.shape[0])
                                ]
                            ).to(self.device)
                        # task_id = torch.gather(torch.tensor(task_embedding, dtype=torch.float32), dim=1, index=torch.tensor(task_id, dtype=torch.int64).unsqueeze(1))
                        # if self.config.mode == "online":
                        #     task_id = task_id.squeeze(1)
                        else:
                            task_id = task_id.to(self.device)
                else:
                    task_id = None

                # update actor
                with torch.set_grad_enabled(True):
                    # import pdb
                    # pdb.set_trace()
                    print(":task_id:", task_id.shape)
                    logits, logits_tuple = model.encode(o.squeeze(1), pre_a, rtg, task_id.squeeze(1), t)  # logits_tuple = (mu, log_pi, log_std)
                    mu, log_pi, log_std = logits_tuple
                    pi = logits
                    actor_Q1 = critic_model(next_o.squeeze(1), pi.squeeze(1), rtg, t, task_id.squeeze(1)).detach()  # 48.1.1
                    actor_Q2 = critic_model_2(next_o.squeeze(1), pi.squeeze(1), rtg, t, task_id.squeeze(1)).detach()
                    actor_Q = torch.min(actor_Q1, actor_Q2)
                    # actor_loss = self.alpha * log_pi - actor_Q
                    ###  here, we use fixed alpha=0.05
                    act_entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
                    loss = - actor_Q[:,0,0] - self.alpha * act_entropy

                    entropy_info = act_entropy.mean().item()
                    loss = loss.mean()
                    loss_info = loss.item()
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_norm_clip
                    )
                    self.optimizer.step()

                    ###  here, we use fixed alpha=0.05
                    # self.log_alpha_optimizer.zero_grad()
                    # alpha_loss = (self.alpha) * (-log_pi - act_entropy).detach()
                    # alpha_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.log_alpha.parameters(), config.grad_norm_clip)
                    # self.log_alpha_optimizer.step()

                # update critic
                critic_loss_info = 0.0
                if train_critic:
                    with torch.set_grad_enabled(True):

                        pi, logits_tuple = model.encode(next_o.squeeze(1), pre_a, rtg, task_id.squeeze(1),t)  # logits_tuple = (mu, log_pi, log_std)
                        mu, log_pi, log_std = logits_tuple
                        target_Q1 = target_critic(next_o.squeeze(1), pi.squeeze(1), rtg, t,task_id.squeeze(1)) # 48.1.1
                        target_Q2 = target_critic_2(next_o.squeeze(1), pi.squeeze(1), rtg, t, task_id.squeeze(1))
                        target_V = (torch.min(target_Q1, target_Q2) - self.alpha * log_pi.unsqueeze(-1)).detach()
                        target_Q = r + ((1-done) * 0.99 * target_V)
                        current_Q1 = critic_model(o.squeeze(1), lgts.squeeze(1), rtg, t, task_id.squeeze(1)) # 48.1.1
                        current_Q2 = critic_model_2(o.squeeze(1), lgts.squeeze(1), rtg, t, task_id.squeeze(1))
                        # current_Q1 = q1
                        # current_Q2 = q2
                        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
                        critic_loss_1_info = critic_loss.mean().item()
                        critic_loss_2_info = critic_loss.mean().item()

                    critic_model.zero_grad()
                    critic_model_2.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(critic_model.parameters()) + list(critic_model_2.parameters()), config.grad_norm_clip)
                    self.critic_optimizer.step()

                    # critic_model_2.zero_grad()
                    # critic_loss_2.backward()
                    # torch.nn.utils.clip_grad_norm_(critic_model_2.parameters(), config.grad_norm_clip)
                    # self.critic_2_optimizer.step()

                # report progress
                pbar.set_description(
                    f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}."
                )
            self.warmup_epochs += 1
            return (
                loss_info,
                self.alpha,
                entropy_info,
                critic_loss_1_info,
                critic_loss_2_info,
            )

        actor_loss_ret, critic_loss_ret, entropy, ratio, confidence = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        for epoch in range(config.max_epochs):
            actor_loss_ret, critic_loss_ret, entropy, ratio, confidence = run_epoch()
            # torch.save(model.state_dict(), self.weights_path)

        return actor_loss_ret, critic_loss_ret, entropy, ratio, confidence
