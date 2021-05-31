#  MIT License
#
#  Copyright (c) 2017 Ilya Kostrikov and (c) 2020 Google LLC
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

# import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd

from third_party.a2c_ppo_acktr.baselines.common.running_mean_std import RunningMeanStd


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(Discriminator, self).__init__()

        self.device = device

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()

        # self.optimizer = torch.optim.RMSprop(self.trunk.parameters(), lr=5e-4)  # TODO
        self.optimizer = torch.optim.Adam(self.trunk.parameters())

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10.):

        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        grad_pen = self.compute_grad_pen_combined(expert_data, policy_data, lambda_)

        return grad_pen

    def compute_grad_pen_combined(self,
                                  expert_combined,
                                  policy_combined,
                                  lambda_=10.
                                  ):
        alpha = torch.rand(expert_combined.size(0), 1)
        alpha = alpha.expand_as(expert_combined).to(expert_combined.device)

        mixup_data = alpha * expert_combined + (1 - alpha) * policy_combined
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts, obsfilt=None, is_gail_dyn=False, a_dim=None):
        # if is_gail_dyn, the policy is a dynamics model to be learned
        # if is_gail_dyn, s_dim should be an int storing original state dimension (wo act)
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = expert_loss_t = policy_loss_t = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            if not is_gail_dyn:
                policy_state, policy_action = policy_batch[0], policy_batch[2]
            else:
                cur_obs_feat_batch = policy_batch[-2]
                cur_obs_a_batch = policy_batch[0][:, -a_dim:]
                cur_obs_feat_a = torch.cat((cur_obs_feat_batch, cur_obs_a_batch), 1)
                policy_state, policy_action = cur_obs_feat_a, policy_batch[-1]

            # policy_state, policy_action = policy_state[:2], policy_action[:2]

            policy_d = self.trunk(
                torch.cat([policy_state, policy_action], dim=1))

            expert_state, expert_action = expert_batch
            # expert_state, expert_action = expert_state[:2], expert_action[:2]
            if obsfilt is not None:
                expert_state = obsfilt(expert_state.cpu().numpy(), update=False)
                expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(
                torch.cat([expert_state, expert_action], dim=1))

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()
            expert_loss_t += expert_loss.item()
            policy_loss_t += policy_loss.item()
            n += 1

            # if n > 200:
            #     break

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
            #
            # # TODO: Clip weights of discriminator
            # for p in self.trunk.parameters():
            #     p.data.clamp_(-0.1, 0.1)

        return loss / n, expert_loss_t / n, policy_loss_t / n

    def update_gail_dyn(self, expert_loader, rollouts):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = expert_loss_t = policy_loss_t = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):

            expert_data = expert_batch[0]
            policy_data = policy_batch[-1]      # see feed_forward_generator yield

            policy_d = self.trunk(policy_data)
            expert_d = self.trunk(expert_data)

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen_combined(expert_data, policy_data)

            loss += (gail_loss + grad_pen).item()
            expert_loss_t += expert_loss.item()
            policy_loss_t += policy_loss.item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
            #
            # # TODO: Clip weights of discriminator
            # for p in self.trunk.parameters():
            #     p.data.clamp_(-0.1, 0.1)
        return loss / n, expert_loss_t / n, policy_loss_t / n

    def predict_reward(self, state, action, gamma, masks, offset=0.0):
        with torch.no_grad():
            self.eval()
            d_in = torch.cat([state, action], dim=1)
            return self.predict_reward_combined(d_in, gamma, masks, offset)

    def predict_reward_combined(self, d_in, gamma, masks, offset=0.0):
        with torch.no_grad():
            d = self.trunk(d_in)
            s = torch.sigmoid(d)
            reward = (s + 1e-7).log() - (1 - s + 1e-7).log() + offset
            if self.returns is None:
                self.returns = reward.clone()
            else:
                self.returns = self.returns * gamma * masks + reward
            return reward, self.returns

    def predict_prob_single_step(self, state, action):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)
            return s  # s=1, think real, 0 think fake

# class ExpertDataset(torch.utils.data.Dataset):
#     def __init__(self, file_name, num_trajectories=4, subsample_frequency=20):
#         all_trajectories = torch.load(file_name)
#
#         perm = torch.randperm(all_trajectories['states'].size(0))
#         idx = perm[:num_trajectories]
#
#         self.trajectories = {}
#
#         # See https://github.com/pytorch/pytorch/issues/14886
#         # .long() for fixing bug in torch v0.4.1
#         start_idx = torch.randint(
#             0, subsample_frequency, size=(num_trajectories, )).long()
#
#         for k, v in all_trajectories.items():
#             data = v[idx]
#
#             if k != 'lengths':
#                 samples = []
#                 for i in range(num_trajectories):
#                     samples.append(data[i, start_idx[i]::subsample_frequency])
#                 self.trajectories[k] = torch.stack(samples)
#             else:
#                 self.trajectories[k] = data // subsample_frequency
#
#         self.i2traj_idx = {}
#         self.i2i = {}
#
#         self.length = self.trajectories['lengths'].sum().item()
#
#         traj_idx = 0
#         i = 0
#
#         self.get_idx = []
#
#         for j in range(self.length):
#
#             while self.trajectories['lengths'][traj_idx].item() <= i:
#                 i -= self.trajectories['lengths'][traj_idx].item()
#                 traj_idx += 1
#
#             self.get_idx.append((traj_idx, i))
#
#             i += 1
#
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, i):
#         traj_idx, i = self.get_idx[i]
#
#         return self.trajectories['states'][traj_idx][i], self.trajectories[
#             'actions'][traj_idx][i]
