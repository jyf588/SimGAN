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

import os
import time
from collections import deque

import gym
import numpy as np
import torch

from third_party.a2c_ppo_acktr import algo, utils
from third_party.a2c_ppo_acktr.arguments import get_args
from third_party.a2c_ppo_acktr.envs import make_vec_envs
from third_party.a2c_ppo_acktr.model import Policy
from third_party.a2c_ppo_acktr.storage import RolloutStorage

from my_pybullet_envs import utils as gan_utils

import logging
import sys

from my_pybullet_envs.laikago import mirror_obs, mirror_action

sys.path.append("third_party")


def main():
    args, extra_dict = get_args()

    # this file for normal ppo training, sim-gan(gail-dyn) training in main_gail_dyn_ppo.py
    assert not args.gail

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, render=False, **extra_dict)

    if args.warm_start == '':
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy, 'hidden_size': args.hidden_size})
        actor_critic.to(device)
    else:
        # TODO: assume no state normalize ob_rms
        if args.cuda:
            actor_critic, _ = torch.load(args.warm_start)
        else:
            actor_critic, _ = torch.load(args.warm_start, map_location='cpu')

        actor_critic.reset_critic(envs.observation_space.shape)
        if args.warm_start_logstd is not None:
            actor_critic.reset_variance(envs.action_space, args.warm_start_logstd)
        actor_critic.to(device)

    dummy = gym.make(args.env_name, render=False, **extra_dict)
    save_path = os.path.join(args.save_dir, args.algo)
    print("SAVE PATH:")
    print(save_path)
    try:
        os.makedirs(save_path)
    except FileExistsError:
        print("warning: path existed")
        # input("warning: path existed")
    except OSError:
        exit()
    pathname = os.path.join(save_path, "source_test.py")
    text_file = open(pathname, "w+")
    text_file.write(dummy.getSourceCode())
    text_file.close()
    print("source file stored")
    # input("source file stored press enter")

    dummy.reset()
    # dummy.close()

    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler("{0}/{1}.log".format(save_path, "console_output"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        if args.loss_sym > 0.0:
            agent = algo.PPO(
                actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                symmetry_coef=args.loss_sym,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm,
                mirror_act=mirror_action,
                mirror_obs=mirror_obs
            )
        else:
            agent = algo.PPO(
                actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)
    else:
        agent = None

    feat_select_func = None

    obs = envs.reset()
    obs_feat = gan_utils.replace_obs_with_feat(obs, args.cuda, feat_select_func, return_tensor=True)
    feat_len = obs_feat.size(1)  # TODO: multi-dim obs broken

    if args.dup_sym:
        buffer_np = args.num_processes * 2
    else:
        buffer_np = args.num_processes
    rollouts = RolloutStorage(args.num_steps, buffer_np,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size,
                              feat_len)
    rollouts.to(device)

    if args.dup_sym:
        obs_s = gan_utils.mirror_obsact_batch(obs, args.cuda, mirror_obs, augment=True)
        obs_feat_s = obs_feat.repeat(2, 1)
        rollouts.obs[0].copy_(obs_s)
        rollouts.obs_feat[0].copy_(obs_feat_s)
    else:
        rollouts.obs[0].copy_(obs)
        rollouts.obs_feat[0].copy_(obs_feat)

    episode_rewards = deque(maxlen=10000)
    total_num_episodes = 0
    j = 0
    max_num_episodes = args.num_episodes if args.num_episodes else np.infty

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    while j < num_updates and total_num_episodes < max_num_episodes:

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # print(args.num_steps) 300*8
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step, :args.num_processes, :],
                    rollouts.recurrent_hidden_states[step, :args.num_processes, :],
                    rollouts.masks[step, :args.num_processes, :])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            obs_feat = gan_utils.replace_obs_with_feat(obs, args.cuda, feat_select_func, return_tensor=True)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = Tensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = Tensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            if args.dup_sym:
                obs_s = gan_utils.mirror_obsact_batch(obs, args.cuda, mirror_obs, augment=True)
                action_s = gan_utils.mirror_obsact_batch(action, args.cuda, mirror_action, augment=True)
                recurrent_hidden_states_s = recurrent_hidden_states.repeat(2, 1)
                action_log_prob_s = action_log_prob.repeat(2, 1)
                value_s = value.repeat(2, 1)
                reward_s = reward.repeat(2, 1)
                masks_s = masks.repeat(2, 1)
                bad_masks_s = bad_masks.repeat(2, 1)
                obs_feat_s = obs_feat.repeat(2, 1)
                rollouts.insert(obs_s, recurrent_hidden_states_s, action_s,
                                action_log_prob_s, value_s, reward_s, masks_s, bad_masks_s, obs_feat_s)
            else:
                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks, obs_feat)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, not args.no_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + "_" + str(j) + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            root_logger.info(
                ("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes:" +
                 " mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, " +
                 "dist en {}, l_pi {}, l_vf {} \n").format(
                    j, total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards), np.mean(episode_rewards),
                    np.median(episode_rewards), np.min(episode_rewards),
                    np.max(episode_rewards), dist_entropy, value_loss,
                    action_loss
                )
            )
            # actor_critic.dist.logstd._bias,

        total_num_episodes += len(episode_rewards)
        episode_rewards.clear()
        j += 1


if __name__ == "__main__":
    main()
