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
from torch.utils.data import DataLoader, TensorDataset

from third_party.a2c_ppo_acktr import algo, utils
from third_party.a2c_ppo_acktr.algo import gail
from third_party.a2c_ppo_acktr.arguments import get_args
from third_party.a2c_ppo_acktr.envs import make_vec_envs
from third_party.a2c_ppo_acktr.model import Policy
from third_party.a2c_ppo_acktr.model_split import SplitPolicy
from third_party.a2c_ppo_acktr.storage import RolloutStorage

from my_pybullet_envs import utils as gan_utils

import logging
import sys

sys.path.append("third_party")
np.set_printoptions(precision=2, suppress=None, threshold=sys.maxsize)


def main():
    args, extra_dict = get_args()

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
        if args.use_split_pi:
            actor_critic = SplitPolicy(
                envs.observation_space.shape,
                envs.action_space,
                base_kwargs={'hidden_size': args.hidden_size, 'num_feet': args.num_feet})
        else:
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
    dummy.close()

    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler("{0}/{1}.log".format(save_path, "console_output"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    if args.algo == 'ppo':
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
    else:
        raise ValueError("only support PPO in gail dyn")

    assert len(envs.observation_space.shape) == 1

    expert_sas_w_past = gan_utils.load_sas_wpast_from_pickle(
        args.gail_traj_path,
        downsample_freq=int(args.gail_downsample_frequency),
        load_num_trajs=args.gail_traj_num
    )
    # assume in the order of s_old,..., a_old,..., st+1
    s_dim = expert_sas_w_past[-1].shape[1]
    a_dim = expert_sas_w_past[-2].shape[1]

    # s_idx = np.array([0, 3])
    # a_idx = np.array([0, 3])
    s_idx = np.array([0])
    a_idx = np.array([0])
    # s_idx = np.array([0, 4, 8])
    # a_idx = np.array([0, 4, 8])
    # s_idx = np.array([8])
    # a_idx = np.array([0, 4, 8])

    info_length = len(s_idx) * s_dim + len(a_idx) * a_dim + s_dim       # last term s_t+1
    discr = gail.Discriminator(
        info_length, args.gail_dis_hdim,
        device)
    expert_merged_sas = gan_utils.select_and_merge_sas(expert_sas_w_past, a_idx=a_idx, s_idx=s_idx)
    assert expert_merged_sas.shape[1] == info_length
    expert_dataset = TensorDataset(Tensor(expert_merged_sas))

    gail_tar_length = expert_merged_sas.shape[0] * 1.0 / args.gail_traj_num * args.gail_downsample_frequency
    # print(gail_tar_length)

    drop_last = len(expert_dataset) > args.gail_batch_size
    gail_train_loader = DataLoader(
        expert_dataset,
        batch_size=args.gail_batch_size,
        shuffle=True,
        drop_last=drop_last)

    obs = envs.reset()

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size,
                              info_length)
    # reset does not have info dict, but is okay,
    # and keep rollouts.obs_feat[0] 0, will not be used, insert from 1 (for backward compact)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10000)
    gail_rewards = deque(maxlen=10)  # this is just a moving average filter
    total_num_episodes = 0
    j = 0
    max_num_episodes = args.num_episodes if args.num_episodes else np.infty

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    from third_party.a2c_ppo_acktr.baselines.common.running_mean_std import RunningMeanStd
    ret_rms = RunningMeanStd(shape=())

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
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            sas_feat = np.zeros((args.num_processes, info_length))
            for core_idx, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                # get the past info here and apply filter
                sas_info = info["sas_window"]
                sas_feat[core_idx, :] = gan_utils.select_and_merge_sas(sas_info, s_idx=s_idx, a_idx=a_idx)

            # print(sas_feat)
            # If done then clean the history of observations.
            masks = Tensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = Tensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks, Tensor(sas_feat))

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        gail_loss, gail_loss_e, gail_loss_p = None, None, None
        gail_epoch = args.gail_epoch

        # TODO: odd. turn this off for now since no state normalize
        # if j >= 10:
        #     envs.venv.eval()
        # Warm up
        # if j < 10:
        #     gail_epoch = 100

        # use next obs feat batch during update...
        # if j % 2 == 0:
        for _ in range(gail_epoch):
            gail_loss, gail_loss_e, gail_loss_p = discr.update_gail_dyn(gail_train_loader, rollouts)

        num_of_dones = (1.0 - rollouts.masks).sum().cpu().numpy() \
            + args.num_processes / 2
        # print(num_of_dones)
        num_of_expert_dones = (args.num_steps * args.num_processes) / gail_tar_length
        # print(num_of_expert_dones)

        # d_sa < 0.5 if pi too short (too many pi dones),
        # d_sa > 0.5 if pi too long
        d_sa = 1 - num_of_dones / (num_of_dones + num_of_expert_dones)
        # print(d_sa)
        if args.no_alive_bonus:
            r_sa = 0
        else:
            r_sa = np.log(d_sa) - np.log(1 - d_sa)  # d->1, r->inf

        # use next obs feat to overwrite reward...
        # overwriting rewards by gail
        for step in range(args.num_steps):
            rollouts.rewards[step], returns = \
                discr.predict_reward_combined(
                    rollouts.obs_feat[step + 1], args.gamma,
                    rollouts.masks[step], offset=-r_sa
                )

            # print(rollouts.obs[step, 0])
            # print(rollouts.obs_feat[step+1, 0])

            # redo reward normalize after overwriting
            # print(rollouts.rewards[step], returns)
            ret_rms.update(returns.view(-1).cpu().numpy())
            rews = rollouts.rewards[step].view(-1).cpu().numpy()
            rews = np.clip(rews / np.sqrt(ret_rms.var + 1e-7),
                           -10.0, 10.0)
            # print(ret_rms.var)    # just one number
            rollouts.rewards[step] = Tensor(rews).view(-1, 1)
            # print("after", rollouts.rewards[step], returns)

            # final returns
            # print(returns)
            gail_rewards.append(torch.mean(returns).cpu().data)

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

            if args.gail:
                torch.save(discr, os.path.join(save_path, args.env_name + "_D.pt"))
                torch.save(discr, os.path.join(save_path, args.env_name + "_" + str(j) + "_D.pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            root_logger.info(
                ("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes:" +
                 " mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, " +
                 "dist en {}, l_pi {}, l_vf {}, recent_gail_r {}," +
                 "loss_gail {}, loss_gail_e {}, loss_gail_p {}\n").format(
                    j, total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards), np.mean(episode_rewards),
                    np.median(episode_rewards), np.min(episode_rewards),
                    np.max(episode_rewards), dist_entropy, value_loss,
                    action_loss, np.mean(gail_rewards),
                    gail_loss, gail_loss_e, gail_loss_p
                )
            )
            # actor_critic.dist.logstd._bias,

        total_num_episodes += len(episode_rewards)
        episode_rewards.clear()
        j += 1


if __name__ == "__main__":
    main()
