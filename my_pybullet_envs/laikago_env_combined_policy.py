#  Copyright 2020 Google LLC and Stanford University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from .laikago import LaikagoBullet

from pybullet_utils import bullet_client
import pybullet
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
import torch
from my_pybullet_envs import utils
from collections import deque

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class LaikagoCombinedEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 render=True,
                 init_noise=True,
                 act_noise=False,
                 obs_noise=False,
                 control_skip=10,

                 max_tar_vel=2.5,
                 energy_weight=0.1,
                 jl_weight=0.5,
                 ab=5.0,
                 q_pen_weight=0.4,
                 acc_pen_weight=0.03,
                 vel_r_weight=4.0,

                 train_dyn=True,  # if false, fix dyn and train motor policy
                 pretrain_dyn=False,        # pre-train with deviation to sim
                 behavior_dir="trained_models_laika_bullet_61/ppo",
                 behavior_env_name="LaikagoBulletEnv-v4",
                 behavior_logstd=None,      # if float, reset log std to make behavior pi more diverse
                 behavior_iter=None,
                 dyn_dir="",
                 dyn_env_name="LaikagoCombinedEnv-v1",
                 dyn_iter=None,

                 cuda_env=True,
                 task_y=False       # generalize to tilt y walking
                 ):

        self.render = render
        self.init_noise = init_noise
        self.obs_noise = obs_noise
        self.act_noise = act_noise
        self.control_skip = int(control_skip)
        self._ts = 1. / 500.

        self.max_tar_vel = max_tar_vel
        self.energy_weight = energy_weight
        self.jl_weight = jl_weight
        self.ab = ab
        self.q_pen_weight = q_pen_weight
        self.acc_pen_weight = acc_pen_weight
        self.vel_r_weight = vel_r_weight

        self.train_dyn = train_dyn
        self.pretrain_dyn = pretrain_dyn
        self.cuda_env = cuda_env
        self.task_y = task_y

        self.ratio = None

        if self.render:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self.np_random = None
        self.robot = LaikagoBullet(init_noise=self.init_noise,
                                   time_step=self._ts,
                                   np_random=self.np_random)
        self.seed(0)  # used once temporarily, will be overwritten outside though superclass api
        self.viewer = None
        self.timer = 0

        self.behavior_past_obs_t_idx = [0, 4, 8]

        self.past_obs_array = deque(maxlen=10)
        self.past_bact_array = deque(maxlen=10)     # only need to store past behavior action

        if self.train_dyn:
            if behavior_iter:
                behavior_iter = int(behavior_iter)
            self.dyn_actor_critics = None
            # load fixed behavior policy
            self.go_actor_critic, _, \
                self.recurrent_hidden_states, \
                self.masks = utils.load(
                    behavior_dir, behavior_env_name, self.cuda_env, behavior_iter
                )
        else:
            # if dyn_iter:
            #     dyn_iter = int(dyn_iter)
            # train motor policy
            self.go_actor_critic = None
            # load fixed dynamics model
            # self.dyn_actor_critic, _, \
            #     self.recurrent_hidden_states, \
            #     self.masks = utils.load(
            #         dyn_dir, dyn_env_name, self.cuda_env, dyn_iter
            #     )
            # TODO: arbitrary ensemble
            dyn_actor_critic_1, _, \
                self.recurrent_hidden_states, \
                self.masks = utils.load(
                    dyn_dir, dyn_env_name, self.cuda_env, 950
                )
            dyn_actor_critic_2, _, \
                _, \
                _ = utils.load(
                    dyn_dir, dyn_env_name, self.cuda_env, 850
                )
            dyn_actor_critic_3, _, \
                _, \
                _ = utils.load(
                    dyn_dir, dyn_env_name, self.cuda_env, 750
                )
            dyn_actor_critic_4, _, \
                _, \
                _ = utils.load(
                    dyn_dir, dyn_env_name, self.cuda_env, 650
                )
            dyn_actor_critic_5, _, \
                _, \
                _ = utils.load(
                    dyn_dir, dyn_env_name, self.cuda_env, 550
                )
            self.dyn_actor_critics = [dyn_actor_critic_1, dyn_actor_critic_2,
                                      dyn_actor_critic_3, dyn_actor_critic_4,
                                      dyn_actor_critic_5]

            #
            # self.discri = utils.load_gail_discriminator(dyn_dir,
            #                                             dyn_env_name,
            #                                             self.cuda_env,
            #                                             dyn_iter)
            #
            # self.feat_select_func = self.robot.feature_selection_all_laika

        self.reset_const = 100
        self.reset_counter = self.reset_const  # do a hard reset first

        # self.action_dim = 12

        self.init_state = None
        obs = self.reset()

        if self.train_dyn:
            self.action_dim = 16 + 12   # 16+12D action scales, see beginning of step() for comment
        else:
            self.action_dim = len(self.robot.ctrl_dofs)

        self.act = [0.0] * self.action_dim
        self.action_space = gym.spaces.Box(low=np.array([-1.] * self.action_dim),
                                           high=np.array([+1.] * self.action_dim))
        obs_dummy = np.array([1.12234567] * len(obs))
        self.observation_space = gym.spaces.Box(low=-np.inf * obs_dummy, high=np.inf * obs_dummy)

        if self.train_dyn and behavior_logstd:
            device = "cuda" if self.cuda_env else "cpu"
            act_space_b = gym.spaces.Box(low=np.array([-1.] * len(self.robot.ctrl_dofs)),
                                         high=np.array([+1.] * len(self.robot.ctrl_dofs)))
            self.go_actor_critic.reset_variance(act_space_b, behavior_logstd)
            self.go_actor_critic.to(device)

    def reset(self):

        if self.reset_counter < self.reset_const:
            self.reset_counter += 1

            self._p.restoreState(self.init_state)
            self.robot.soft_reset(self._p)
        else:
            self.reset_counter = 0

            self._p.resetSimulation()
            self._p.setTimeStep(self._ts)
            self._p.setGravity(0, 0, -10)
            self._p.setPhysicsEngineParameter(numSolverIterations=100)
            # self._p.setPhysicsEngineParameter(restitutionVelocityThreshold=0.000001)

            self.robot.reset(self._p)

            self.floor_id = self._p.loadURDF(
                os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0.0], useFixedBase=1
            )

            self.init_state = self._p.saveState()

        # reset feet
        for ind in self.robot.feet:
            self._p.changeDynamics(self.robot.go_id, ind,
                                   contactDamping=1000, contactStiffness=1.0,
                                   lateralFriction=1.0, spinningFriction=0.1, restitution=0.0)
        self._p.changeDynamics(self.floor_id, -1, lateralFriction=0.5, spinningFriction=0.0, restitution=1.0,
                               contactDamping=0, contactStiffness=1.0)

        self._p.stepSimulation()

        self.timer = 0
        self.past_obs_array.clear()
        self.past_bact_array.clear()
        # self.d_scores = []
        obs = self.get_extended_observation()

        # self.ratios = np.array([[]]).reshape(0, self.action_dim)

        return obs

    def step(self, a):
        # TODO: currently for laika, env_action is 16+12D, 4 feet *
        # 4D contact coeffs + 4* 3D leg battery level

        if self.train_dyn:
            assert len(a) == 16 + 12
            env_action = a
            robo_action = self.past_bact_array[0]       # after tanh
        else:
            robo_action = a
            robo_action = np.tanh(robo_action)
            # update past_bact after tanh
            utils.push_recent_value(self.past_bact_array, robo_action)

            # env_pi_obs = utils.select_and_merge_from_s_a(
            #                 s_mt=list(self.past_obs_array),
            #                 a_mt=list(self.past_bact_array),
            #                 s_idx=self.generator_past_obs_t_idx,
            #                 a_idx=self.generator_past_act_t_idx
            #             )

            # # calculate 4x split obs here
            # env_pi_obs = self.get_all_feet_local_obs(robo_action)

            # old q, dq, a obs
            obs_w_dq = self.robot.get_robot_observation(with_vel=True)
            env_pi_obs = np.concatenate((obs_w_dq, robo_action))

            env_pi_obs_nn = utils.wrap(env_pi_obs, is_cuda=self.cuda_env)

            ind = self.np_random.choice(len(self.dyn_actor_critics))
            with torch.no_grad():
                _, env_action_nn, _, self.recurrent_hidden_states = self.dyn_actor_critics[ind].act(
                    env_pi_obs_nn, self.recurrent_hidden_states, self.masks, deterministic=False
                )
            env_action = utils.unwrap(env_action_nn, is_cuda=self.cuda_env)

            assert len(env_action) == 16 + 12

        root_pos, _ = self.robot.get_link_com_xyz_orn(-1)
        x_0 = root_pos[0]
        y_0 = root_pos[1]

        # this is post noise (unseen), different from seen diversify of logstd
        if self.act_noise:
            robo_action = utils.perturb(robo_action, 0.05, self.np_random)

        # when call info, should call before sim_step() as in v4 (append s_t+1 later)
        # this info will be used to construct D input outside.
        past_info = self.construct_past_traj_window()

        _, dq_old = self.robot.get_q_dq(self.robot.ctrl_dofs)

        for _ in range(self.control_skip):
            battery_levels = self.set_con_coeff_and_return_battery_level(env_action)
            self.robot.apply_action(robo_action * battery_levels)
            self._p.stepSimulation()
            if self.render:
                time.sleep(self._ts * 1.5)
            self.timer += 1

        obs_new = self.get_extended_observation()       # and update past_obs_array
        past_info += [self.past_obs_array[0]]       # s_t+1

        root_pos, _ = self.robot.get_link_com_xyz_orn(-1)
        x_1 = root_pos[0]
        y_1 = root_pos[1]
        self.velx = (x_1 - x_0) / (self.control_skip * self._ts)
        vely = (y_1 - y_0) / (self.control_skip * self._ts)

        height = root_pos[2]
        q, dq = self.robot.get_q_dq(self.robot.ctrl_dofs)
        # print(np.max(np.abs(dq)))
        # in_support = self.robot.is_root_com_in_support()

        if not self.pretrain_dyn:
            reward = self.ab  # alive bonus
            tar = np.minimum(self.timer / 500, self.max_tar_vel)

            if self.task_y:
                # reward += np.minimum(self.velx / np.sqrt(5) * 2 + vely / np.sqrt(5), tar) * self.vel_r_weight
                reward += np.minimum(vely, tar) * self.vel_r_weight * 1.5
            else:
                reward += np.minimum(self.velx, tar) * self.vel_r_weight

            # print("v", self.velx, "tar", tar)
            reward += -self.energy_weight * np.square(robo_action).sum()
            # print("act norm", -self.energy_weight * np.square(robo_action).sum())

            pos_mid = 0.5 * (self.robot.ll + self.robot.ul)
            q_scaled = 2 * (q - pos_mid) / (self.robot.ul - self.robot.ll)
            joints_at_limit = np.count_nonzero(np.abs(q_scaled) > 0.97)
            reward += -self.jl_weight * joints_at_limit
            # print("jl", -self.jl_weight * joints_at_limit)

            reward += -np.minimum(np.sum(np.abs(dq - dq_old)) * self.acc_pen_weight, 5.0)
            weight = np.array([2.0, 1.0, 1.0] * 4)
            reward += -np.minimum(np.sum(np.square(q - self.robot.init_q) * weight) * self.q_pen_weight, 5.0)
            # print("vel pen", -np.minimum(np.sum(np.abs(dq)) * self.dq_pen_weight, 5.0))
            # print("pos pen", -np.minimum(np.sum(np.square(q - self.robot.init_q)) * self.q_pen_weight, 5.0))

            if self.task_y:
                # reward += -np.abs(x_1 - 2 * y_1) * 1.0
                reward += -np.abs(x_1) * 1.0
            else:
                reward += -y_1 * 0.5        # a minor bug, should be abs(y_1)

            # print("dev pen", -y_1*0.5)
        else:
            # reward = self.calc_obs_dist_pretrain(self.img_obs[:-4], self.obs[:len(self.img_obs[:-4])])
            reward = 0  # TODO

        # print("______")
        # print(in_support)

        # print("h", height)
        # print("dq.", np.abs(dq))
        # print((np.abs(dq) < 50).all())

        # print("------")
        _, root_orn = self.robot.get_link_com_xyz_orn(-1)
        rpy = self._p.getEulerFromQuaternion(root_orn)
        diff = (np.array(rpy) - [1.5708, 0.0, 1.5708])

        if self.task_y:
            # do not terminate as in multi-task it may be moving toward y axis
            diff[0] = 0.0

        not_done = (np.abs(dq) < 90).all() and (height > 0.3) and (np.abs(diff) < 1.2).all()

        return obs_new, reward, not not_done, {"sas_window": past_info}

    def set_con_coeff_and_return_battery_level(self, con_f):

        len_foot = 4

        for foot_ind, link in enumerate(self.robot.feet):

            this_fric = np.tanh(con_f[foot_ind * len_foot: (foot_ind + 1) * len_foot])   # [-1 ,1]

            this_fric[0:2] = (this_fric[0:2] + 1) / 2.0 * 5.0     # 0 ~ 5, lateral & spin Fric
            this_fric[2] = (this_fric[2] + 1) / 2.0 * 15.0       # 0 ~ 15, restitution
            this_fric[3] = (this_fric[3] + 1) / 2.0 * 2.0 + 1.0        # 1 ~ 3, log (damping/2)
            this_fric[3] = np.exp(this_fric[3]) * 2                     # 20 ~ 2000, damping

            # print(this_fric)

            self._p.changeDynamics(self.robot.go_id, link, lateralFriction=this_fric[0], spinningFriction=this_fric[1],
                                   restitution=this_fric[2], contactDamping=this_fric[3], contactStiffness=1.0)

        # battery level from -0.5 to 1.5
        all_battery_levels = np.tanh(con_f[4 * len_foot:]) + 0.5

        return np.array(all_battery_levels)

    def construct_past_traj_window(self):
        # st, ... st-9, at, ..., at-9
        # call this before s_t+1 enters deque
        # order does not matter as long as it is the same in policy & expert batch
        # print(list(self.past_obs_array) + list(self.past_act_array))
        return list(self.past_obs_array) + list(self.past_bact_array)

    def get_ave_dx(self):
        return self.velx

    def get_dist(self):
        return self.robot.get_link_com_xyz_orn(-1)[0][0]

    def get_extended_observation(self):

        # with vel false
        cur_state = self.robot.get_robot_observation(with_vel=False)

        if self.obs_noise:
            cur_state = utils.perturb(cur_state, 0.1, self.np_random)

        # then update past obs
        utils.push_recent_value(self.past_obs_array, cur_state)

        # then construct behavior obs
        b_obs_all = utils.select_and_merge_from_s_a(
            s_mt=list(self.past_obs_array),
            a_mt=list(self.past_bact_array),
            s_idx=self.behavior_past_obs_t_idx,
            a_idx=np.array([])
        )
        # if train motor, return behavior obs and we are done
        if not self.train_dyn:
            return b_obs_all

        # else, train dyn
        # rollout b_pi
        obs_nn = utils.wrap(b_obs_all, is_cuda=self.cuda_env)
        with torch.no_grad():
            _, action_nn, _, self.recurrent_hidden_states = self.go_actor_critic.act(
                obs_nn, self.recurrent_hidden_states, self.masks, deterministic=False
            )
        b_cur_act = list(utils.unwrap(action_nn, is_cuda=self.cuda_env))
        b_cur_act = np.tanh(b_cur_act)

        # Store action after tanh (-1,1)
        utils.push_recent_value(self.past_bact_array, b_cur_act)

        # old q, dq, a obs
        obs_w_dq = self.robot.get_robot_observation(with_vel=True)
        g_obs_all = np.concatenate((obs_w_dq, b_cur_act))

        return g_obs_all

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s

    def cam_track_torso_link(self):
        distance = 1.5
        yaw = 0
        root_pos, _ = self.robot.get_link_com_xyz_orn(-1)
        distance -= root_pos[1]
        self._p.resetDebugVisualizerCamera(distance, yaw, -20, [root_pos[0], 0.0, 0.1])
