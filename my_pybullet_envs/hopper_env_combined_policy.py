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

from .hopper import HopperURDF

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


class HopperCombinedEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 render=True,
                 init_noise=True,
                 act_noise=False,
                 obs_noise=False,
                 control_skip=10,

                 correct_obs_dx=True,  # if need to correct dx obs,

                 train_dyn=True,  # if false, fix dyn and train motor policy
                 behavior_dir="trained_models_hopper_bullet_new0/ppo",
                 behavior_env_name="HopperURDFEnv-v3",
                 behavior_logstd=None,  # if float, reset log std to make behavior pi more diverse
                 behavior_iter=None,
                 dyn_dir="",
                 dyn_env_name="HopperCombinedEnv-v1",   # itself
                 dyn_iter=None,

                 cuda_env=True
                 ):

        self.render = render
        self.init_noise = init_noise
        self.obs_noise = obs_noise
        self.act_noise = act_noise
        self.control_skip = int(control_skip)
        self._ts = 1. / 500.
        self.correct_obs_dx = correct_obs_dx

        self.train_dyn = train_dyn
        self.cuda_env = cuda_env

        if self.render:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self.np_random = None
        self.robot = HopperURDF(init_noise=self.init_noise,
                                time_step=self._ts,
                                np_random=self.np_random)
        self.seed(0)  # used once temporarily, will be overwritten outside though superclass api
        self.viewer = None
        self.timer = 0

        self.behavior_past_obs_t_idx = [0]
        self.past_obs_array = deque(maxlen=10)
        self.past_bact_array = deque(maxlen=10)     # only need to store past behavior action

        if self.train_dyn:
            if behavior_iter:
                behavior_iter = int(behavior_iter)
            self.dyn_actor_critics = None
            # load fixed behavior policy
            self.hopper_actor_critic, _, \
                self.recurrent_hidden_states, \
                self.masks = utils.load(
                    behavior_dir, behavior_env_name, self.cuda_env, behavior_iter
                )
            if behavior_logstd:
                device = "cuda" if self.cuda_env else "cpu"
                act_space_b = gym.spaces.Box(low=np.array([-1.] * 3),
                                             high=np.array([+1.] * 3))
                self.hopper_actor_critic.reset_variance(act_space_b, behavior_logstd)
                self.hopper_actor_critic.to(device)
        else:
            # if dyn_iter:
            #     dyn_iter = int(dyn_iter)
            # train motor policy
            self.hopper_actor_critic = None
            # # load fixed dynamics model
            # self.dyn_actor_critic, _, \
            #     self.recurrent_hidden_states, \
            #     self.masks = utils.load(
            #         dyn_dir, dyn_env_name, self.cuda_env, dyn_iter
            #     )

            # TODO: arbitrary ensemble
            dyn_actor_critic_1, _, \
                self.recurrent_hidden_states, \
                self.masks = utils.load(
                    dyn_dir, dyn_env_name, self.cuda_env, 80
                )
            dyn_actor_critic_2, _, \
                _, \
                _ = utils.load(
                    dyn_dir, dyn_env_name, self.cuda_env, 100
                )
            dyn_actor_critic_3, _, \
                _, \
                _ = utils.load(
                    dyn_dir, dyn_env_name, self.cuda_env, 120
                )
            dyn_actor_critic_4, _, \
                _, \
                _ = utils.load(
                    dyn_dir, dyn_env_name, self.cuda_env, 140
                )
            dyn_actor_critic_5, _, \
                _, \
                _ = utils.load(
                    dyn_dir, dyn_env_name, self.cuda_env, 160
                )
            self.dyn_actor_critics = [dyn_actor_critic_1, dyn_actor_critic_2,
                                      dyn_actor_critic_3, dyn_actor_critic_4,
                                      dyn_actor_critic_5]

        self.obs = []
        self.reset()  # and update init obs

        if self.train_dyn:
            self.action_dim = 7  # see beginning of step() for comment
        else:
            self.action_dim = len(self.robot.ctrl_dofs)

        self.act = [0.0] * self.action_dim
        self.action_space = gym.spaces.Box(low=np.array([-1.] * self.action_dim),
                                           high=np.array([+1.] * self.action_dim))
        obs_dummy = np.array([1.12234567] * len(self.obs))
        self.observation_space = gym.spaces.Box(low=-np.inf * obs_dummy, high=np.inf * obs_dummy)

    def reset(self):
        self._p.resetSimulation()
        self._p.setTimeStep(self._ts)
        self._p.setGravity(0, 0, -10)
        self.timer = 0

        self._p.setPhysicsEngineParameter(numSolverIterations=100)
        # self._p.setPhysicsEngineParameter(restitutionVelocityThreshold=0.000001)

        self.floor_id = self._p.loadURDF(
            os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0.05], useFixedBase=1
        )

        self.robot.reset(self._p)
        self.robot.update_x(reset=True)
        # # should be after reset!

        # reset foot
        self._p.changeDynamics(self.robot.hopper_id, self.robot.n_total_dofs-1,
                               contactDamping=1000, contactStiffness=1.0,
                               lateralFriction=1.0, spinningFriction=0.1, restitution=0.0)
        self._p.changeDynamics(self.floor_id, -1, lateralFriction=0.5, spinningFriction=0.0, restitution=1.0,
                               contactDamping=0, contactStiffness=1.0)

        self._p.stepSimulation()
        self.past_obs_array.clear()
        self.past_bact_array.clear()

        self.update_extended_observation()

        return self.obs

    def construct_past_traj_window(self):
        # st, ... st-9, at, ..., at-9
        # call this before s_t+1 enters deque
        # order does not matter as long as it is the same in policy & expert batch
        # print(list(self.past_obs_array) + list(self.past_act_array))
        return list(self.past_obs_array) + list(self.past_bact_array)

    def step(self, a):
        if self.train_dyn:
            # in hopper case, env_action is 7D, state-dependent contact coeffs (4D) and battery level (3D)
            env_action = a
            robo_action = np.array(self.past_bact_array[0])       # after tanh
        else:
            robo_action = a
            robo_action = np.tanh(robo_action)
            # update past_bact after tanh
            utils.push_recent_value(self.past_bact_array, robo_action)

            # env_pi_obs = self.get_foot_local_obs(robo_action)
            env_pi_obs = np.concatenate((self.past_obs_array[0], robo_action))

            env_pi_obs_nn = utils.wrap(env_pi_obs, is_cuda=self.cuda_env)

            ind = self.np_random.choice(len(self.dyn_actor_critics))
            with torch.no_grad():
                _, env_action_nn, _, self.recurrent_hidden_states = self.dyn_actor_critics[ind].act(
                    env_pi_obs_nn, self.recurrent_hidden_states, self.masks, deterministic=False
                )
            env_action = utils.unwrap(env_action_nn, is_cuda=self.cuda_env)

        if self.act_noise:
            robo_action = utils.perturb(robo_action, 0.05, self.np_random)

        # when call info, should call before sim_step() (append s_t+1 later)
        # this info will be used to construct D input outside.
        past_info = self.construct_past_traj_window()

        _, dq_old = self.robot.get_q_dq(self.robot.ctrl_dofs)

        for _ in range(self.control_skip):
            self.robot.apply_action(robo_action)
            # self.apply_scale_clip_conf_from_pi_easy(env_action)
            battery_levels = self.set_con_coeff_and_return_battery_level(env_action)
            self.robot.apply_action(robo_action * battery_levels)
            self._p.stepSimulation()
            if self.render:
                time.sleep(self._ts * 0.5)
            self.timer += 1
        self.robot.update_x()
        self.update_extended_observation()
        past_info += [self.past_obs_array[0]]  # s_t+1

        obs_unnorm = np.array(self.past_obs_array[0]) / self.robot.obs_scaling

        reward = 3.0  # alive bonus
        reward += self.get_ave_dx()
        # print("v", self.get_ave_dx())
        reward += -0.5 * np.square(robo_action).sum()
        # print("act norm", -0.5 * np.square(robo_action).sum())

        q = np.array(obs_unnorm[2:5])
        pos_mid = 0.5 * (self.robot.ll + self.robot.ul)
        q_scaled = 2 * (q - pos_mid) / (self.robot.ul - self.robot.ll)
        joints_at_limit = np.count_nonzero(np.abs(q_scaled) > 0.97)
        reward += -3.0 * joints_at_limit
        # print("jl", -2.0 * joints_at_limit)

        dq = np.array(obs_unnorm[8:11])
        reward -= np.minimum(np.sum(np.abs(dq - dq_old)) * 0.05, 5.0)
        # print(np.minimum(np.sum(np.abs(dq - dq_old)) * 0.05, 5.0))

        height = obs_unnorm[0]
        # ang = self._p.getJointState(self.robot.hopper_id, 2)[0]

        # print(dq)
        # print(height)
        # print("ang", ang)

        not_done = (np.abs(dq) < 50).all() and (height > 0.6) and (height < 1.8)

        return self.obs, reward, not not_done, {"sas_window": past_info}

    def set_con_coeff_and_return_battery_level(self, con_f):
        foot_link = self.robot.n_total_dofs - 1
        this_fric = np.tanh(con_f)   # [-1 ,1]

        this_fric[0:2] = (this_fric[0:2] + 1) / 2.0 * 5.0     # 0 ~ 5, lateral & spin Fric
        this_fric[2] = (this_fric[2] + 1) / 2.0 * 15.0       # 0 ~ 10, restitution
        this_fric[3] = (this_fric[3] + 1) / 2.0 * 2.0 + 1.0        # 1 ~ 3, log (damping/2)
        this_fric[3] = np.exp(this_fric[3]) * 2                     # 20 ~ 2000, damping

        # print(this_fric)

        self._p.changeDynamics(self.robot.hopper_id, foot_link,
                               lateralFriction=this_fric[0], spinningFriction=this_fric[1],
                               restitution=this_fric[2], contactDamping=this_fric[3], contactStiffness=1.0)

        # battery level from -0.5 to 1.5
        battery_level = (this_fric[4:7] + 0.5)
        return battery_level

    def get_dist(self):
        return self.robot.x

    def get_ave_dx(self):
        if self.robot.last_x:
            return (self.robot.x - self.robot.last_x) / (self.control_skip * self._ts)
        else:
            return 0.0

    def update_extended_observation(self):
        # in out dyn policy setting, the obs is actually cat(st,at)

        self.obs = self.robot.get_robot_observation()

        if self.correct_obs_dx:
            dx = self.get_ave_dx() * self.robot.obs_scaling[5]
            self.obs[5] = dx

        if self.obs_noise:
            self.obs = utils.perturb(self.obs, 0.1, self.np_random)

        utils.push_recent_value(self.past_obs_array, self.obs)

        if self.train_dyn:
            obs_nn = utils.wrap(self.obs, is_cuda=self.cuda_env)
            with torch.no_grad():
                _, action_nn, _, self.recurrent_hidden_states = self.hopper_actor_critic.act(
                    obs_nn, self.recurrent_hidden_states, self.masks, deterministic=False
                )
            action = utils.unwrap(action_nn, is_cuda=self.cuda_env)
            action = np.tanh(action)

            # Store action after tanh (-1,1)
            utils.push_recent_value(self.past_bact_array, action)

            # self.obs = self.get_foot_local_obs(action)
            self.obs = np.concatenate((self.past_obs_array[0], action))

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s

    def cam_track_torso_link(self):
        distance = 3
        yaw = 0
        torso_x = self._p.getLinkState(self.robot.hopper_id, 2, computeForwardKinematics=1)[0]
        self._p.resetDebugVisualizerCamera(distance, yaw, -20, torso_x)
