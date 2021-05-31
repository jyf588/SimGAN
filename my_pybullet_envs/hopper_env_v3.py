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
import pybullet_data
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
from my_pybullet_envs import utils
from collections import deque

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class HopperURDFEnvV3(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 render=True,
                 init_noise=True,
                 act_noise=True,
                 obs_noise=True,
                 control_skip=10,

                 correct_obs_dx=True,  # if need to correct dx obs,

                 soft_floor_env=False,
                 deform_floor_env=False,
                 low_power_env=False,
                 emf_power_env=False,
                 heavy_head_env=False,

                 randomization_train=False,
                 randomization_train_addi=False,

                 acc_pen_weight=0.05
                 ):
        self.render = render
        self.init_noise = init_noise
        self.obs_noise = obs_noise
        self.act_noise = act_noise
        self.control_skip = int(control_skip)
        self._ts = 1. / 500.
        self.correct_obs_dx = correct_obs_dx

        self.soft_floor_env = soft_floor_env
        self.deform_floor_env = deform_floor_env
        self.low_power_env = low_power_env
        self.emf_power_env = emf_power_env
        self.heavy_head_env = heavy_head_env

        self.randomization_train = randomization_train
        self.randomization_train_addi = randomization_train_addi
        if randomization_train_addi:
            assert randomization_train
        self.randomize_params = {}

        self.acc_pen_weight = acc_pen_weight

        if self.render:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.np_random = None
        self.robot = HopperURDF(init_noise=self.init_noise,
                                time_step=self._ts,
                                np_random=self.np_random,
                                heavy_head=self.heavy_head_env)
        self.seed(0)  # used once temporarily, will be overwritten outside though superclass api
        self.viewer = None
        self.timer = 0

        self.floor_id = None

        self.past_obs_array = deque(maxlen=10)
        self.past_act_array = deque(maxlen=10)

        self.obs = []
        self.reset()  # and update init obs

        self.action_dim = len(self.robot.ctrl_dofs)
        self.act = [0.0] * len(self.robot.ctrl_dofs)
        self.action_space = gym.spaces.Box(low=np.array([-1.] * self.action_dim),
                                           high=np.array([+1.] * self.action_dim))
        self.obs_dim = len(self.obs)
        obs_dummy = np.array([1.12234567] * self.obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf * obs_dummy, high=np.inf * obs_dummy)

    def reset(self):
        if self.deform_floor_env:
            self._p.resetSimulation(self._p.RESET_USE_DEFORMABLE_WORLD)
        else:
            self._p.resetSimulation()

        self._p.setTimeStep(self._ts)
        self._p.setGravity(0, 0, -10)
        self.timer = 0

        self._p.setPhysicsEngineParameter(numSolverIterations=100)

        self.robot.reset(self._p)
        self.robot.update_x(reset=True)
        # # should be after reset!

        if self.soft_floor_env:
            self.floor_id = self._p.loadURDF(
                os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0.0], useFixedBase=1
            )
            self._p.changeDynamics(self.floor_id, -1, lateralFriction=0.8)  # match gym
            self._p.changeDynamics(self.floor_id, -1, restitution=0.5)

            self._p.changeDynamics(self.floor_id, -1, contactDamping=100.0, contactStiffness=600.0)
            for ind in range(self.robot.n_total_dofs):
                self._p.changeDynamics(self.robot.hopper_id, ind, contactDamping=100.0, contactStiffness=600.0)
        elif self.deform_floor_env:
            self.floor_id = self._p.loadURDF(
                os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, -10.10], useFixedBase=1
            )
            self._p.changeDynamics(self.floor_id, -1, lateralFriction=100.0)    # make large enough

            # something is wrong with this API
            # seems that you must add cube.obj to pybullet_data folder for this to work
            # it cannot search relative path in the repo
            _ = self._p.loadSoftBody(os.path.join(currentdir,  "assets/cube_fat.obj"),
                                     basePosition=[7, 0, -5.05], scale=20, mass=20.,
                                     useNeoHookean=1,
                                     useBendingSprings=1, useMassSpring=1, springElasticStiffness=1500,
                                     springDampingStiffness=50, springDampingAllDirections=1,
                                     useSelfCollision=0,
                                     frictionCoeff=1.0, useFaceContact=1)
        else:
            # source env
            if self.randomization_train:
                self.set_randomize_params()
                self.robot.randomize_robot(
                    self.randomize_params["mass_scale"],
                    self.randomize_params["inertia_scale"],
                    self.randomize_params["power_scale"],
                    self.randomize_params["joint_damping"]
                )
            fric = self.randomize_params["friction"] if self.randomization_train else 0.8  # match gym
            resti = self.randomize_params["restitution"] if self.randomization_train else 0.5
            spinfric = self.randomize_params["spinfric"] if self.randomization_train_addi else 0.0
            damp = self.randomize_params["damping"] if self.randomization_train_addi else 2000

            self.floor_id = self._p.loadURDF(
                os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0.0], useFixedBase=1
            )

            self._p.changeDynamics(self.floor_id, -1,
                                   lateralFriction=fric, restitution=resti,
                                   contactStiffness=1.0, contactDamping=damp,
                                   spinningFriction=spinfric)

            self._p.changeDynamics(self.robot.hopper_id, self.robot.n_total_dofs - 1,
                                   lateralFriction=1.0, restitution=1.0,
                                   contactStiffness=1.0, contactDamping=0.0,
                                   spinningFriction=0.0)

        if self.low_power_env:
            self.robot.max_forces[2] = 100

        self._p.changeDynamics(self.robot.hopper_id, self.robot.n_total_dofs-1, lateralFriction=1.0)   # match gym

        self._p.stepSimulation()

        self.past_obs_array.clear()
        self.past_act_array.clear()

        self.update_extended_observation()

        return self.obs

    def construct_past_traj_window(self):
        # st, ... st-9, at, ..., at-9
        # call this before s_t+1 enters deque
        # order does not matter as long as it is the same in policy & expert batch
        # print(list(self.past_obs_array) + list(self.past_act_array))
        return list(self.past_obs_array) + list(self.past_act_array)

    def set_randomize_params(self):
        self.randomize_params = {
            # robot
            "mass_scale": self.np_random.uniform(0.5, 1.5, 4),
            "inertia_scale": self.np_random.uniform(0.4, 1.8, 4),
            "power_scale": self.np_random.uniform(0.5, 1.5, 3),
            "joint_damping": self.np_random.uniform(0.2, 3.0, 3),
            # act/obs latency
            "act_latency": self.np_random.uniform(0, 0.02),
            "obs_latency": self.np_random.uniform(0, 0.02),
            # contact
            "friction": self.np_random.uniform(0.4, 1.5),
            "restitution": self.np_random.uniform(0., 1.5),
            # contact additional
            "damping": np.power(10, self.np_random.uniform(1.2, 3.2)),
            "spinfric": self.np_random.uniform(0.0, 0.2),
        }

    def step(self, a):

        # self.act = np.clip(a, -1, 1)
        self.act = np.tanh(a)

        # ***push in deque the a after tanh
        utils.push_recent_value(self.past_act_array, self.act)
        past_info = self.construct_past_traj_window()

        act_latency = self.randomize_params["act_latency"] if self.randomization_train else 0
        a0 = np.array(self.past_act_array[0])
        a1 = np.array(self.past_act_array[1])
        interp = act_latency / 0.02
        self.act = a0 * (1 - interp) + a1 * interp

        if self.act_noise:
            self.act = utils.perturb(self.act, 0.05, self.np_random)

        if self.emf_power_env:
            _, dq = self.robot.get_q_dq(self.robot.ctrl_dofs)
            max_force_ratio = np.clip(1 - dq/10, 0, 1)
            self.act *= max_force_ratio

        _, dq_old = self.robot.get_q_dq(self.robot.ctrl_dofs)

        for _ in range(self.control_skip):
            # action is in not -1,1
            if a is not None:
                self.robot.apply_action(self.act)
            self._p.stepSimulation()
            if self.render:
                time.sleep(self._ts * 0.5)
            self.timer += 1
        self.robot.update_x()
        self.update_extended_observation()      # and push  s_t+1 to self.past_obs_array[0]
        past_info += [self.past_obs_array[0]]  # s_t+1

        obs_unnorm = np.array(self.obs) / self.robot.obs_scaling

        reward = 3.0  # alive bonus
        reward += self.get_ave_dx()
        # print("v", self.get_ave_dx())
        reward += -0.5 * np.square(a).sum()
        # print("act norm", -0.4 * np.square(a).sum())

        q = np.array(obs_unnorm[2:5])
        pos_mid = 0.5 * (self.robot.ll + self.robot.ul)
        q_scaled = 2 * (q - pos_mid) / (self.robot.ul - self.robot.ll)
        joints_at_limit = np.count_nonzero(np.abs(q_scaled) > 0.97)
        reward += -3.0 * joints_at_limit
        # print("jl", -2.0 * joints_at_limit)

        dq = np.array(obs_unnorm[8:11])
        reward -= np.minimum(np.sum(np.abs(dq - dq_old)) * self.acc_pen_weight, 5.0)
        # print(np.minimum(np.sum(np.abs(dq - dq_old)) * self.acc_pen_weight, 5.0))

        height = obs_unnorm[0]
        # ang = self._p.getJointState(self.robot.hopper_id, 2)[0]

        # print(joints_dq)
        # print(height)
        # print("ang", ang)

        not_done = (np.abs(dq) < 50).all() and (height > 0.6) and (height < 1.8)

        return self.obs, reward, not not_done, {"sas_window": past_info}

    def get_dist(self):
        return self.robot.x

    def get_ave_dx(self):
        if self.robot.last_x:
            return (self.robot.x - self.robot.last_x) / (self.control_skip * self._ts)
        else:
            return 0.0

    def update_extended_observation(self):
        self.obs = self.robot.get_robot_observation()

        if self.correct_obs_dx:
            dx = self.get_ave_dx() * self.robot.obs_scaling[5]
            self.obs[5] = dx

        if self.obs_noise:
            self.obs = utils.perturb(self.obs, 0.1, self.np_random)

        utils.push_recent_value(self.past_obs_array, self.obs)

        obs_latency = self.randomize_params["obs_latency"] if self.randomization_train else 0
        s0 = np.array(self.past_obs_array[0])
        s1 = np.array(self.past_obs_array[1])
        interp = obs_latency / 0.02
        self.obs = s0 * (1 - interp) + s1 * interp

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random  # use the same np_randomizer for robot as for env
        return [seed]

    def getSourceCode(self):
        s = inspect.getsource(type(self))
        s = s + inspect.getsource(type(self.robot))
        return s

    def cam_track_torso_link(self):
        distance = 2.0
        yaw = 0
        torso_x = self._p.getLinkState(self.robot.hopper_id, 2, computeForwardKinematics=1)[0]
        self._p.resetDebugVisualizerCamera(distance, yaw, -20, [torso_x[0], 0.0, 0.1])
