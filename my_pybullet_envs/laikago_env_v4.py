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
import pybullet_data
import time
import gym, gym.utils.seeding, gym.spaces
import numpy as np
from my_pybullet_envs import utils
from collections import deque

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class LaikagoBulletEnvV4(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 render=True,
                 init_noise=True,
                 act_noise=True,
                 obs_noise=True,
                 control_skip=10,

                 max_tar_vel=2.5,
                 energy_weight=0.1,
                 jl_weight=0.5,
                 ab=4.5,
                 q_pen_weight=0.35,
                 acc_pen_weight=0.03,
                 vel_r_weight=4.0,

                 enlarge_act_range=0.0,     # during data collection, make pi softer

                 soft_floor_env=False,
                 deform_floor_env=False,
                 low_power_env=False,
                 emf_power_env=False,
                 heavy_leg_env=False,
                 randomization_train=False,
                 randomization_train_addi=False,
                 randomforce_train=False,
                 sysid_data_collection=False,

                 final_test=False
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

        self.enlarge_act_range = enlarge_act_range

        self.soft_floor_env = soft_floor_env
        self.deform_floor_env = deform_floor_env
        self.low_power_env = low_power_env
        self.emf_power_env = emf_power_env
        self.heavy_leg_env = heavy_leg_env
        self.randomization_train = randomization_train
        self.randomization_train_addi = randomization_train_addi
        if randomization_train_addi:
            assert randomization_train
        self.randomforce_train = randomforce_train
        self.sysid_data_collection = sysid_data_collection

        self.final_test = final_test

        self.randomize_params = {}

        if self.render:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # self._p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

        self.np_random = None
        self.robot = LaikagoBullet(init_noise=self.init_noise,
                                   time_step=self._ts,
                                   np_random=self.np_random,
                                   heavy_leg=self.heavy_leg_env,
                                   no_init_vel=self.sysid_data_collection)
        self.seed(0)  # used once temporarily, will be overwritten outside though superclass api
        self.viewer = None
        self.timer = 0

        self.floor_id = None
        self.velx = 0

        # self.behavior_past_obs_t_idx = [0, 3, 6, 9]  # t-3. t-6. t-9
        self.behavior_past_obs_t_idx = np.array([0, 4, 8])

        # self.behavior_past_obs_t_idx = [0]
        self.past_obs_array = deque(maxlen=10)
        self.past_act_array = deque(maxlen=10)

        self.reset_const = 100
        self.reset_counter = self.reset_const    # do a hard reset first
        self.init_state = None
        obs = self.reset()  # and update init obs

        self.action_dim = len(self.robot.ctrl_dofs)
        self.act = [0.0] * len(self.robot.ctrl_dofs)
        self.action_space = gym.spaces.Box(low=np.array([-1.] * self.action_dim),
                                           high=np.array([+1.] * self.action_dim))
        self.obs_dim = len(obs)
        # print(self.obs_dim)
        obs_dummy = np.array([1.12234567] * self.obs_dim)
        self.observation_space = gym.spaces.Box(low=-np.inf * obs_dummy, high=np.inf * obs_dummy)

        # self.b2d_feat_select = self.feature_selection_B2D_laika_v3

    def reset(self):

        if self.reset_counter < self.reset_const:
            self.reset_counter += 1

            self._p.restoreState(self.init_state)
            self.robot.soft_reset(self._p)
        else:
            if self.deform_floor_env or self.soft_floor_env or self.sysid_data_collection:
                self._p.resetSimulation(self._p.RESET_USE_DEFORMABLE_WORLD)
                # always use hard reset if soft-floor-env
                # always use hard reset if sysid collection
            else:
                self._p.resetSimulation()
                self.reset_counter = 0      # use soft reset later on

            self._p.setTimeStep(self._ts)
            self._p.setGravity(0, 0, -10)
            self._p.setPhysicsEngineParameter(numSolverIterations=100)

            self.robot.reset(self._p)

            if self.soft_floor_env:
                self.floor_id = self._p.loadURDF(
                  os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0.0], useFixedBase=1
                )
                # reset
                for ind in self.robot.feet:
                    self._p.changeDynamics(self.robot.go_id, ind,
                                           contactDamping=100, contactStiffness=100)
                self._p.changeDynamics(self.floor_id, -1, contactDamping=50, contactStiffness=100)
            elif self.deform_floor_env:
                self.floor_id = self._p.loadURDF(
                    os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, -10.02], useFixedBase=1
                )

                _ = self._p.loadSoftBody(os.path.join(currentdir,  "assets/cube_fat.obj"),
                                         basePosition=[7, 0, -5], scale=20, mass=4000.,
                                         useNeoHookean=0,
                                         useBendingSprings=1, useMassSpring=1, springElasticStiffness=60000,
                                         springDampingStiffness=150, springDampingAllDirections=1,
                                         useSelfCollision=0,
                                         frictionCoeff=1.0, useFaceContact=1)
            else:
                if self.randomization_train:
                    self.set_randomize_params()
                    self.robot.randomize_robot(
                        self.randomize_params["mass_scale"],
                        self.randomize_params["inertia_scale"],
                        self.randomize_params["power_scale"],
                        self.randomize_params["joint_damping"]
                    )

                fric = self.randomize_params["friction"] if self.randomization_train else 0.5
                resti = self.randomize_params["restitution"] if self.randomization_train else 0.0
                spinfric = self.randomize_params["spinfric"] if self.randomization_train_addi else 0.0
                damp = self.randomize_params["damping"] if self.randomization_train_addi else 2000

                self.floor_id = self._p.loadURDF(
                  os.path.join(currentdir, 'assets/plane.urdf'), [0, 0, 0.0], useFixedBase=1
                )

                self._p.changeDynamics(self.floor_id, -1,
                                       lateralFriction=fric, restitution=resti,
                                       contactStiffness=1.0, contactDamping=damp,
                                       spinningFriction=spinfric)
                for ind in self.robot.feet:
                    self._p.changeDynamics(self.robot.go_id, ind,
                                           lateralFriction=1.0, restitution=1.0,
                                           contactStiffness=1.0, contactDamping=0.0,
                                           spinningFriction=0.0)

                if self.sysid_data_collection:
                    # try sysID for emf
                    self.emf_power_env = True

            self.init_state = self._p.saveState()

        if self.low_power_env:
            # deprecated
            self.robot.max_forces = [30.0] * 3 + [15.0] * 3 + [30.0] * 6

        if not self.sysid_data_collection:
            self._p.stepSimulation()        # may not be helpful if want to reproduce traj

        for foot in self.robot.feet:
            cps = self._p.getContactPoints(self.robot.go_id, self.floor_id, foot, -1)
            if len(cps) > 0:
                print("warning")

        self.timer = 0
        self.past_obs_array.clear()
        self.past_act_array.clear()
        #
        # self.normal_forces = np.array([[0., 0, 0, 0]])
        # self.tan_forces = np.array([[0., 0, 0, 0]])

        obs = self.get_extended_observation()

        return np.array(obs)

    def set_randomize_params(self):
        self.randomize_params = {
            # robot
            "mass_scale": self.np_random.uniform(0.8, 1.2, 13),
            "inertia_scale": self.np_random.uniform(0.5, 1.5, 13),
            "power_scale": self.np_random.uniform(0.8, 1.2, 12),
            "joint_damping": self.np_random.uniform(0.2, 2.0, 12),
            # act/obs latency
            "act_latency": self.np_random.uniform(0, 0.02),
            "obs_latency": self.np_random.uniform(0, 0.02),
            # contact
            "friction": self.np_random.uniform(0.4, 1.25),
            "restitution": self.np_random.uniform(0., 0.5),
            # contact additional
            # uniform does not make sense for this param
            # "damping": np.power(10, self.np_random.uniform(1.0, 3.0)) * 5,
            # "spinfric": self.np_random.uniform(0.1, 0.5),
            # "damping": np.power(10, self.np_random.uniform(2.0, 3.0)) * 2,
            # "spinfric": self.np_random.uniform(0.0, 0.2),
            "damping": np.power(10, self.np_random.uniform(2.0, 3.2)),
            "spinfric": self.np_random.uniform(0.0, 0.1),
        }

    def step(self, a):

        root_pos, _ = self.robot.get_link_com_xyz_orn(-1)
        x_0 = root_pos[0]

        # TODO: parameter space noise.
        # make pi softer during data collection, different from hidden act_noise below
        # print(self.enlarge_act_range)
        a = utils.perturb(a, self.enlarge_act_range, self.np_random)
        a = np.tanh(a)

        # ***push in deque the a after tanh
        utils.push_recent_value(self.past_act_array, a)

        act_latency = self.randomize_params["act_latency"] if self.randomization_train else 0

        a0 = np.array(self.past_act_array[0])
        a1 = np.array(self.past_act_array[1])
        interp = act_latency / 0.02
        a = a0 * (1 - interp) + a1 * interp

        if self.act_noise:
            a = utils.perturb(a, 0.05, self.np_random)

        if self.emf_power_env:
            _, dq = self.robot.get_q_dq(self.robot.ctrl_dofs)
            max_force_ratio = np.clip(1 - dq/15., 0, 1)
            a *= max_force_ratio

        past_info = self.construct_past_traj_window()

        _, dq_old = self.robot.get_q_dq(self.robot.ctrl_dofs)

        for _ in range(self.control_skip):
            if a is not None:
                self.act = a
                self.robot.apply_action(a)

            if self.randomforce_train:
                for foot_ind, link in enumerate(self.robot.feet):
                    # first dim represents fz
                    fz = np.random.uniform(-80, 80)
                    # second dim represents fx
                    fx = np.random.uniform(-80, 80)
                    # third dim represents fy
                    fy = np.random.uniform(-80, 80)

                    utils.apply_external_world_force_on_local_point(self.robot.go_id, link,
                                                                    [fx, fy, fz],
                                                                    [0, 0, 0],
                                                                    self._p)
            self._p.stepSimulation()
            if self.render:
                time.sleep(self._ts * 1.5)
            self.timer += 1

        root_pos, _ = self.robot.get_link_com_xyz_orn(-1)
        x_1 = root_pos[0]
        self.velx = (x_1 - x_0) / (self.control_skip * self._ts)

        q, dq = self.robot.get_q_dq(self.robot.ctrl_dofs)

        reward = self.ab  # alive bonus
        tar = np.minimum(self.timer / 500, self.max_tar_vel)
        reward += np.minimum(self.velx, tar) * self.vel_r_weight
        # print("v", self.velx, "tar", tar)
        reward += -self.energy_weight * np.square(a).sum()
        # print("act norm", -self.energy_weight * np.square(a).sum())

        pos_mid = 0.5 * (self.robot.ll + self.robot.ul)
        q_scaled = 2 * (q - pos_mid) / (self.robot.ul - self.robot.ll)
        joints_at_limit = np.count_nonzero(np.abs(q_scaled) > 0.97)
        reward += -self.jl_weight * joints_at_limit
        # print("jl", -self.jl_weight * joints_at_limit)

        # instead pen joint vels, pen joint delta vels (accs)
        reward += -np.minimum(np.sum(np.abs(dq - dq_old)) * self.acc_pen_weight, 5.0)
        # print(np.minimum(np.sum(np.abs(dq - dq_old)) * self.dq_pen_weight, 5.0))
        weight = np.array([2.0, 1.0, 1.0] * 4)
        reward += -np.minimum(np.sum(np.square(q - self.robot.init_q) * weight) * self.q_pen_weight, 5.0)

        y_1 = root_pos[1]
        reward += -y_1 * 0.5
        # print("dev pen", -y_1*0.5)
        height = root_pos[2]

        obs = self.get_extended_observation()

        _, root_orn = self.robot.get_link_com_xyz_orn(-1)
        rpy = self._p.getEulerFromQuaternion(root_orn)
        diff = (np.array(rpy) - [1.5708, 0.0, 1.5708])

        if self.final_test:
            # do not terminate as in multi-task it may be moving toward y axis
            diff[0] = diff[1] = diff[2] = 0.0

        # during final testing, terminate more tolerantly, some policy can recover from low height
        # esp. in deform env where floor can be lower than 0.
        height_thres = 0.15 if self.final_test else 0.3
        not_done = (np.abs(dq) < 90).all() and (height > height_thres) and (np.abs(diff) < 1.2).all()

        # during testing limit time, since final good policy might go beyond end of mattress
        if self.final_test and self.timer >= 3500:
            not_done = False

        past_info += [self.past_obs_array[0]]       # s_t+1

        return obs, reward, not not_done, {"sas_window": past_info}

    def construct_past_traj_window(self):
        # st, ... st-9, at, ..., at-9
        # call this before s_t+1 enters deque
        # order does not matter as long as it is the same in policy & expert batch
        # print(list(self.past_obs_array) + list(self.past_act_array))
        return list(self.past_obs_array) + list(self.past_act_array)

    def get_dist(self):
        return self.robot.get_link_com_xyz_orn(-1)[0][0]

    def get_ave_dx(self):
        return self.velx

    def get_extended_observation(self):
        obs = self.robot.get_robot_observation(with_vel=False)

        if self.obs_noise:
            obs = utils.perturb(obs, 0.1, self.np_random)

        utils.push_recent_value(self.past_obs_array, obs)

        obs_latency = self.randomize_params["obs_latency"] if self.randomization_train else 0

        obs_all_0 = utils.select_and_merge_from_s_a(
            s_mt=list(self.past_obs_array),
            a_mt=list(self.past_act_array),
            s_idx=self.behavior_past_obs_t_idx,
            a_idx=np.array([])
        )
        obs_all_1 = utils.select_and_merge_from_s_a(
            s_mt=list(self.past_obs_array),
            a_mt=list(self.past_act_array),
            s_idx=self.behavior_past_obs_t_idx + 1,
            a_idx=np.array([])
        )

        interp = obs_latency / 0.02
        obs_all = obs_all_0 * (1 - interp) + obs_all_1 * interp

        return list(obs_all)

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
