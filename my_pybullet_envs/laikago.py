#  Copyright 2020 Google LLC
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

# DoF index, DoF (joint) Name, joint type (0 means hinge joint), joint lower and upper limits, child link of this joint
# (0, b'FR_hip_motor_2_chassis_joint', 0) -0.873 1.0472 b'FR_hip_motor'
# (1, b'FR_upper_leg_2_hip_motor_joint', 0) -1.3 3.4 b'FR_upper_leg'
# (2, b'FR_lower_leg_2_upper_leg_joint', 0) -2.164 0.0 b'FR_lower_leg'
# (3, b'jtoeFR', 4) 0.0 -1.0 b'toeFR'
# (4, b'FL_hip_motor_2_chassis_joint', 0) -0.873 1.0472 b'FL_hip_motor'
# (5, b'FL_upper_leg_2_hip_motor_joint', 0) -1.3 3.4 b'FL_upper_leg'
# (6, b'FL_lower_leg_2_upper_leg_joint', 0) -2.164 0.0 b'FL_lower_leg'
# (7, b'jtoeFL', 4) 0.0 -1.0 b'toeFL'
# (8, b'RR_hip_motor_2_chassis_joint', 0) -0.873 1.0472 b'RR_hip_motor'
# (9, b'RR_upper_leg_2_hip_motor_joint', 0) -1.3 3.4 b'RR_upper_leg'
# (10, b'RR_lower_leg_2_upper_leg_joint', 0) -2.164 0.0 b'RR_lower_leg'
# (11, b'jtoeRR', 4) 0.0 -1.0 b'toeRR'
# (12, b'RL_hip_motor_2_chassis_joint', 0) -0.873 1.0472 b'RL_hip_motor'
# (13, b'RL_upper_leg_2_hip_motor_joint', 0) -1.3 3.4 b'RL_upper_leg'
# (14, b'RL_lower_leg_2_upper_leg_joint', 0) -2.164 0.0 b'RL_lower_leg'
# (15, b'jtoeRL', 4) 0.0 -1.0 b'toeRL'
# ctrl dofs: [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]

import numpy as np
from my_pybullet_envs import utils
import pybullet
from scipy.spatial.transform import Rotation as R       # need matrix->quat

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


class LaikagoBullet:
    def __init__(self,
                 init_noise=True,
                 time_step=1. / 500,
                 np_random=None,
                 heavy_leg=False,
                 no_init_vel=False
                 ):

        self.init_noise = init_noise

        self._ts = time_step
        self.np_random = np_random

        self.base_init_pos = np.array([0, 0, .56])  # starting position
        self.base_init_euler = np.array([1.5708, 0, 1.5708])  # starting orientation

        self.feet = [3, 7, 11, 15]

        self.nominal_max_forces = [30.0] * 12
        self.max_forces = self.nominal_max_forces.copy()     # joint torque limits

        # ang vel scaled to 0.2, dq scaled to 0.04
        self.robo_obs_scale = np.array([1.0] * (1 + 9 + 3 + 12 + 12) + [0.2] * 3 + [0.04] * 12)

        self.init_q = [0.0, 0.0, -0.5] * 4
        self.ctrl_dofs = []

        self.heavy_leg = heavy_leg
        self.no_init_vel = no_init_vel

        self._p = None  # bullet session to connect to
        self.go_id = -2  # bullet id for the loaded humanoid, to be overwritten
        self.torque = None  # if using torque control, the current torque vector to apply

        self.ll = None  # stores joint lower limits
        self.ul = None  # stores joint upper limits

        # for domain randomization
        self.last_mass_scaling = np.array([1.0] * 13)
        self.last_inertia_scaling = np.array([1.0] * 13)

    def reset(
            self,
            bullet_client
    ):
        self._p = bullet_client

        base_init_pos, base_init_euler, base_init_vel = self.get_perturbed_base_state()

        if self.heavy_leg:
            path = "assets/laikago/laikago_toes_limits_dragging.urdf"
        else:
            path = "assets/laikago/laikago_toes_limits.urdf"

        self.go_id = self._p.loadURDF(os.path.join(currentdir,
                                                   path),
                                      list(base_init_pos-[0.043794, 0.0, 0.03]),
                                      list(self._p.getQuaternionFromEuler(list(base_init_euler))),
                                      flags=self._p.URDF_USE_SELF_COLLISION,
                                      useFixedBase=0)

        # self.print_all_joints_info()
        self._p.resetBaseVelocity(self.go_id, base_init_vel[:3], base_init_vel[3:])

        for j in range(self._p.getNumJoints(self.go_id)):
            self._p.changeDynamics(self.go_id, j, jointDamping=0.5)  # TODO

        if len(self.ctrl_dofs) == 0:
            for j in range(self._p.getNumJoints(self.go_id)):
                info = self._p.getJointInfo(self.go_id, j)
                joint_type = info[2]
                if joint_type == self._p.JOINT_PRISMATIC or joint_type == self._p.JOINT_REVOLUTE:
                    self.ctrl_dofs.append(j)

        # print("ctrl dofs:", self.ctrl_dofs)

        self.reset_joints(self.init_q, np.array([0.0] * len(self.ctrl_dofs)))

        # turn off root default control:
        # use torque control
        self._p.setJointMotorControlArray(
            bodyIndex=self.go_id,
            jointIndices=self.ctrl_dofs,
            controlMode=self._p.VELOCITY_CONTROL,
            forces=[0.0] * len(self.ctrl_dofs))
        self.torque = [0.0] * len(self.ctrl_dofs)

        self.ll = np.array([self._p.getJointInfo(self.go_id, i)[8] for i in self.ctrl_dofs])
        self.ul = np.array([self._p.getJointInfo(self.go_id, i)[9] for i in self.ctrl_dofs])

        assert len(self.ctrl_dofs) == len(self.init_q)
        assert len(self.max_forces) == len(self.ctrl_dofs)
        assert len(self.max_forces) == len(self.ll)

    def get_perturbed_base_state(self):
        vel = 0.2 if not self.no_init_vel else 0.0
        if self.init_noise:
            base_init_pos = utils.perturb(self.base_init_pos, 0.03, self.np_random)
            base_init_euler = utils.perturb(self.base_init_euler, 0.1, self.np_random)
            base_init_vel = utils.perturb([0.0] * 6, vel, self.np_random)
        else:
            base_init_pos = self.base_init_pos
            base_init_euler = self.base_init_euler
            base_init_vel = [0.0] * 6
            # base_init_pos= np.array([-0.03,  0.03,  0.57])
            # base_init_euler=[1.53, - 0.08,  1.61]
            # base_init_vel=[-0.04,  0.07, -0.18,  0.01, -0.18, 0.19]

        return base_init_pos, base_init_euler, base_init_vel

    def soft_reset(
            self,
            bullet_client
    ):
        self._p = bullet_client

        base_init_pos, base_init_euler, base_init_vel = self.get_perturbed_base_state()

        self._p.resetBaseVelocity(self.go_id, base_init_vel[:3], base_init_vel[3:])
        self._p.resetBasePositionAndOrientation(self.go_id,
                                                list(base_init_pos),
                                                list(self._p.getQuaternionFromEuler(list(base_init_euler)))
                                                )

        self.reset_joints(self.init_q, np.array([0.0] * len(self.ctrl_dofs)))

    def soft_reset_to_state(self, bullet_client, state_vec):
        # TODO: unfinished, state vec unused
        # state vec following this order:
        # root dq [6]
        # root q [3+4(quat)]
        # all q/dq
        self._p = bullet_client

        base_init_pos, base_init_euler, base_init_vel = self.get_perturbed_base_state()
        # self._p.resetBaseVelocity(self.go_id, base_init_vel[:3], base_init_vel[3:])

        # base_init_euler += [0., 0., 0.00001]

        self._p.resetBaseVelocity(self.go_id, base_init_vel[:3], base_init_vel[3:])
        # self._p.resetBaseVelocity(self.go_id, state_vec[:3], state_vec[3:6])
        # self._p.resetBasePositionAndOrientation(self.go_id,
        #                                         state_vec[6:9],
        #                                         state_vec[9:13])

        self._p.resetBasePositionAndOrientation(self.go_id,
                                                list(base_init_pos),
                                                list(self._p.getQuaternionFromEuler(list(base_init_euler)))
                                                )
        # print(base_init_pos)
        # print(self._p.getQuaternionFromEuler(list(base_init_euler)))
        # print(state_vec[6:9])
        # print(state_vec[9:13])

        qdq = state_vec[13:]
        assert len(qdq) == len(self.ctrl_dofs) * 2
        init_noise_old = self.init_noise
        self.init_noise = False
        self.reset_joints(qdq[:len(self.ctrl_dofs)], qdq[len(self.ctrl_dofs):])
        self.init_noise = init_noise_old

    def hard_reset_to_state(self, bullet_client, state_vec):
        # TODO: unfinished, state vec unused
        # state vec following this order:
        # root dq [6]
        # root q [3+4(quat)]
        # all q/dq

        self._p = bullet_client

        base_init_pos, base_init_euler, base_init_vel = self.get_perturbed_base_state()
        self._p.resetBaseVelocity(self.go_id, base_init_vel[:3], base_init_vel[3:])

        # print(base_init_pos)
        base_init_pos = np.array(state_vec[6:9])
        # print(base_init_pos)

        # print(base_init_euler)
        base_init_quat = state_vec[9:13]
        # print(pybullet.getEulerFromQuaternion(base_init_quat))
        # # base_init_vel = state_vec[:6]

        self.go_id = self._p.loadURDF(os.path.join(currentdir,
                                                   "assets/laikago/laikago_toes_limits.urdf"),
                                      list(base_init_pos-[0.043794, 0.0, 0.03]),
                                      list(base_init_quat),
                                      flags=self._p.URDF_USE_SELF_COLLISION,
                                      useFixedBase=0)

        for j in range(self._p.getNumJoints(self.go_id)):
            self._p.changeDynamics(self.go_id, j, jointDamping=0.5)  # TODO

        if len(self.ctrl_dofs) == 0:
            for j in range(self._p.getNumJoints(self.go_id)):
                info = self._p.getJointInfo(self.go_id, j)
                joint_type = info[2]
                if joint_type == self._p.JOINT_PRISMATIC or joint_type == self._p.JOINT_REVOLUTE:
                    self.ctrl_dofs.append(j)

        # print("ctrl dofs:", self.ctrl_dofs)

        qdq = state_vec[13:]
        assert len(qdq) == len(self.ctrl_dofs) * 2
        init_noise_old = self.init_noise
        self.init_noise = False
        self.reset_joints(qdq[:len(self.ctrl_dofs)], qdq[len(self.ctrl_dofs):])
        self.init_noise = init_noise_old
        # self.reset_joints(self.init_q, np.array([0.0] * len(self.ctrl_dofs)))

        # turn off root default control:
        # use torque control
        self._p.setJointMotorControlArray(
            bodyIndex=self.go_id,
            jointIndices=self.ctrl_dofs,
            controlMode=self._p.VELOCITY_CONTROL,
            forces=[0.0] * len(self.ctrl_dofs))
        self.torque = [0.0] * len(self.ctrl_dofs)

        self.ll = np.array([self._p.getJointInfo(self.go_id, i)[8] for i in self.ctrl_dofs])
        self.ul = np.array([self._p.getJointInfo(self.go_id, i)[9] for i in self.ctrl_dofs])

    def reset_joints(self, q, dq):
        vel = 0.1 if not self.no_init_vel else 0.0
        if self.init_noise:
            init_q = utils.perturb(q, 0.01, self.np_random)
            init_dq = utils.perturb(dq, vel, self.np_random)
        else:
            init_q = q
            init_dq = dq

        for pt, ind in enumerate(self.ctrl_dofs):
            self._p.resetJointState(self.go_id, ind, init_q[pt], init_dq[pt])

    def print_all_joints_info(self):
        for i in range(self._p.getNumJoints(self.go_id)):
            print(self._p.getJointInfo(self.go_id, i)[0:3],
                  self._p.getJointInfo(self.go_id, i)[8], self._p.getJointInfo(self.go_id, i)[9],
                  self._p.getJointInfo(self.go_id, i)[12])

    def apply_action(self, a):

        self.torque = np.array(a) * self.max_forces

        self._p.setJointMotorControlArray(
            bodyIndex=self.go_id,
            jointIndices=self.ctrl_dofs,
            controlMode=self._p.TORQUE_CONTROL,
            forces=self.torque)

    def get_q_dq(self, dofs):
        joints_state = self._p.getJointStates(self.go_id, dofs)
        joints_q = np.array(joints_state)[:, [0]]
        joints_q = np.hstack(joints_q.flatten())
        joints_dq = np.array(joints_state)[:, [1]]
        joints_dq = np.hstack(joints_dq.flatten())
        return joints_q, joints_dq

    def get_link_com_xyz_orn(self, link_id, fk=1):
        # get the world transform (xyz and quaternion) of the Center of Mass of the link
        assert link_id >= -1
        if link_id == -1:
            link_com, link_quat = self._p.getBasePositionAndOrientation(self.go_id)
        else:
            link_com, link_quat, *_ = self._p.getLinkState(self.go_id, link_id, computeForwardKinematics=fk)
        return list(link_com), list(link_quat)

    def get_robot_raw_state_vec(self):
        # state vec following this order:
        # root dq [6]
        # root q [3+4(quat)]
        # all q/dq
        state = []
        base_vel, base_ang_vel = self._p.getBaseVelocity(self.go_id)
        state.extend(base_vel)
        state.extend(base_ang_vel)
        root_pos, root_orn = self.get_link_com_xyz_orn(-1)
        state.extend(root_pos)
        state.extend(root_orn)
        q, dq = self.get_q_dq(self.ctrl_dofs)
        state.extend(q)
        state.extend(dq)
        return state

    def get_state_from_obs_without_vel(self, obs):
        # state vec following this order:
        # root dq [6]
        # root q [3+4(quat)]
        # all q/dq
        # if unknown info (vel) from obs vec, fill 0

        state = []
        base_vel = obs[10:13]
        base_ang_vel = [0., 0, 0]
        state.extend(base_vel)
        state.extend(base_ang_vel)

        root_pos = [0., 0., obs[0]]
        root_rot_mat = np.array(obs[1:10]).reshape(3, 3)
        root_orn = R.from_matrix(root_rot_mat).as_quat()
        state.extend(root_pos)
        state.extend(list(root_orn))

        q = obs[13:25]
        dq = [0.0] * len(q)
        state.extend(q)
        state.extend(dq)

        # print(state)
        return state

    def get_robot_observation(self, with_vel=False):
        obs = []

        # root z and root rot mat (1+9)
        root_pos, root_orn = self.get_link_com_xyz_orn(-1)
        root_x, root_y, root_z = root_pos
        obs.extend([root_z])
        obs.extend(pybullet.getMatrixFromQuaternion(root_orn))

        # root_rot_mat = np.array(obs[1:10]).reshape(3, 3)
        # root_orn_recover = R.from_matrix(root_rot_mat).as_quat()
        # a = pybullet.getEulerFromQuaternion(root_orn_recover)
        # b = pybullet.getEulerFromQuaternion(root_orn)
        #
        # if np.linalg.norm(np.array(a)-b) > 1e-3:
        #     print(np.linalg.norm(np.array(a)-b))

        # root lin vel (3)
        base_vel, base_ang_vel = self._p.getBaseVelocity(self.go_id)
        obs.extend(base_vel)

        # non-root joint q (12)
        q, dq = self.get_q_dq(self.ctrl_dofs)
        obs.extend(q)

        # feet (offset from root) (12)
        for link in self.feet:
            pos, _ = self.get_link_com_xyz_orn(link, fk=1)
            pos[0] -= root_x
            pos[1] -= root_y
            pos[2] -= root_z
            obs.extend(pos)

        length_wo_vel = len(obs)

        obs.extend(base_ang_vel)    # (3)
        obs.extend(dq)      # (12)

        obs = np.array(obs) * self.robo_obs_scale

        if not with_vel:
            obs = obs[:length_wo_vel]

        return list(obs)

    def is_root_com_in_support(self):
        root_com, _ = self._p.getBasePositionAndOrientation(self.go_id)
        feet_max_x = -1000
        feet_min_x = 1000
        feet_max_y = -1000
        feet_min_y = 1000
        for foot in self.feet:
            x, y, _ = list(self.get_link_com_xyz_orn(foot)[0])
            if x > feet_max_x:
                feet_max_x = x
            if x < feet_min_x:
                feet_min_x = x
            if y > feet_max_y:
                feet_max_y = y
            if y < feet_min_y:
                feet_min_y = y
        return (feet_min_x - 0.05 < root_com[0]) and (root_com[0] < feet_max_x + 0.05) \
            and (feet_min_y - 0.05 < root_com[1]) and (root_com[1] < feet_max_y + 0.05)

    def randomize_robot(self, mass_scale, inertia_scale, power_scale, damping_scale):
        # the scales being numpy vectors
        for link_ind, dof in enumerate([-1] + self.ctrl_dofs):
            dyn = self._p.getDynamicsInfo(self.go_id, dof)
            mass = dyn[0] / self.last_mass_scaling[link_ind] * mass_scale[link_ind]
            lid = np.array(dyn[2]) / self.last_inertia_scaling[link_ind] * inertia_scale[link_ind]
            self._p.changeDynamics(self.go_id, dof, mass=mass)
            self._p.changeDynamics(self.go_id, dof, localInertiaDiagonal=tuple(lid))

        for joint_ind, j in enumerate(self.ctrl_dofs):
            self._p.changeDynamics(self.go_id, j, jointDamping=damping_scale[joint_ind])
        self.max_forces = self.nominal_max_forces * power_scale

        self.last_mass_scaling = np.copy(mass_scale)
        self.last_inertia_scaling = np.copy(inertia_scale)


def mirror_foot_pos(rlxyz):
    # rl xyz is 6D
    rxyz = rlxyz[:3]
    lxyz = rlxyz[3:6]
    return [lxyz[0], -lxyz[1], lxyz[2], rxyz[0], -rxyz[1], rxyz[2]]


def mirror_leg_q(rlq):
    return list(rlq[3:6]) + list(rlq[:3])


def mirror_obs(obs_old):

    obs = list(obs_old).copy()
    assert (len(obs) // 37) * 37 == len(obs)

    for i in range(len(obs) // 37):
        obs[i*37: (i+1)*37] = mirror_obs_per_step(obs_old[i*37: (i+1)*37])

    return obs


def mirror_obs_per_step(obs_old):
    # assume with_vel False since not used by G for now.
    # FR, FL, RR, RL

    # root z unchanged
    # change root rot mat

    assert len(obs_old) == 37
    obs = list(obs_old).copy()

    root_rot_mat = np.array(obs_old[1:10]).reshape(3, 3)
    root_orn = R.from_matrix(root_rot_mat).as_quat()
    root_rpy = pybullet.getEulerFromQuaternion(root_orn)
    root_rpy_m = [root_rpy[0], -root_rpy[1], 3.14159-root_rpy[2]]
    root_orn_m = pybullet.getQuaternionFromEuler(root_rpy_m)
    obs[1:10] = list(pybullet.getMatrixFromQuaternion(root_orn_m))

    # root lin vel
    obs[10:13] = [obs_old[10], -obs_old[11], obs_old[12]]

    # non-root joint q (12)
    obs[13:19] = mirror_leg_q(obs_old[13:19])
    obs[19:25] = mirror_leg_q(obs_old[19:25])

    # feet (offset from root) (12)
    obs[25:31] = mirror_foot_pos(obs_old[25:31])
    obs[31:37] = mirror_foot_pos(obs_old[31:37])

    return obs


def mirror_action(act_old):

    act = list(act_old).copy()
    assert len(act) == 12

    act[:6] = mirror_leg_q(act_old[:6])
    act[6:12] = mirror_leg_q(act_old[6:12])
    return act
