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

import gym
from gym.envs.registration import registry, make, spec

from .laikago_env_v4 import LaikagoBulletEnvV4
from .hopper_env_v3 import HopperURDFEnvV3

from .hopper_env_combined_policy import HopperCombinedEnv
from .laikago_env_combined_policy import LaikagoCombinedEnv


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


# ------------bullet-------------

register(
    id="HopperURDFEnv-v3",
    entry_point="my_pybullet_envs:HopperURDFEnvV3",
    max_episode_steps=500,
)

register(
    id="HopperCombinedEnv-v1",
    entry_point="my_pybullet_envs:HopperCombinedEnv",
    max_episode_steps=500,
)

register(
    id="LaikagoBulletEnv-v4",
    entry_point="my_pybullet_envs:LaikagoBulletEnvV4",
    max_episode_steps=500,
)

register(
    id="LaikagoCombinedEnv-v1",
    entry_point="my_pybullet_envs:LaikagoCombinedEnv",
    max_episode_steps=500,
)

def getList():
    btenvs = [
        "- " + spec.id
        for spec in gym.envs.registry.all()
        if spec.id.find("Bullet") >= 0
    ]
    return btenvs
