# Pytorch Implementation of SimGAN

SimGAN: Hybrid Simulator Identification for Domain Adaptation via Adversarial Reinforcement Learning

Arxiv: https://arxiv.org/abs/2101.06005

Copyright 2020 Google LLC and Stanford University,

Licensed under the Apache License, Version 2.0

### This is not an officially supported Google product

## Installation:

We recommend Ubuntu or MacOS as the operating systems. However, the following installation instructions "should" work for Windows as well.

1.Go to (https://www.anaconda.com/download/) and install the Python 3 version of Anaconda.

2.Open a new terminal and run the following commands to create a new conda environment (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

```conda create -n sim-gan python=3.6```

3.Activate & enter the new environment you just creared:

```conda activate sim-gan```

4.Inside the new environment, and inside the project directory:

```pip install -r requirements.txt```

5.Install pytorch (should work with or without cuda):

```conda install pytorch==1.5.0 -c pytorch``` 	(check pytorch website for your favorite version)

## Training and Testing:

The following six scripts will reproduce the six (2*3) experiments in the paper:

```
sh train_laika_deform.sh 
sh train_laika_heavy.sh
sh train_laika_power.sh
sh train_hopper_deform.sh 
sh train_hopper_heavy.sh
sh train_hopper_power.sh  
```
(The first command in each script runs Hybrid Simulator Identification as listed in Algorithm 1 of the paper, 
while the second command runs the Policy Refinement part as listed in Algorithm 1)

And the corresponding six test commands, to visualize performance in the six target envs (as proxies of the "real environments" in sim-to-real):

```
python -m third_party.a2c_ppo_acktr.collect_tarsim_traj --env-name "LaikagoBulletEnv-v4"  --load-dir trained_models_laika_bullet_FTGAIL_deform70_comb_f0/ppo --render 1  --non-det 0 --seed 0 --act_noise 1 --obs_noise 1  --deform-floor-env 1  --num-traj 10 --src-env-name "LaikagoCombinedEnv-v1" --final-test 1

python -m third_party.a2c_ppo_acktr.collect_tarsim_traj --env-name "LaikagoBulletEnv-v4"  --load-dir trained_models_laika_bullet_FTGAIL_heavy70_comb_f0/ppo --render 1  --non-det 0 --seed 0 --act_noise 1 --obs_noise 1  --heavy-leg-env 1  --num-traj 10 --src-env-name "LaikagoCombinedEnv-v1" --final-test 1

python -m third_party.a2c_ppo_acktr.collect_tarsim_traj --env-name "LaikagoBulletEnv-v4"  --load-dir trained_models_laika_bullet_FTGAIL_emf70_comb_f0/ppo --render 1  --non-det 0 --seed 0 --act_noise 1 --obs_noise 1  --emf-power-env 1  --num-traj 10 --src-env-name "LaikagoCombinedEnv-v1" --final-test 1

python -m third_party.a2c_ppo_acktr.collect_tarsim_traj --env-name "HopperURDFEnv-v3" --load-dir trained_models_hopper_bullet_FTGAIL_deform_new11_comb_f0/ppo --render 1 --non-det 0 --seed 0 --act_noise 1 --obs_noise 1 --deform-floor-env 1 --num-trajs 10 --src-env-name "HopperCombinedEnv-v1"

python -m third_party.a2c_ppo_acktr.collect_tarsim_traj --env-name "HopperURDFEnv-v3" --load-dir trained_models_hopper_bullet_FTGAIL_heavy_new11_comb_f0/ppo --render 1 --non-det 0 --seed 0 --act_noise 1 --obs_noise 1 --heavy-head-env 1 --num-trajs 10 --src-env-name "HopperCombinedEnv-v1"

python -m third_party.a2c_ppo_acktr.collect_tarsim_traj --env-name "HopperURDFEnv-v3" --load-dir trained_models_hopper_bullet_FTGAIL_low_new11_comb_f0/ppo --render 1 --non-det 0 --seed 0 --act_noise 1 --obs_noise 1 --low-power-env 1 --num-trajs 10 --src-env-name "HopperCombinedEnv-v1"
```

## Notes:

This implementation is based on the PPO/GAIL code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail,
and OpenAI Baselines: https://github.com/openai/baselines,

### Overview of important files:

my_pybullet_envs/hopper_env_v3.py: Source Env ("Sim") to train the behavior policy and Target Envs ("Real") to collect behavior data and visualize adapted policies using SimGAN, for the OpenAI Gym Hopper

my_pybullet_envs/laikago_env_v4.py: Same as above, but for the Laikago Robot

my_pybullet_envs/hopper_env_combined_policy.py: Training Env of the Hybrid Simulator, used both for learning the simulator (where the policy is fixed) and policy refinement (where the learned simulator is fixed), for the OpenAI Gym Hopper

my_pybullet_envs/laikago_env_combined_policy.py: Same as above, but for the Laikago Robot

third_party/a2c_ppo_acktr/algo/gail.py: GAN+DRL used to train the hybrid simulator. This essentially becomes the GAIL algorithm (for training a dynamics function rather than a controller). Adapted from ikostrikov's GAIL implementation.

third_party/a2c_ppo_acktr/collect_tarsim_traj.py: Main program to collect behavior policy data, and to visualize adapted policies

third_party/a2c_ppo_acktr/main.py: Main program to train the behavior policy and to do policy refinement stage training

third_party/a2c_ppo_acktr/main_gail_dyn_ppo.py: Main program to run GAN+DRL to train the hybrid simulator

### TODOs: 
behavior training and collect data

link to video