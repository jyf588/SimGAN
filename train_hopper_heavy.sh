#!/usr/bin/env bash
id="0"
seed="2100000"

python -m third_party.a2c_ppo_acktr.main_gail_dyn_ppo --env-name "HopperCombinedEnv-v1" --num-steps 1000 --num-processes 8 --lr 3e-4 --entropy-coef 0 --num-mini-batch 16 --num-env-steps 2000000 --gail-traj-path "./hopper_new11_heavy_n200_3.pkl" --save-dir "trained_models_Gdyn_hopper_bullet_heavy_new11_comb_f${id}" --seed ${seed} --gail-traj-num 200 --train_dyn 1 --gail-epoch 5 --act_noise 1 --obs_noise 1 --behavior-dir "trained_models_hopper_bullet_new11/ppo" --behavior_env_name "HopperURDFEnv-v3" --hidden-size 100 --cuda_env 0 --gail_downsample_frequency 1 --gail-dis-hdim 100 --behavior-logstd -1.3 --use-split-pi


python -m third_party.a2c_ppo_acktr.main --env-name "HopperCombinedEnv-v1" --num-steps 1000 --num-processes 8 --lr 1.5e-4 --entropy-coef 0 --ppo-epoch 2 --num-mini-batch 8 --num-env-steps 2000000 --use-linear-lr-decay --clip-param 0.1 --train_dyn 0  --dyn_dir "trained_models_Gdyn_hopper_bullet_heavy_new11_comb_f${id}/ppo" --save-dir "trained_models_hopper_bullet_FTGAIL_heavy_new11_comb_f${id}" --seed ${seed} --warm-start "./trained_models_hopper_bullet_new11/ppo/HopperURDFEnv-v3.pt" --act_noise 1 --obs_noise 1 --warm-start-logstd -1.3 --cuda-env 0