#!/usr/bin/env bash
id="0"
seed="1100000"

python -m third_party.a2c_ppo_acktr.main_gail_dyn_ppo --env-name "LaikagoCombinedEnv-v1" --num-steps 1000 --num-processes 8 --lr 3e-4 --entropy-coef 0 --ppo-epoch 10 --num-mini-batch 16 --num-env-steps 8000000 --gail-traj-path "./laika_70_heavy_n200_0.pkl" --save-dir "trained_models_Gdyn_laika_bullet_heavy70_comb_f${id}" --seed ${seed} --gail-traj-num 200 --train_dyn 1 --gail-epoch 5 --act_noise 1 --obs_noise 1 --behavior-dir "trained_models_laika_bullet_70/ppo" --behavior_env_name "LaikagoBulletEnv-v4" --hidden-size 100 --cuda_env 0 --gail_downsample_frequency 1 --gamma 0.99 --gail-dis-hdim 100 --behavior-logstd -1.3 --use-split-pi --num-feet 4

python -m third_party.a2c_ppo_acktr.main --env-name "LaikagoCombinedEnv-v1" --num-steps 1000 --num-processes 8 --lr 1.5e-4 --entropy-coef 0 --ppo-epoch 10 --num-mini-batch 8 --num-env-steps 4000000 --use-linear-lr-decay --clip-param 0.1 --train_dyn 0  --dyn_dir "trained_models_Gdyn_laika_bullet_heavy70_comb_f${id}/ppo" --save-dir "trained_models_laika_bullet_FTGAIL_heavy70_comb_f${id}" --seed ${seed} --warm-start "./trained_models_laika_bullet_70/ppo/LaikagoBulletEnv-v4.pt" --act_noise 1 --obs_noise 1 --warm-start-logstd -1.3 --cuda-env 0