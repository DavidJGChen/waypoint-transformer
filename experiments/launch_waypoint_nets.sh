#!/usr/bin/env bash

config="experiments/config/d4rl/goal_network.cfg"
declare -a envs=("antmaze-large-diverse-v2")
#declare -a envs=("antmaze-umaze-v0" "antmaze-umaze-diverse-v0" "antmaze-medium-diverse-v0" "antmaze-medium-play-v0" "antmaze-large-diverse-v0" "antmaze-large-play-v0")
# declare -a envs=("kitchen-complete-v0" "kitchen-mixed-v0" "kitchen-partial-v0")
# declare -a envs=("hopper-medium-replay-v2" "halfcheetah-medium-replay-v2" "walker2d-medium-replay-v2" "hopper-medium-v2" "halfcheetah-medium-v2" "walker2d-medium-v2")
seeds=13
use_gpu=true
for env in "${envs[@]}"; do
	for seed in $(seq 11 $((seeds))); do
	      PYTORCH_ENABLE_MPS_FALLBACK=1 D4RL_SUPPRESS_IMPORT_ERROR=1 PYTHONPATH="src" CC=/opt/homebrew/bin/gcc-11 python src/wt/train.py --configs "$config" --env_name "$env" --seed "$seed" --train_goal_net &
	done
done
wait
