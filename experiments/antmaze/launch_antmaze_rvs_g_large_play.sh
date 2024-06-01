#!/usr/bin/env bash

config="experiments/config/d4rl/antmaze_rvs_g.cfg"
#declare -a envs=("antmaze-umaze-v0" "antmaze-umaze-diverse-v0" "antmaze-medium-diverse-v0" "antmaze-medium-play-v0" "antmaze-large-diverse-v0" "antmaze-large-play-v0")
declare -a envs=("antmaze-large-play-v2")
seeds=25
use_gpu=false

for env in "${envs[@]}"; do
  for seed in $(seq 25 $((seeds))); do
    if [ "$use_gpu" = true ]; then
      python src/wt/train.py --configs "$config" --env_name "$env" --seed "$seed" --use_gpu
    else
      python src/wt/train.py --configs "$config" --env_name "$env" --seed "$seed"
    fi
  done
done
