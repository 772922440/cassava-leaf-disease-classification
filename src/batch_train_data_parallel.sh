#!/bin/bash
trap 'onCtrlC' INT

function onCtrlC () {
  echo 'Ctrl+C is captured'
  for pid in $(jobs -p); do
    kill -9 $pid
  done
  exit 1
}

config=$1  # ensemble_train 配置文件
args=$2 # 其它args，逗号分割
gpus=$3    # 0,1
k_flods=$4   # 5

args=(${args//,/ })

if [ ! $k_flods ]; then
  k_flods=5
fi

echo "CONFIG:" $config
echo "K_FLODS:" $k_flods
echo "GPUS:" $gpus

# run parallel
for((i=0;i<k_flods;i++)); do
    CUDA_VISIBLE_DEVICES="$gpus" python3 ensemble_train.py name="$config" k="$i" "${args[@]}" &
    wait
    sleep 1
done
wait