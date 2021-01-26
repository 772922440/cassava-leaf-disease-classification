#!/bin/bash
config=$1  # ensemble_train 配置文件
args=$2 # 其它args，逗号分割
gpus=$3    # 0,1
threads=$4 # 2 并行进程数量
k_flods=$5   # 5

gpus=(${gpus//,/ })
args=(${args//,/ })

if [ ! $gpus ]; then
  gpus=(0)
fi
len_gpu=${#gpus[@]}

if [ ! $threads ]; then
  threads=$len_gpu
fi

if [ ! $k_flods ]; then
  k_flods=5
fi

echo "CONFIG:" $config
echo "GPU LIST:" "${gpus[@]}"
echo "THREADS:" $threads
echo "K_FLODS:" $k_flods

# run parallel
for((i=0;i<k_flods;i++)); do
    gpu=${gpus[$((i % len_gpu))]}  
    CUDA_VISIBLE_DEVICES="$gpu" python3 ensemble_train.py name="$config" k="$i" "${args[@]}" &

    if [ $(((i+1) % threads)) -eq 0 ]; then
        wait
    fi
    sleep 3
done
wait