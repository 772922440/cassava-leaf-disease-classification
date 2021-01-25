#!/bin/bash
config=$1  # ensemble_train
gpus=$2    # 0,1
k_flods=$3   # 5

gpus=(${gpus//,/ })

if [ ! $gpus ]; then
  gpus=(0)
fi

if [ ! $k_flods ]; then
  k_flods=5
fi

echo "CONFIG:" $config
echo "GPU LIST:" "${gpus[@]}"
echo "K_FLODS:" $k_flods


# run parallel
len_gpu=${#gpus[@]}
for((i=0;i<k_flods;i++)); do
    gpu=${gpus[$((i % len_gpu))]}  
    CUDA_VISIBLE_DEVICES="$gpu" python3 ensemble_train.py name="$config" k="$i" &

    if [ $(((i+1) % len_gpu)) -eq 0 ]; then
        wait
    fi
    sleep 3
done
wait