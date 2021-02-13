#!/bin/bash

trap 'onCtrlC' INT

function onCtrlC () {
  echo 'Ctrl+C is captured'
  for pid in $(jobs -p); do
    kill -9 $pid
  done

  kill -HUP $( ps -A -ostat,ppid | grep -e '^[Zz]' | awk '{print $2}')
  exit 1
}

seed=$1
args=$2 # 其它args，逗号分割
gpus=$3    # 0,1
threads=$4 # 2 并行进程数量
k_flods=$5   # 5

if [ ! $seed ]; then
  echo "please input seed..." 
  exit 1
fi

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

echo "SEED:" $seed
echo "GPU LIST:" "${gpus[@]}"
echo "ARG LIST:" "${args[@]}"
echo "THREADS:" $threads
echo "K_FLODS:" $k_flods

# split csv
python3 ensemble_split.py name=mobilenetv3_large_100 seed="$seed" k_folds_csv=kfold5_"$seed".csv

# run parallel
for((i=0;i<k_flods;i++)); do
    gpu=${gpus[$((i % len_gpu))]}  
    CUDA_VISIBLE_DEVICES="$gpu" python3 ensemble_train.py name=mobilenetv3_large_100 k="$i"\
         seed="$seed" k_folds_csv=kfold5_"$seed".csv model_suffix="$seed" "${args[@]}" &

    if [ $(((i+1) % threads)) -eq 0 ]; then
        wait
    fi
    sleep 1
done
wait