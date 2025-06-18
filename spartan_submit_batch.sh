#!/bin/bash

conda_env="ecnn_torch"
script_name="cifar10_ecnn_qary_v5.py"

runtime="5:00:00"
mem="32G"
cpus_per_task="16"
a100_gpus="1"
deeplearn_gpus="1"

num_models="30"
batch_size="128"
epochs="200"
log_det_lamda="0.1"
cm_path="./all_matrix/30/2/1.txt"
seed="42"

partition_list=("a100" "feita100" "deeplearn") # "a100" "feita100" "deeplearn" # a100 feita100 deeplearn

# Run the script with arguments
for partition in "${partition_list[@]}"; do
    exps_root="exps/${partition}"
    
    # Generate a random experiment name
    id=$((RANDOM % 9000000 + 1000000))
    exp_name="${exps_root}/num_models_${num_models}_log_det_lamda_${log_det_lamda}_seed_${seed}_id_${id}"

    # Construct arguments string
    arguments="--exp_name $exp_name --num_models $num_models --batch_size $batch_size --epochs $epochs --log_det_lamda $log_det_lamda --cm_path $cm_path --seed $seed"
    echo "Running experiment: $exp_name"
    bash spartan_submit.sh --partition $partition --conda-env $conda_env --script-name $script_name --arguments "$arguments" --runtime $runtime --mem $mem --cpus-per-task $cpus_per_task --a100-gpus $a100_gpus --deeplearn-gpus $deeplearn_gpus --id $id
done