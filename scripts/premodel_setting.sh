#!/bin/bash
mode=$1
algos=$2
epoch=5
mem_size=1000

cd ..

common_cmd="general_main.py --data cifar100 --cl_type nc --store"

declare -A model_cmd
model_cmd=( ["reduced_rn18"]="--model reduced_rn18 --learning_rate 0.1"
            ["pretrained_rn18"]="--model pretrained_rn18 --learning_rate 0.01"
            ["pretrained_rn50"]="--model pretrained_rn50 --learning_rate 0.01"
            ["clip"]="--model clip --learning_rate 1e-6 --fix_bn"
            ["clip_highlr"]="--model clip --learning_rate 0.01"
            ["clip_fix_bn"]="--model clip --learning_rate 0.01 --fix_bn" )

declare -A algo_cmd
algo_cmd=( ["er"]="--agent ER --retrieve random --update random --mem_size $mem_size"
           ["sgd"]="--agent ER --retrieve random --update random --mem_size 0"
           ["lwf"]="--agent LWF"
           ["gdumb"]="--agent GDUMB --mem_size $mem_size --mem_epoch 30 --minlr 0.0005 --clip 10" )

declare -A mode_cmd
mode_cmd=( ["offline"]="--epoch $epoch --online False"
           ["cil"]="--epoch $epoch --online True"
           ["online"]="--epoch 1 --online True" )

# Default to run all algorithms if algos not specified
if [ "$mode" = "offline" ]; then
    algos="sgd"
fi

if [ -z "$algos" ]; then
    algos="er sgd lwf gdumb"
fi

if [ "$mode" = "cil" ]; then
    algos=${algos//"gdumb"/}
fi

for model in reduced_rn18 pretrained_rn18 pretrained_rn50 clip clip_highlr clip_fix_bn; do
    for algo in $algos; do
        if [ "$algo" = 'sgd' ] || [ "$algo" = 'lwf' ]; then
            expname_cmd="--exp_name ""$algo"_"$model"_"$mode"
        else
            expname_cmd="--exp_name ""$algo"_"$model"_"$mem_size"_"$mode"
        fi

        set -x
        python $common_cmd ${model_cmd[$model]} ${algo_cmd[$algo]} ${mode_cmd[$mode]} $expname_cmd
        set +x
    done
done
