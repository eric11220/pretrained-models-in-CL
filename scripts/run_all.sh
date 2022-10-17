#!/bin/bash
arch=$1
mem_size=$2
dataset=$3
ntask=$4
offline_ep=$5

if [ -z "$arch" ] | [ -z "$mem_size" ] | [ -z "$dataset" ] | [ -z "$ntask" ]; then
    echo "Usage: bash run_all.sh arch mem_size dataset ntask #2nd_stage_epochs"
    exit
fi

exp_postfix=""
addition_cmd=""
if [ ! -z "$offline_ep" ]; then
    addition_cmd="--offline_epoch $offline_ep"
    exp_postfix="_offline""$offline_ep"
fi

# Pre-trained model should pair with a lower LR
if [[ "$arch" == *"pretrained"* ]] || [[ "$arch" == "swav" ]] || [[ "$arch" == "simclr" ]] || [[ "$arch" == "barlow_twins" ]]; then
    lr=0.01
elif [[ "$arch" == "clip" ]]; then
    lr=1e-6
    addition_cmd="$addition_cmd --fix_bn"
else
    lr=0.1
fi
common_cmd="--model $arch --store --learning_rate $lr"

cd ..
set -x

#ER
python general_main.py --data $dataset --cl_type nc --agent ER --retrieve random --update random --mem_size $mem_size --exp_name er_"$arch"_"$mem_size"$exp_postfix $common_cmd --num_tasks $ntask $addition_cmd

#MIR
python general_main.py --data $dataset --cl_type nc --agent ER --retrieve MIR --update random --mem_size $mem_size --exp_name mir_"$arch"_"$mem_size" $common_cmd --num_tasks $ntask $addition_cmd

#GSS
python general_main.py --data $dataset --cl_type nc --agent ER --retrieve random --update GSS --eps_mem_batch 10 --gss_mem_strength 20 --mem_size $mem_size --exp_name gss_"$arch"_"$mem_size" $common_cmd --num_tasks $ntask $addition_cmd

#LwF
#python general_main.py --data $dataset --cl_type nc --agent LWF --exp_name lwf_"$arch" $common_cmd --num_tasks $ntask $addition_cmd

iCaRL
python general_main.py --data $dataset --cl_type nc --agent ICARL --retrieve random --update random --mem_size $mem_size --exp_name icarl_"$arch"_"$mem_size"$exp_postfix $common_cmd --num_tasks $ntask $addition_cmd

#EWC++
python general_main.py --data $dataset --cl_type nc --agent EWC --fisher_update_after 50 --alpha 0.9 --lambda 100 --exp_name ewc++_"$arch" $common_cmd --num_tasks $ntask $addition_cmd

#GDumb
python general_main.py --data $dataset --cl_type nc --agent GDUMB --mem_size $mem_size --mem_epoch 30 --minlr 0.0005 --clip 10 --exp_name gdumb_"$arch"_"$mem_size" $common_cmd --num_tasks $ntask $addition_cmd

#AGEM
python general_main.py --data $dataset --cl_type nc --agent AGEM --retrieve random --update random --mem_size $mem_size --exp_name agem_"$arch"_"$mem_size" $common_cmd --num_tasks $ntask $addition_cmd

#SCR
python general_main.py --data $dataset --cl_type nc --agent SCR --retrieve random --update random --mem_size $mem_size --head mlp --temp 0.07 --eps_mem_batch 100 --exp_name scr_"$arch"_"$mem_size"$exp_postfix $common_cmd --num_tasks $ntask $addition_cmd
