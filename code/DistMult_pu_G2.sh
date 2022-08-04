#!/bin/bash

for bs in 64 128 256
do  
    for lr in 1e-2 5e-3 1e-3
    do
        for emb_dim in 16 32 64
        do
            for lmbda in 0.5 1 2
            do
                python main.py --bs $bs --lr $lr --emb_dim $emb_dim --lmbda $lmbda --prior 1e-1 --gpu 0 --loss_type pu --base_model DistMult --graph 2
                python main.py --bs $bs --lr $lr --emb_dim $emb_dim --lmbda $lmbda --prior 1e-2 --gpu 0 --loss_type pu --base_model DistMult --graph 2
                python main.py --bs $bs --lr $lr --emb_dim $emb_dim --lmbda $lmbda --prior 1e-3 --gpu 0 --loss_type pu --base_model DistMult --graph 2
                python main.py --bs $bs --lr $lr --emb_dim $emb_dim --lmbda $lmbda --prior 1e-4 --gpu 0 --loss_type pu --base_model DistMult --graph 2
                python main.py --bs $bs --lr $lr --emb_dim $emb_dim --lmbda $lmbda --prior 1e-5 --gpu 0 --loss_type pu --base_model DistMult --graph 2
            done
        done 
    done
done


