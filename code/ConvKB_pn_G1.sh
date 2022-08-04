#!/bin/bash

for bs in 64 128 256
do  
    for lr in 1e-2 5e-3 1e-3
    do
        for emb_dim in 50 100
        do
            for lmbda in 0.5 1 2
            do
                python main.py --bs $bs --lr $lr --emb_dim $emb_dim --lmbda $lmbda --kernel_size 1 --out_channels 64 --gpu 1 --loss_type pn --base_model ConvKB --graph 1
            done
        done 
    done
done

