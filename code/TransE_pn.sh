#!/bin/bash

for bs in 64 128 256
do  
    for lr in 1e-2 5e-3 1e-3
    do
        for emb_dim in 16 32 64
        do
            for lmbda in 0.5 1 2
            do
                python main.py --bs $bs --lr $lr --emb_dim $emb_dim --lmbda $lmbda --gpu 1 --loss_type pn --base_model TransE
            done
        done 
    done
done