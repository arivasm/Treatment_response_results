#!/bin/bash

for bs in 32 64 128 256 512
do  
    for lr in 1e-1 1e-2 1e-3 1e-4 1e-5
    do
        for emb_dim in 16 32 64 128 256
        do
            for num_ng in 2 4 8 16 32
            do 
                for lmbda in 0.25 0.5 1 2 4
                do 
                    for prior in 1e-1 1e-2 1e-3 1e-4 1e-5
                    do
                        python main.py --bs $bs --lr $lr --emb_dim $emb_dim --num_ng $num_ng --lmbda $lmbda --gpu 1 --loss_type pu
                    done
                done
            done
        done 
    done
done

