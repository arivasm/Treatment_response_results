#!/bin/bash

for bs in 64 128 256
do
    for lr in 1e-2 5e-3 1e-3
    do
        for lmbda in 0.5 1 2
        do
          python main.py --bs $bs --lr $lr --lmbda $lmbda --kernel_size 3 --out_channels 32 --emb_dim 200 --gpu 1 --loss_type pn --base_model ConvE
        done
    done
done

