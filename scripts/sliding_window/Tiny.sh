#!/bin/bash

# Tiny, IPC=50
for beta in $(seq 0 0.1 0.9);
do
    echo "beta: $beta"
    python eval.py --device cuda:0 \
    --dataset Tiny --model ResNet18BN_Tiny \
    --method window --score forgetting --aug dsa \
    --ipc 50 --beta $beta \
    --epoch_eval_train 500
done

# Tiny, IPC=100
for beta in $(seq 0 0.1 0.8);
do
    echo "beta: $beta"
    python eval.py --device cuda:0 \
    --dataset Tiny --model ResNet18BN_Tiny \
    --method window --score forgetting --aug dsa \
    --ipc 100 --beta $beta \
    --epoch_eval_train 250
done
