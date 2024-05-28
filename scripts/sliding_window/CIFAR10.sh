#!/bin/bash

# CIFAR-10, IPC=250
for beta in $(seq 0 0.1 0.9);
do
    echo "beta: $beta"
    python eval.py --device cuda:0 \
    --dataset CIFAR10 --model ResNet18BN \
    --method window --score cscore --aug dsa \
    --ipc 250 --beta $beta \
    --epoch_eval_train 1000
done

# CIFAR-10, IPC=500
for beta in $(seq 0 0.1 0.9);
do
    echo "beta: $beta"
    python eval.py --device cuda:0 \
    --dataset CIFAR10 --model ResNet18BN \
    --method window --score cscore --aug dsa \
    --ipc 500 --beta $beta \
    --epoch_eval_train 500
done

# CIFAR-10, IPC=1000
for beta in $(seq 0 0.1 0.8);
do
    echo "beta: $beta"
    python eval.py --device cuda:0 \
    --dataset CIFAR10 --model ResNet18BN \
    --method window --score cscore --aug dsa \
    --ipc 1000 --beta $beta \
    --epoch_eval_train 250
done

# CIFAR-10, IPC=1500
for beta in $(seq 0 0.1 0.7);
do
    echo "beta: $beta"
    python eval.py --device cuda:0 \
    --dataset CIFAR10 --model ResNet18BN \
    --method window --score cscore --aug dsa \
    --ipc 1500 --beta $beta \
    --epoch_eval_train 167
done