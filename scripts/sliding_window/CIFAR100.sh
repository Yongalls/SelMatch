#!/bin/bash

# CIFAR-100, IPC=25
for beta in $(seq 0 0.1 0.9);
do
    echo "beta: $beta"
    python eval.py --device cuda:0 \
    --dataset CIFAR100 --model ResNet18BN \
    --method window --score cscore --aug dsa \
    --ipc 25 --beta $beta \
    --epoch_eval_train 1000
done

# CIFAR-100, IPC=50
for beta in $(seq 0 0.1 0.9);
do
    echo "beta: $beta"
    python eval.py --device cuda:0 \
    --dataset CIFAR100 --model ResNet18BN \
    --method window --score cscore --aug dsa \
    --ipc 50 --beta $beta \
    --epoch_eval_train 500
done

# CIFAR-100, IPC=100
for beta in $(seq 0 0.1 0.8);
do
    echo "beta: $beta"
    python eval.py --device cuda:0 \
    --dataset CIFAR100 --model ResNet18BN \
    --method window --score cscore --aug dsa \
    --ipc 100 --beta $beta \
    --epoch_eval_train 250
done

# CIFAR-100, IPC=150
for beta in $(seq 0 0.1 0.7);
do
    echo "beta: $beta"
    python eval.py --device cuda:0 \
    --dataset CIFAR100 --model ResNet18BN \
    --method window --score cscore --aug dsa \
    --ipc 150 --beta $beta \
    --epoch_eval_train 167
done