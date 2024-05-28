#!/bin/bash

# CIFAR-100, IPC=25
python eval.py --device cuda:0 \
--dataset CIFAR100 --model ResNet18BN \
--method condensed --aug combined \
--ipc 25 --alpha 0.8 \
--epoch_eval_train 1000

# CIFAR-100, IPC=50
python eval.py --device cuda:0 \
--dataset CIFAR100 --model ResNet18BN \
--method condensed --aug combined \
--ipc 50 --alpha 0.6 \
--epoch_eval_train 500

# CIFAR-100, IPC=100
python eval.py --device cuda:0 \
--dataset CIFAR100 --model ResNet18BN \
--method condensed --aug combined \
--ipc 100 --alpha 0.3 \
--epoch_eval_train 250

# CIFAR-100, IPC=150
python eval.py --device cuda:0 \
--dataset CIFAR100 --model ResNet18BN \
--method condensed --aug combined \
--ipc 150 --alpha 0.2 \
--epoch_eval_train 167

