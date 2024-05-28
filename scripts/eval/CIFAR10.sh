#!/bin/bash

# CIFAR-10, IPC=250
python eval.py --device cuda:0 \
--dataset CIFAR10 --model ResNet18BN \
--method condensed --aug combined \
--ipc 250 --alpha 0.6 \
--epoch_eval_train 1000

# CIFAR-10, IPC=500
python eval.py --device cuda:0 \
--dataset CIFAR10 --model ResNet18BN \
--method condensed --aug combined \
--ipc 500 --alpha 0.2 \
--epoch_eval_train 500

# CIFAR-10, IPC=1000
python eval.py --device cuda:0 \
--dataset CIFAR10 --model ResNet18BN \
--method condensed --aug combined \
--ipc 1000 --alpha 0.1 \
--epoch_eval_train 250

# CIFAR-10, IPC=1500
python eval.py --device cuda:0 \
--dataset CIFAR10 --model ResNet18BN \
--method condensed --aug combined \
--ipc 1500 --alpha 0.1 \
--epoch_eval_train 167

