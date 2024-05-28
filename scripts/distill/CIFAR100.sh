#!/bin/bash

# CIFAR-100, IPC=25
python distill.py --load_all --device cuda:0 \
--dataset CIFAR100 --initialize window --score cscore \
--ipc 25 --alpha 0.8 --beta 0.8 \
--epoch_eval_train 1000

# CIFAR-100, IPC=50
python distill.py --load_all --device cuda:0 \
--dataset CIFAR100 --initialize window --score cscore \
--ipc 50 --alpha 0.6 --beta 0.7 \
--epoch_eval_train 500

# CIFAR-100, IPC=100
python distill.py --load_all --device cuda:0 \
--dataset CIFAR100 --initialize window --score cscore \
--ipc 100 --alpha 0.3 --beta 0.6 \
--epoch_eval_train 250

# CIFAR-100, IPC=150
python distill.py --load_all --device cuda:0 \
--dataset CIFAR100 --initialize window --score cscore \
--ipc 150 --alpha 0.2 --beta 0.4 \
--epoch_eval_train 167