#!/bin/bash

# CIFAR-10, IPC=250
python distill.py --load_all --device cuda:0 \
--dataset CIFAR10 --initialize window --score cscore \
--ipc 250 --alpha 0.6 --beta 0.5 \
--epoch_eval_train 1000

# CIFAR-10, IPC=500
python distill.py --load_all --device cuda:0 \
--dataset CIFAR10 --initialize window --score cscore \
--ipc 500 --alpha 0.2 --beta 0.3 \
--epoch_eval_train 500

# CIFAR-10, IPC=1000
python distill.py --load_all --device cuda:0 \
--dataset CIFAR10 --initialize window --score cscore \
--ipc 1000 --alpha 0.1 --beta 0.2 \
--epoch_eval_train 250 --lr_img 10000

# CIFAR-10, IPC=1500
python distill.py --load_all --device cuda:0 \
--dataset CIFAR10 --initialize window --score cscore \
--ipc 1500 --alpha 0.1 --beta 0.2 \
--epoch_eval_train 167 --lr_img 10000