#!/bin/bash

# Tiny, IPC=50
CUDA_VISIBLE_DEVICES=0,1,2 python distill.py --load_all --device cuda \
--dataset Tiny --model ConvNetD4BN --model_eval ResNet18BN_Tiny --syn_steps 20 \
--initialize window --score forgetting \
--ipc 50 --alpha 0.6 --beta 0.8 \
--epoch_eval_train 500

# Tiny, IPC=100
CUDA_VISIBLE_DEVICES=0,1,2 python distill.py --load_all --device cuda \
--dataset Tiny --model ConvNetD4BN --model_eval ResNet18BN_Tiny --syn_steps 20 \
--initialize window --score forgetting \
--ipc 100 --alpha 0.5 --beta 0.7 \
--epoch_eval_train 250