#!/bin/bash

# Tiny, IPC=50
python eval.py --device cuda:0 \
--dataset Tiny --model ResNet18BN_Tiny \
--method condensed --aug combined \
--ipc 50 --alpha 0.6 \
--epoch_eval_train 500

# Tiny, IPC=100
python eval.py --device cuda:0 \
--dataset Tiny --model ResNet18BN_Tiny \
--method condensed --aug combined \
--ipc 100 --alpha 0.5 \
--epoch_eval_train 250
