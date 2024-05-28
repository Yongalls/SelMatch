#!/bin/bash

python buffer.py --device cuda:0 \
--dataset CIFAR100 --model ConvNetBN \
--train_epochs=100 --num_experts=100