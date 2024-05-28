#!/bin/bash

python buffer.py --device cuda:0 \
--dataset Tiny --model ConvNetD4BN \
--train_epochs=100 --num_experts=100