import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import torchvision.transforms as transforms

from tqdm import tqdm
from utils import get_dataset, get_network, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import copy
import random
from reparam_module import ReparamModule
import time

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(args):
    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
    args.dsa_param = ParamDiffAug()

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)

    args.num_classes = num_classes
    args.im_size = im_size

    print('Hyper-parameters: \n', args.__dict__)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to(args.device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

    if args.method == 'random':
        selection = []
        for c in range(num_classes):
            selection += random.sample(indices_class[c], args.ipc)
        images_eval = images_all[selection]
        labels_eval = labels_all[selection]

    elif args.method == 'window':
        ths = int(args.ipc * args.beta)
        selection = []
        if args.score == 'cscore':
            score = np.load(f'scores/cscores_{args.dataset}.npz')['scores']
            reverse = False
        elif args.score == 'forgetting':
            score = np.load(f'scores/forgetting_{args.dataset}.npy')
            reverse = True
        else:
            print("Wrong score: ", args.score)
            exit()

        for c in range(num_classes):
            selection += sorted(indices_class[c], key=lambda i: score[i], reverse=reverse)[ths:ths + args.ipc]
        images_eval = images_all[selection]
        labels_eval = labels_all[selection]

    elif args.method == 'condensed':
        args.cpc = int(args.ipc * args.alpha) # number of condensed images per class
        args.spc = args.ipc - args.cpc # number of selected images per class
        condensed_path = f'condensed/{args.dataset}/IPC{args.ipc}'
        images_eval = torch.load(f'{condensed_path}/images_best.pt')
        labels_eval = torch.load(f'{condensed_path}/labels_best.pt')

    else:
        print("Wrong method: ", args.method)
        exit()

    accs_test = []
    accs_train = []
    for it in range(args.Iteration):
        net_eval = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)
        _, acc_train, acc_test = evaluate_synset(0, net_eval, images_eval, labels_eval, testloader, args, aug=args.aug)
        accs_test.append(acc_test)
        accs_train.append(acc_train)

    accs_test = np.array(accs_test)
    accs_train = np.array(accs_train)
    acc_test_mean = np.mean(accs_test)
    acc_test_std = np.std(accs_test)
    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), args.model, acc_test_mean, acc_test_std))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')
    parser.add_argument('--model', type=str, default='ResNet18BN', help='model')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')

    parser.add_argument('--epoch_eval_train', type=int, default=500, help='epochs to train a model with synthetic data')
    parser.add_argument('--lr_net', type=float, default=0.1, help='learning rate for evaulation')
    parser.add_argument('--batch_train', type=int, default=128, help='batch size for training networks')
    parser.add_argument('--Iteration', type=int, default=5, help='number of experiments')

    parser.add_argument('--method', type=str, default='condensed', help='Dataset to evaluate (random, window, condensed)')
    parser.add_argument('--score', type=str, default='cscore', help='Sample difficulty metric (cscore, forgetting)')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--alpha', type=float, default=0.5, help='Distillation portion')
    parser.add_argument('--beta', type=float, default=0.5, help='Difficulty level')
    parser.add_argument('--aug', type=str, default='combined', help='augmentation method (dsa, simple, combined)')

    args = parser.parse_args()

    main(args)
