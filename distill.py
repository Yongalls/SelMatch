import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import copy
import random
from reparam_module import ReparamModule

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(args):
    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa_strategy = 'color_crop_cutout_flip_scale_rotate'
    args.dsa_param = ParamDiffAug()

    eval_it_pool = [0, 500, 1000, 2500, 5000, 7500, 10000]

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)

    args.num_classes = num_classes
    args.im_size = im_size

    if args.batch_syn == -1:
        args.batch_syn = num_classes * args.ipc

    if args.device == 'cuda':
        args.distributed = True
    else:
        args.distributed = False

    save_dir = os.path.join(args.save_path, args.dataset, f"IPC{args.ipc}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

    ''' initialize the synthetic data '''
    args.cpc = int(args.ipc * args.alpha) # number of condensed images per class
    args.spc = args.ipc - args.cpc # number of selected images per class
    args.ths = int(args.ipc * args.beta) # window starting point for each class (difficulty level)
    print(args.cpc, args.spc, args.ths)

    image_syn_c = torch.randn(size=(num_classes * args.cpc, channel, im_size[0], im_size[1]), dtype=torch.float)
    image_syn_s = torch.randn(size=(num_classes * args.spc, channel, im_size[0], im_size[1]), dtype=torch.float)

    label_syn_c = torch.tensor([np.ones(args.cpc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    label_syn_s = torch.tensor([np.ones(args.spc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    if args.initialize == 'window':
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
            idx_s = sorted(indices_class[c], key=lambda i: score[i], reverse=reverse)[args.ths:args.ths + args.spc]
            idx_c = sorted(indices_class[c], key=lambda i: score[i], reverse=reverse)[args.ths + args.spc:args.ths + args.spc + args.cpc]

            image_syn_s.data[c * args.spc:(c + 1) * args.spc] = images_all[idx_s].detach().data
            image_syn_c.data[c * args.cpc:(c + 1) * args.cpc] = images_all[idx_c].detach().data

    elif args.initialize == 'random':
        for c in range(num_classes):
            tmp = np.random.permutation(indices_class[c])[:args.ipc]
            idx_s = tmp[:args.spc]
            idx_c = tmp[args.spc:]

            image_syn_s.data[c * args.spc:(c + 1) * args.spc] = images_all[idx_s].detach().data
            image_syn_c.data[c * args.cpc:(c + 1) * args.cpc] = images_all[idx_c].detach().data

    else:
        print("Wrong Initialization: ", args.initialize)
        exit()

    image_syn_c = image_syn_c.detach().to(args.device).requires_grad_(True)
    image_syn_s = image_syn_s.detach().to(args.device).requires_grad_(False)

    label_syn = torch.cat((label_syn_c, label_syn_s), dim=0)

    syn_lr = torch.tensor(args.lr_init).to(args.device)

    ''' training '''
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)

    optimizer_img = torch.optim.SGD([image_syn_c], lr=args.lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_img.zero_grad()

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    expert_dir = os.path.join(args.buffer_path, args.dataset, args.model)
    print("Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)

    best_acc = 0
    accs_save = []
    for it in range(0, args.Iteration+1):
        save_this_it = False

        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            image_syn = torch.cat((image_syn_c, image_syn_s), dim=0)
            print("syn_dataset: ", image_syn.shape)
            print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, args.model_eval, it))
            accs_test = []
            accs_train = []
            for it_eval in range(args.num_eval):
                net_eval = get_network(args.model_eval, channel, num_classes, im_size, dist=args.distributed).to(args.device) # get a random model

                image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification

                _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, aug=args.aug)
                accs_test.append(acc_test)
                accs_train.append(acc_train)

            accs_test = np.array(accs_test)
            accs_train = np.array(accs_train)
            acc_test_mean = np.mean(accs_test)
            acc_test_std = np.std(accs_test)
            accs_save.append((it, acc_test_mean, acc_test_std))

            print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), args.model_eval, acc_test_mean, acc_test_std))

            image_save, label_save = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
            torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
            torch.save(label_save.cpu(), os.path.join(save_dir, "labels_{}.pt".format(it)))

            if acc_test_mean > best_acc:
                best_acc = acc_test_mean
                torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt"))
                torch.save(label_save.cpu(), os.path.join(save_dir, "labels_best.pt"))


        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model
        student_net = ReparamModule(student_net)
        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                print("loading file {}".format(expert_files[file_idx]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)

        start_epoch = np.random.randint(0, args.max_start_epoch)
        starting_params = expert_trajectory[start_epoch]

        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        image_syn = torch.cat((image_syn_c, image_syn_s), dim=0)
        syn_images = image_syn

        y_hat = label_syn.to(args.device)

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []

        for step in range(args.syn_steps):

            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()

            x = syn_images[these_indices]
            this_y = y_hat[these_indices]

            x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
            x = student_net(x, flat_param=forward_params)
            ce_loss = criterion(x, this_y)

            grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]

            student_params.append(student_params[-1] - syn_lr * grad)

        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)

        param_loss /= num_params
        param_dist /= num_params
        param_loss /= param_dist
        grand_loss = param_loss

        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()

        grand_loss.backward()

        optimizer_img.step()
        optimizer_lr.step()

        norm_c = torch.linalg.norm(image_syn_c, dim=(1,2,3))
        norm_c = torch.mean(norm_c)

        norm_s = torch.linalg.norm(image_syn_s, dim=(1,2,3))
        norm_s = torch.mean(norm_s)

        for _ in student_params:
            del _

        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f, syn_lr = %.4f, norm_c = %.4f, norm_s = %.4f' % (get_time(), it, grand_loss.item(), syn_lr.item(), norm_c, norm_s))

    print('\n==================== Final Results ====================\n')
    for it, acc_mean, acc_std in accs_save:
        print(f"Iteration: {it}, test acc: {acc_mean} ({acc_std})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # General settings
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')
    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")
    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')
    parser.add_argument('--data_path', type=str, default='./data/', help='dataset (original) path')
    parser.add_argument('--save_path', type=str, default='./logged_files/', help='save path (condensed result)')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')

    # Distillation
    parser.add_argument('--model', type=str, default='ConvNetBN', help='model')
    parser.add_argument('--Iteration', type=int, default=10000, help='how many distillation steps to perform')
    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_init', type=float, default=0.001, help='initialization for synthetic learning rate')
    parser.add_argument('--batch_syn', type=int, default=125, help='should only use this if you run out of VRAM')
    parser.add_argument('--expert_epochs', type=int, default=2, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=55, help='how many steps to take on synthetic data') #29
    parser.add_argument('--max_start_epoch', type=int, default=80, help='max epoch we can start at')

    # SelMatch
    parser.add_argument('--initialize', type=str, default='window', help='how to initialize dataset (random, window)')
    parser.add_argument('--score', type=str, default='cscore', help='Sample difficulty metric (cscore, forgetting)')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--alpha', type=float, default=0.5, help='Distillation portion')
    parser.add_argument('--beta', type=float, default=0.5, help='Difficulty level')

    # Evaluation
    parser.add_argument('--model_eval', type=str, default='ResNet18BN', help='model for evaluation')
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    parser.add_argument('--epoch_eval_train', type=int, default=500, help='epochs to train a model with synthetic data')
    parser.add_argument('--lr_net', type=float, default=0.1, help='learning rate for evaluation')
    parser.add_argument('--batch_train', type=int, default=128, help='batch size for training networks')
    parser.add_argument('--aug', type=str, default='combined', help='augmentation method (dsa, simple, combined)')

    args = parser.parse_args()

    main(args)
