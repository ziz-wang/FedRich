#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # global arguments
    parser.add_argument('--machine', type=str, default='local', choices=['local', 'remote'],
                        help='run the code on which machine')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda:0'],
                        help='cpu or gpu')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fmnist', 'cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--seed', type=int, default=7,
                        help='random seed for random initialization')
    parser.add_argument('--n_clients', type=int, default=100,
                        help='number total clients')
    parser.add_argument('--resumed', type=int, default=0, choices=[0, 1],
                        help='whether resume the trained model, 1 ---> resume')
    parser.add_argument('--resumed_name', type=str, default='SFL_09131722/SFL_round_1000.pth',
                        help='the path of resumed model')
    parser.add_argument('--protocol', type=str, default='FedRich', choices=['FedRich'],
                        help='training protocol')
    parser.add_argument('--rounds', type=int, default=1000,
                        help='communication rounds between server and edge devices')
    parser.add_argument('--participating_ratio', type=float, default=0.2,
                        help='the fraction of total clients participating training')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate of the client')
    parser.add_argument('--lr_decay', type=float, default=0.998,
                        help='learning rate decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='SGD weight_decay')
    parser.add_argument('--record_step', type=int, default=1000,
                        help='save global model every {record_step} rounds')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='local batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='the number of local epochs')
    parser.add_argument('--aggregation_method', type=str, default='w_avg', choices=['w_avg', 'avg'],
                        help='weighted average or average')
    parser.add_argument('--feature_memo', type=int, default=0, choices=[0, 1],
                        help='whether print the memory usage of the features, 1 ---> print')

    # data distribution arguments
    parser.add_argument('--classes_per_client', type=int, default=2,
                        help='classes of each distributed client')
    parser.add_argument('--balancedness', type=float, default=0.98,
                        help='control the data size distribution, 1 ---> uniform distribution')

    # protocol arguments
    # shared arguments of FedRich, HFL and SFL
    parser.add_argument('--sampling_ratio', type=float, default=0.06,
                        help='sampling ratio of the local data')
    parser.add_argument('--mediator_num', type=int, default=3,
                        help='the number of mediators')
    parser.add_argument('--mediator_epochs', type=int, default=10,
                        help='training epochs of mediators')
    parser.add_argument('--keep_bn', type=int, default=1, choices=[0, 1],
                        help='whether keep BN layers in a neural network, 1 ---> keep')
    parser.add_argument('--compress_ratio', type=float, default=1,
                        help='whether compress the features, less than 0.5 ---> compress')
    parser.add_argument('--gap', type=int, default=0, choices=[0, 1],
                        help='whether gap the features with a specific matrix, 1 ---> gap')
    parser.add_argument('--sigma', type=float, default=0,
                        help='scale of the noise that follows a gaussian distribution')
    parser.add_argument('--l2_norm_clip', type=float, default=1,
                        help='if adding noise, clipping the l2 norm of the gradient')
    # FedRich
    parser.add_argument('--redundant_forward', type=int, default=1, choices=[0, 1],
                        help='whether conduct redundant forward after updating the local model')
    parser.add_argument('--heuristic_search', type=int, default=1, choices=[0, 1],
                        help='whether select participants using heuristic search strategy')
    parser.add_argument('--split_points', type=str, default='1, 5, 9',
                        help='split point of a complete neural network')
    parser.add_argument('--random_select_ratio', type=float, default=0.4,
                        help='split point of a complete neural network')
    parser.add_argument('--beta', type=float, default=0.54369, choices=[0.54369, 0.51879],
                        help='the base of an exponential distribution')

    # HFL
    parser.add_argument('--split_point', type=int, default=4,
                        help='split point of a complete neural network')
    parser.add_argument('--cluster', type=int, default=0, choices=[0, 1],
                        help='whether cluster the clients according their likelihood distribution, 1 ---> cluster')
    parser.add_argument('--n_clusters', type=int, default=5,
                        help='the number of the clusters')
    parser.add_argument('--device_distribution', type=str, default='exponential',
                        choices=['exponentially', 'dirichlet'], help='the number of devices in each mediator')
    parser.add_argument('--alpha', type=float, default=3,
                        help='the hyper-parameter of dirichlet')

    config_arg = parser.parse_args()
    return config_arg


config_args = args_parser()
