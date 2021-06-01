import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np
import os
import torch
# from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data
from model.GMVAE import *
from my_dataset import get_train_loader, get_test_loader

#########################################################
## Input Parameters
#########################################################
parser = argparse.ArgumentParser(description='PyTorch Implementation of DGM Clustering')

## Used only in notebooks
parser.add_argument('-f', '--file', help='Path for input file. First line should contain number of lines to search in')

## Dataset
parser.add_argument('--dataset', type=str, choices=['20ng'], default='20ng', help='dataset (default: mnist)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')

## GPU
parser.add_argument('--cuda', type=int, default=1, help='use of cuda (default: 1)')
parser.add_argument('--gpuID', type=int, default=0, help='set gpu id to use (default: 0)')

## Training
parser.add_argument('--epochs', type=int, default=400, help='number of total epochs to run (default: 200)')
parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size (default: 64)')
parser.add_argument('--batch_size_val', default=200, type=int, help='mini-batch size of validation (default: 200)')
parser.add_argument('--learning_rate', default=1e-2, type=float, help='learning rate (default: 0.001)')
parser.add_argument('--decay_epoch', default=-1, type=int, help='Reduces the learning rate every decay_epoch')
parser.add_argument('--lr_decay', default=0.5, type=float, help='Learning rate decay for training (default: 0.5)')

## Architecture
parser.add_argument('--num_classes', type=int, default=20, help='number of classes (default: 10)')
parser.add_argument('--gaussian_size', default=20, type=int, help='gaussian size (default: 64)')
parser.add_argument('--input_size', default=2000, type=int, help='input size (default: 784)')

## Partition parameters
parser.add_argument('--train_proportion', default=1.0, type=float, help='proportion of examples to consider for training only (default: 1.0)')

## Gumbel parameters
parser.add_argument('--init_temp', default=0.5, type=float, help='Initial temperature used in gumbel-softmax (recommended 0.5-1.0, default:1.0)')
parser.add_argument('--decay_temp', default=1, type=int, help='Set 1 to decay gumbel temperature at every epoch (default: 1)')
parser.add_argument('--hard_gumbel', default=0, type=int, help='Set 1 to use the hard version of gumbel-softmax (default: 1)')
parser.add_argument('--min_temp', default=0.5, type=float, help='Minimum temperature of gumbel-softmax after annealing (default: 0.5)' )
parser.add_argument('--decay_temp_rate', default=0.013862944, type=float, help='Temperature decay rate at every epoch (default: 0.013862944)')

## Loss function parameters
parser.add_argument('--w_gauss', default=1, type=float, help='weight of gaussian loss (default: 1)')
parser.add_argument('--w_categ', default=1, type=float, help='weight of categorical loss (default: 1)')
parser.add_argument('--w_rec', default=1, type=float, help='weight of reconstruction loss (default: 1)')
parser.add_argument('--rec_type', type=str, choices=['bce', 'mse', 'nse'], default='nse', help='desired reconstruction loss function (default: bce)')

## Others
parser.add_argument('--verbose', default=1, type=int, help='print extra information at every epoch.(default: 0)')
parser.add_argument('--random_search_it', type=int, default=20, help='iterations of random search (default: 20)')

args = parser.parse_args()

if args.cuda == 1:
   os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

## Random Seed
SEED = args.seed
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if args.cuda:
  torch.cuda.manual_seed(SEED)

## Calculate flatten size of each input data
args.input_size = 2000
print(args.input_size)
#########################################################
## Train and Test Model
#########################################################
gmvae = GMVAE(args)

train_loader, voc, v = get_train_loader('data/20ng.pkl', train_flag=True, batch_size=args.batch_size)
test_loader, voc, v = get_test_loader('data/20ng.pkl', train_flag=False, batch_size=args.batch_size)

# ## Training Phase
history_loss = gmvae.train(train_loader, test_loader)