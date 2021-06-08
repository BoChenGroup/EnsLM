#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import model
import train
from logger import get_logger

from mydatasets import get_loader

parser = argparse.ArgumentParser(description='CNN text classificer')
#parser = argparse.get_args()
# learning
parser.add_argument('-lr', type=float, default=1e-6, help='initial learning rate [default: 0.001]')  # CNN or RNN: 1e-3, DocBert: 1e-6
parser.add_argument('-epochs', type=int, default=50, help='number of epochs for train [default: 256]')
parser.add_argument('-batchsize', type=int, default=8, help='batch size for training [default: 64]')  # CNN or RNN: 50, DocBert: 8
parser.add_argument('-accumulation-steps',
                    type=int, default=2, help='batch size for training [default: 64]')   # CNN or RNN: 1, DocBert: 2

parser.add_argument('-log-interval', type=int, default=10, help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=500, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')

# data
parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
parser.add_argument('-train_dir', type=str, default='../dataset/gmlda_mtl_bert_dataset.pkl', help='Train Dir')
#cnn/rnn:'../dataset/gmlda_bert_dataset.pkl'     bert: '../dataset/gmlda_mtl_bert_dataset.pkl'
parser.add_argument('-dataset', type=str, default='mti_data')
parser.add_argument('-index2word', type=list, default=[])

# model
parser.add_argument('-model', type=str, default='DocBertMtl', help='the model name')
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=150, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3', help='comma-separated kernel size to use for convolution')
parser.add_argument('-query-kernel-num', type=int, default=150, help='query channel num')
parser.add_argument('-layer-num', type=int, default=2, help='dpcnn layer num for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
parser.add_argument('-vocab-num', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-PositionalEmbedding', type=bool, default=False, help='if using Positional Embedding')

# optimizer
parser.add_argument('-optimizer', type=str, default='adam', help='number of embedding dimension [default: 128]')
parser.add_argument('-weight_decay', type=float, default=1e-5, help='number of embedding dimension [default: 128]')
parser.add_argument('-sent_loss', type=bool, default=False, help='number of embedding dimension [default: 128]')
parser.add_argument('-optimizer_warper', type=bool, default=False, help='number of embedding dimension [default: 128]')

# device
parser.add_argument('-obv_device', type=str, default='0', help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')

# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=train, help='train or test')
# args = parser.parse_args(args=[])
args = parser.parse_args()

# usage
logger = get_logger('log/' + args.model + '_' + args.dataset + '.log')
logger.info('start training!')


# load data
print("\n Loading data...")

train_iter, vocab_size, class_num, index2word = get_loader(args.train_dir, batch_size=args.batchsize, train_flag=True)
test_iter, vocab_size, class_num, index2word = get_loader(args.train_dir, batch_size=args.batchsize, train_flag=False)

args.index2word = index2word
# update args and print

args.vocab_num = vocab_size
args.class_num = class_num
args.cuda = (not args.no_cuda) and torch.cuda.is_available()
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\n Parameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))
    logger.info("\t{}={}".format(attr.upper(), value))

# model
if args.model == 'Text_CNN':
   cnn = model.Text_CNN(args)
elif args.model == 'Text_CNN_att':
   cnn = model.Text_CNN_att(args)
elif args.model == 'DPCNN':
   cnn = model.DPCNN(args)
elif args.model == 'DPCNN_att':
    cnn = model.DPCNN_att(args)
elif args.model == 'DPCNN_multi':
    cnn = model.DPCNN_multi(args)
elif args.model == 'Text_CNN_CBAM':
    cnn = model.Text_CNN_CBAM(args)
elif args.model == 'Text_LSTM':
    cnn = model.Text_LSTM(args)
elif args.model == 'Text_MLP':
    cnn = model.Text_MLP(args)
elif args.model == 'Text_CNN_Style':
    cnn = model.Text_CNN_Style(args)
elif args.model == 'WV_Text_CNN':
    cnn = model.WV_Text_CNN(args)
elif args.model == 'Text_LSTM_Style':
    cnn = model.Text_LSTM_Style(args)
elif args.model == 'Text_CNN_att_MTL':
    cnn = model.Text_CNN_att_MTL(args)
elif args.model == 'Text_LSTM_att_MTL':
    cnn = model.Text_LSTM_att_MTL(args)
elif args.model == 'DocBertMtl':
    cnn = model.DocBertMtl(args)


if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

os.environ['CUDA_VISIBLE_DEVICES'] = args.obv_device
args.device = 'cuda' if args.cuda else 'cpu'


train.train(train_iter, test_iter, cnn, args, logger)
logger.info('finish training!')
