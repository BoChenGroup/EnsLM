import re
import os
import random
import tarfile
import urllib
import numpy as np
# from torchtext import data
import pickle
from torch.utils.data import Dataset, DataLoader
# load IMDB dataset

class CustomTrainDataset(Dataset):
    def __init__(self, topic_data_file):
        with open(topic_data_file, 'rb') as f:
            data = pickle.load(f)
        self.train_data = data['train_bow']
        self.N, self.V = self.train_data.shape
        self.voc = data['tm_idx2vocab']
    def __getitem__(self, index):
        topic_data = self.train_data[index].toarray()
        return np.squeeze(topic_data), 0
    def __len__(self):
        return self.N

class CustomTestDataset(Dataset):
    def __init__(self, topic_data_file):
        with open(topic_data_file, 'rb') as f:
            data = pickle.load(f)
        self.train_data = data['test_bow']
        self.N, self.V = self.train_data.shape
        self.voc = data['tm_idx2vocab']
    def __getitem__(self, index):
        topic_data = self.train_data[index].toarray()
        return np.squeeze(topic_data), 0
    def __len__(self):
        return self.N

def get_train_loader(topic_data_file, batch_size=200, shuffle=True, num_workers=0, train_flag=True):
    dataset = CustomTrainDataset(topic_data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.voc, dataset.V

def get_test_loader(topic_data_file, batch_size=200, shuffle=True, num_workers=0, train_flag=True):
    dataset = CustomTestDataset(topic_data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.voc, dataset.V

class CustomDataset(Dataset):
    def __init__(self, data_file, train_flag):
        with open(data_file, 'rb') as f:
            imdb_data = pickle.load(f)

        data_name = data_file.split('.')[-2]
        if data_name[-5:] == 'split':
            if train_flag:
                self.doc_index = imdb_data['train_doc_split']  # [1150:13500]
                self.doc_label = imdb_data['train_doc_label']  # [11500:13500]
                self.index2word = imdb_data['word2index']
                self.class_num = len(imdb_data['labels'])
            else:
                self.doc_index = imdb_data['test_doc_split']
                self.doc_label = imdb_data['test_doc_label']
                self.index2word = imdb_data['word2index']
                self.class_num = len(imdb_data['labels'])
        else:
            if train_flag:
                self.doc_index = imdb_data['train_doc_index']  # [1150:13500]
                self.doc_bow = imdb_data['train_doc_bow'].toarray().float()
                self.doc_label = imdb_data['train_doc_label']  # [11500:13500]
                self.index2word = imdb_data['index2word']
                self.class_num = len(imdb_data['labels'])
                self.doc_task = imdb_data['train_task']
            else:
                self.doc_index = imdb_data['test_doc_index']
                self.doc_bow = imdb_data['test_doc_bow'].toarray().float()
                self.doc_label = imdb_data['test_doc_label']
                self.index2word = imdb_data['index2word']
                self.class_num = len(imdb_data['labels'])
                self.doc_task = imdb_data['test_task']

        self.N = len(self.doc_index)
        self.V = len(self.index2word)

        del imdb_data

    def __getitem__(self, index):
        doc_index = np.array(self.doc_index[index])
        doc_label = np.array(self.doc_label[index])
        doc_bow = np.array(self.doc_bow[index])
        doc_task = np.array(self.doc_task[index])
        return doc_bow, doc_task

    def __len__(self):
        return self.N

def get_loader(data_file, batch_size = 200, train_flag=True, num_workers = 0):
    if train_flag:
       shuffle = True
    else:
       shuffle = False

    dataset = CustomDataset(data_file, train_flag=train_flag)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True),\
           dataset.V, dataset.class_num, dataset.index2word