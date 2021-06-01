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

## Read Data
#########################################################
import pickle
from torch.utils.data import Dataset, DataLoader

class CustomTrainDataset(Dataset):
    def __init__(self, topic_data_file, voc_path='voc.txt', train_flag=True, max_size=None, topic_max_sents=100,
                 topic_v=10000):
        with open(topic_data_file, 'rb') as f:
            data = pickle.load(f)
        train_id = data['train_id']
        train_data = data['data_2000'][train_id]
        train_label = np.array(data['label'])[train_id]
        voc = data['voc2000']
        self.train_data = train_data
        self.train_label = train_label
        self.N, self.V = self.train_data.shape
        self.voc = voc

    def __getitem__(self, index):
        topic_data = self.train_data[index].toarray()
        label = self.train_label[index]
        return np.squeeze(topic_data), np.squeeze(label)

    def __len__(self):
        return self.N

class CustomTestDataset(Dataset):
    def __init__(self, topic_data_file, voc_path='voc.txt', train_flag=True, max_size=None, topic_max_sents=100,
                 topic_v=10000):
        with open(topic_data_file, 'rb') as f:
            data = pickle.load(f)
        test_id = data['test_id']
        test_data = data['data_2000'][test_id]
        test_label = np.array(data['label'])[test_id]
        voc = data['voc2000']
        self.test_data = test_data
        self.test_label = test_label
        self.N, self.V = self.test_data.shape
        self.voc = voc

    def __getitem__(self, index):
        topic_data = self.test_data[index].toarray()
        label = self.test_label[index]
        return np.squeeze(topic_data), np.squeeze(label)

    def __len__(self):
        return self.N

def get_train_loader(topic_data_file, batch_size=200, shuffle=True, num_workers=0, train_flag=True):
    dataset = CustomTrainDataset(topic_data_file, train_flag=train_flag)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.voc, dataset.V

def get_test_loader(topic_data_file, batch_size=200, shuffle=False, num_workers=0, train_flag=True):
    dataset = CustomTestDataset(topic_data_file, train_flag=train_flag)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.voc, dataset.V

class CustomTrainDatasetr8(Dataset):
    def __init__(self, topic_data_file, voc_path='voc.txt', train_flag=True, max_size=None, topic_max_sents=100,
                 topic_v=10000):
        with open(topic_data_file, 'rb') as f:
            data = pickle.load(f)
        train_id = data['train_id']
        train_data = data['bow'][train_id]
        train_label = np.array(data['label'])[train_id]
        voc = data['vocab']
        self.train_data = train_data
        self.train_label = train_label
        self.N, self.V = self.train_data.shape
        self.voc = voc

    def __getitem__(self, index):
        topic_data = self.train_data[index].toarray()
        label = np.argmax(self.train_label[index])
        return np.squeeze(topic_data), np.squeeze(label)

    def __len__(self):
        return self.N

class CustomTestDatasetr8(Dataset):
    def __init__(self, topic_data_file, voc_path='voc.txt', train_flag=True, max_size=None, topic_max_sents=100,
                 topic_v=10000):
        with open(topic_data_file, 'rb') as f:
            data = pickle.load(f)
        test_id = data['test_id']
        test_data = data['bow'][test_id]
        test_label = np.array(data['label'])[test_id]
        voc = data['vocab']
        self.test_data = test_data
        self.test_label = test_label
        self.N, self.V = self.test_data.shape
        self.voc = voc

    def __getitem__(self, index):
        topic_data = self.test_data[index].toarray()
        label = np.argmax(self.test_label[index])
        return np.squeeze(topic_data), np.squeeze(label)

    def __len__(self):
        return self.N

def get_train_loader_r8(topic_data_file, batch_size=200, shuffle=True, num_workers=0, train_flag=True):
    dataset = CustomTrainDatasetr8(topic_data_file, train_flag=train_flag)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.voc, dataset.V

def get_test_loader_r8(topic_data_file, batch_size=200, shuffle=False, num_workers=0, train_flag=True):
    dataset = CustomTestDatasetr8(topic_data_file, train_flag=train_flag)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.voc, dataset.V