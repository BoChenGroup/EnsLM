import torch
from networks.Networks import *
import pickle
import numpy as np
import torch.utils.data
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


import scipy.sparse as sparse

model_dir = 'saves/agnews_gaussian_size_64_num_classes_64'
data_path = 'dataset/apnews/data.pkl'

with open(data_path, 'rb') as f:
    data = pickle.load(f)

train_bow = data['train_bow'].toarray()
test_bow = data['test_bow'].toarray()
v = train_bow.shape[1]

network = GMVAENet(v, 64, 64)
checkpoint = torch.load(model_dir)
network.load_state_dict(checkpoint['net'])
network.cuda()
network.eval()

batch_size = 5000

for i in range(10):
    train_data = torch.tensor(train_bow[i*batch_size:(i+1)*batch_size], dtype=torch.float).cuda()
    train_data = train_data.view(train_data.size(0), -1)
    out_net = network(train_data, 0.5, 0)
    train_domain_label = out_net['categorical'].detach().cpu().numpy()

    train_theta = out_net['mean'].detach().cpu()
    train_theta = train_theta.view(train_theta.size(0), -1)

    if i == 0:
        train_all_data_label = train_domain_label
        train_all_data_theta = train_theta
    else:
        train_all_data_label = np.vstack((train_all_data_label, train_domain_label))
        train_all_data_theta = np.vstack((train_all_data_theta, train_theta))



test_data = torch.tensor(test_bow, dtype=torch.float).cuda()
test_data = test_data.view(test_data.size(0), -1)
out_net = network(test_data, 0.5, 0)
test_domain_label = out_net['categorical'].detach().cpu().numpy()

test_theta = out_net['mean'].detach().cpu()
test_theta = test_theta.view(test_theta.size(0), -1)


data['train_domain_label'] = train_all_data_label
data['test_domain_label'] = test_domain_label

data['train_theta'] = train_all_data_theta
data['test_theta'] = test_theta

pickle.dump(data, open('dataset/apnews/data.pkl', 'wb'))


