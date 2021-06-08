import torch
from networks.Networks import *
import pickle

model_dir = 'saves/mtl_data_gaussian_size_16_num_classes_16'
network = GMVAENet(30000, 16, 16)
checkpoint = torch.load(model_dir)
network.load_state_dict(checkpoint['net'])
network.eval()
network.cuda()

train_dir = '../dataset/mtl_dataset.pkl'
with open(train_dir, 'rb') as f:
     data = pickle.load(f)

train_doc_bow = data['train_doc_bow'].toarray()
test_doc_bow = data['test_doc_bow'].toarray()

train_doc_cluster = []
test_doc_cluster = []

train_doc_theta = []
test_doc_theta = []

for i in range(len(train_doc_bow)):
    doc_bow = torch.from_numpy(train_doc_bow[i]).cuda().float()

    # flatten data
    doc_bow = doc_bow.view(1, -1)

    # forward call
    out_net = network(doc_bow, 0.5, 0)
    pre_label = torch.argmax(out_net['categorical'], 1).cpu().detach().item()
    train_doc_cluster.append(pre_label)

    theta = out_net['mean'].cpu().detach().numpy()
    train_doc_theta.append(theta)

for i in range(len(test_doc_bow)):
    doc_bow = torch.from_numpy(test_doc_bow[i]).cuda().float()

    # flatten data
    doc_bow = doc_bow.view(1, -1)

    # forward call
    out_net = network(doc_bow, 0.5, 0)
    pre_label = torch.argmax(out_net['categorical'], 1).cpu().detach().item()
    test_doc_cluster.append(pre_label)

    theta = out_net['mean'].cpu().detach().numpy()
    test_doc_theta.append(theta)

data['train_doc_cluster'] = train_doc_cluster
data['test_doc_cluster'] = test_doc_cluster

data['train_doc_theta'] = train_doc_theta
data['test_doc_theta'] = test_doc_theta

save_dir = '../dataset/gmlda_mtl_dataset_16.pkl'
pickle.dump(data, open(save_dir, 'wb'))
