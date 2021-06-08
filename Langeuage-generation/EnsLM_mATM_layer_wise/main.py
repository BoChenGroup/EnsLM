import warnings
warnings.filterwarnings("ignore")
import pickle
import json
import random
import torch
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
from model_style.modeling_gpt2 import GPT2LMHeadModel_style
from torch import optim
from tqdm import tqdm
import os
import numpy as np

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
n_gpu = torch.cuda.device_count()

def toogle_grad(model, requires_grad):
    for name, p in model.named_parameters():
        p.requires_grad_(requires_grad)

def sample_sequence(model, length, context, device='cuda'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    generated = context
    with torch.no_grad():
        for _ in range(length):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)
            next_token_logits = outputs[0][-1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            if next_token == 50256:
                break
            else:
                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
        return generated

def Prepare_test_data(data, label):
    maxlength = 0
    sen_index_minibatch_list = []
    for sen in tqdm(data[:]):
        sen = '<|endoftext|> ' + sen + ' <|endoftext|>'
        sen_index_tensor = tokenizer.encode(sen, return_tensors='pt')
        if sen_index_tensor.size()[1] > 1024:
            sen_index_tensor = sen_index_tensor[:, :1024]
        sen_index_minibatch_list.append(sen_index_tensor)
        maxlength = max(sen_index_tensor.size()[1], maxlength)

    sen_minibatch_index_tensor = 50256 * torch.ones([len(data[:]), maxlength], dtype=int)
    label_minibatch_index_tensor = -100 * torch.ones([len(data[:]), maxlength], dtype=int)
    sen_minibatch_label = np.zeros((len(data[:]), Total_num_class))

    for num in tqdm(range(len(sen_index_minibatch_list))):
        sen_index_tensor = sen_index_minibatch_list[num]
        length = sen_index_tensor.size()[1]
        sen_minibatch_index_tensor[num, 0:length] = sen_index_tensor
        label_minibatch_index_tensor[num, 1:length] = sen_index_tensor[0, 1:]
        sen_minibatch_label[num, :] = label[num][:]

    return[sen_minibatch_index_tensor, label_minibatch_index_tensor, sen_minibatch_label]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Load GPT2
tokenizer = GPT2Tokenizer.from_pretrained('gpt/cnndm')
model = GPT2LMHeadModel_style.from_pretrained('gpt/cnndm')
model.train()
model.to(device)

if n_gpu > 1:
    model = torch.nn.DataParallel(model)

#  'dataset/bnc/data.pkl' 'dataset/imdb/data.pkl'

file_path = 'dataset/imdb/data.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

data_train = data['train_doc_lm_sents_txt']
train_label = data['train_domain_label']
data_test = data['test_doc_lm_sents_txt']
test_label = data['test_domain_label']
Total_num_class = 64

data_train_sent = []
data_train_label = []
for i in range(len(data_train)):
    doc = data_train[i]
    label = train_label[i, :]
    for sent in doc:
        data_train_sent.append(sent)
        data_train_label.append(label)

print('train_data_num', len(data_train_sent))

data_test_sent = []
data_test_label = []
for i in range(len(data_test)):
    doc = data_test[i]
    label = test_label[i, :]
    for sent in doc:
        data_test_sent.append(sent)
        data_test_label.append(label)

print('test_data_num', len(data_test_sent))


Test_data = Prepare_test_data(data_test_sent, data_test_label)
Train_num = len(data_train_sent)
batchsize = 8
accomplish = 1

## construct optimizer
GPT_params = model.parameters()
Total_lr_decay = 0
learning_rate = 1e-6
GPT_optimizer = optim.Adam(GPT_params, lr=learning_rate)

##
best_loss = 1000000
not_improved = 0

toogle_grad(model, True)
for epoch in range(500):
    print('Start epoch %d...' % epoch)

    # random.shuffle(data_train)

    Index = np.arange(len(data_train_sent))
    np.random.shuffle(Index)

    for iteration in range(Train_num // batchsize):
        mini_batch_index = Index[iteration * batchsize: (iteration + 1) * batchsize]
        data_minibatch = []

        data_label_minibatch = np.zeros((batchsize, Total_num_class))
        for i in range(mini_batch_index.shape[0]):
            data_minibatch.append(data_train_sent[mini_batch_index[i]])
            data_label_minibatch[i, :] = data_train_label[mini_batch_index[i]][:]

        data_label_minibatch = torch.from_numpy(data_label_minibatch).float().to(device)
        Loss = 0
        maxlength = 0
        sen_index_minibatch_list = []
        for sen in data_minibatch:
            sen = '<|endoftext|> ' + sen + ' <|endoftext|>'
            sen_index_tensor = tokenizer.encode(sen, return_tensors='pt')
            sen_index_minibatch_list.append(sen_index_tensor)
            maxlength = max(sen_index_tensor.size()[1], maxlength)

        if maxlength*batchsize > 2400:
            continue

        sen_minibatch_index_tensor = 50256*torch.ones([batchsize, maxlength], dtype=int).to(device)
        label_minibatch_index_tensor = -100 * torch.ones([batchsize, maxlength], dtype=int).to(device)

        for num in range(len(sen_index_minibatch_list)):
            sen_index_tensor = sen_index_minibatch_list[num]
            length = sen_index_tensor.size()[1]
            sen_minibatch_index_tensor[num, 0:length] = sen_index_tensor
            label_minibatch_index_tensor[num, 1:length] = sen_index_tensor[0, 1:]

        output = model(sen_minibatch_index_tensor, labels=label_minibatch_index_tensor, cluster=data_label_minibatch, label_ignore=-100)
        if n_gpu > 1:
            Loss = output[0].mean()
        else:
            Loss = output[0]

        Loss.backward()
        if iteration % accomplish == 0:
            GPT_optimizer.step()

        if (iteration+1) % 10 == 0:
            print('The training loss at epoch %d iteration %d is %4f, Total lr decay is %d' % (epoch+1, iteration+1, Loss, Total_lr_decay))

        # if (iteration+1) % 1000 == 0:
        #     model = model.module if hasattr(model, "module") else model
        #     model.eval()
        #     prompt_text = "<|endoftext|>"
        #     context_tokens = tokenizer.encode(prompt_text)
        #
        #     index = np.random.choice(test_label, 1, replace=False)
        #
        #     index = torch.tensor(to_categorical(index, num_classes=Total_num_class)).float().to(device)
        #
        #     out = model.generate_style(cluster_index=index, max_length=30, bos_token_id=tokenizer.bos_token_id,
        #                                eos_token_ids=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
        #                                do_sample=True, num_beams=5, num_return_sequences=1, temperature=1)
        #
        #     generated_list = out[0, 1:].tolist()
        #     text = tokenizer.decode(generated_list)
        #     print('\n')
        #     print('The generated sentence at epoch %d iteration %d is:' % (epoch + 1, iteration + 1))
        #     print(text + '\n')
        #
        #     if n_gpu > 1:
        #         model = torch.nn.DataParallel(model)
        #     model.train()

        if (iteration+1) % 10000 == 0:
            model = model.module if hasattr(model, "module") else model
            model.eval()

            print('Calculate loss on validation dataset')
            with torch.no_grad():
                test_loss = 0
                for it in range(len(Test_data[0]) // 10):
                    test_ids = Test_data[0][it * 10: (it + 1) * 10].to(device)
                    test_label = Test_data[1][it * 10: (it + 1) * 10].to(device)
                    cluster_index = torch.tensor(Test_data[2][it * 10: (it + 1) * 10]).float().to(device)

                    output = model(test_ids, labels=test_label, cluster=cluster_index)
                    loss = output[0]
                    test_loss += loss.cpu().item()

                test_loss = test_loss / (it+1)

            if test_loss < best_loss:
                best_loss = test_loss
                print('\n')
                #torch.save(model, './trained_model/coco_caption/gpt2.pkl')
                not_improved = 0
            else:
                not_improved += 1

                if not_improved == 5:
                    Total_lr_decay += 1
                    not_improved = 0
                    learning_rate = learning_rate*0.5
                    GPT_optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            print('At epoch %d iteration %d, the validation loss is %4f, the best loss is %4f' % (epoch + 1, iteration + 1, test_loss, best_loss))
            print('At epoch %d iteration %d, the validation ppl is %4f, the best ppl is %4f' % (epoch + 1, iteration + 1, np.exp(test_loss), np.exp(best_loss)))

            print('\n')

            if n_gpu > 1:
                model = torch.nn.DataParallel(model)
            model.train()





