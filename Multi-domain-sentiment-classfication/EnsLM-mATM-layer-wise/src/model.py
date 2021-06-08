import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

from model_bert_style.modeling_bert import BertModel

class DocBertMtl(nn.Module):
    def __init__(self, args):
        super(DocBertMtl, self).__init__()
        self.class_num = args.class_num
        self.model = BertModel.from_pretrained('../bert-base-uncased')
        self.dropout_rate = args.dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(768, self.class_num)
        self.task_num = 16

    def forward(self, sen_index, cluster):
        cluster_onehot = torch.zeros(cluster.size(0), self.task_num).cuda()
        cluster_onehot.scatter_(1, cluster.view(-1, 1), 1)

        context_embed, _ = self.model(sen_index, cluster=cluster_onehot)
        context_embed = context_embed.transpose(1, 2)
        # pooled_output = F.max_pool1d(context_embed, kernel_size=(context_embed.shape[2])).squeeze(2)
        pooled_output = context_embed[:, :, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class Conv1D_style(nn.Module):
    def __init__(self, din, dout, Num_cluster):
        super().__init__()
        w = torch.rand(din, dout)
        self.weight = nn.Parameter(w)
        self.weight.data.uniform_(-0.25, 0.25)
        #self.bias = nn.Parameter(torch.zeros(nf))

        self.style_L = nn.Parameter(torch.ones(Num_cluster, din))
        self.style_R = nn.Parameter(torch.ones(Num_cluster, dout))

    def forward(self, x, cluster):
        '''
        :param x: batch * din
        :param cluster: batch * Num_cluster
        :return:
        '''

        tmp_L = torch.matmul(cluster, self.style_L)    #batch * din
        tmp_R = torch.matmul(cluster, self.style_R)    #batch * dout
        gamma = torch.matmul(tmp_L.unsqueeze(2), tmp_R.unsqueeze(1))       #batch * din * dout

        new_weight = self.weight.unsqueeze(0) * gamma
        x = torch.matmul(x, new_weight).squeeze(1)
            # + self.bias.unsqueeze(0).unsqueeze(1)
        return x

class Text_CNN_att_MTL(nn.Module):

    def __init__(self, args):
        super(Text_CNN_att_MTL, self).__init__()
        self.args = args

        self.vocab_num = args.vocab_num
        self.embed_dim = args.embed_dim
        self.kernel_num = args.kernel_num
        self.kernel_sizes = args.kernel_sizes
        self.class_num = args.class_num
        self.task_num = 16

        self.embed = nn.Embedding(self.vocab_num, self.embed_dim, padding_idx=args.vocab_num-1)
        self.embed_drop = nn.Dropout(0.5)  # 0.1-0.2
        self.conv_value = nn.Conv2d(1, self.kernel_num, (3, self.embed_dim))  # kernel_size = 3
        self.conv_query = nn.Conv2d(1, 64, (3, self.embed_dim))
        self.conv_key = Conv1D_style(64, 1, self.task_num)

        self.bn_value = nn.BatchNorm2d(self.kernel_num)
        self.bn_query = nn.BatchNorm2d(64)

        self.fc_drop = nn.Dropout(0.5)  # 0.5
        self.fc1 = nn.Linear(self.kernel_num, self.class_num)

    def forward(self, word_index, cluster):
        cluster_onehot = torch.zeros(cluster.size(0), self.task_num).cuda()
        cluster_onehot.scatter_(1, cluster.view(-1, 1), 1)

        word_embedding = self.embed(word_index)  # N*L*D
        word_embedding = self.embed_drop(word_embedding)
        word_embedding = word_embedding.unsqueeze(1)  # N*1*L*D

        conv_value = F.relu(self.bn_value(self.conv_value(word_embedding))).squeeze(3)  # N*K*L
        conv_query = F.relu(self.bn_query(self.conv_query(word_embedding))).squeeze(3)  # N*64*L

        pool_1 = self.att_pool(conv_value, conv_query, self.conv_key, cluster_onehot)
        concat_1 = self.fc_drop(pool_1)  # N*K
        logit = self.fc1(concat_1)

        return logit

    def att_pool(self, value, query, key, cluster):

        query = query.transpose(1, 2)   # N*L*64
        # att_matrix = torch.matmul(query, key)  # N*L*1
        att_matrix = key(query, cluster)
        att_matrix = F.softmax(att_matrix, dim=1)  # N*L*1
        pool = torch.matmul(value, att_matrix).squeeze(2)  # N*K

        return pool


class Text_LSTM_att_MTL(nn.Module):
    def __init__(self, args):
        super(Text_LSTM_att_MTL, self).__init__()
        self.args = args

        self.vocab_num = args.vocab_num
        self.embed_dim = args.embed_dim
        self.kernel_num = args.kernel_num
        self.kernel_sizes = args.kernel_sizes
        self.class_num = args.class_num
        self.index2word = args.index2word
        self.task_num = 16

        # self.wv_matrix = self.load_vector(self.index2word)
        self.embed = nn.Embedding(self.vocab_num, self.embed_dim, padding_idx=args.vocab_num-1)
        # self.embed.weight.data.copy_(torch.from_numpy(self.wv_matrix))
        self.embed_drop = nn.Dropout(0.5)  # 0.1-0.2
        self.lstm_value = nn.LSTM(input_size=self.embed_dim, hidden_size=self.kernel_num, num_layers=1,
                            batch_first=True, dropout=0)
        self.lstm_query = nn.LSTM(input_size=self.embed_dim, hidden_size=64, num_layers=1,
                                  batch_first=True, dropout=0)

        self.conv_key = Conv1D_style(64, 1, self.task_num)

        self.bn_value = nn.BatchNorm1d(self.kernel_num)
        self.bn_query = nn.BatchNorm1d(64 * 2)

        self.fc_drop = nn.Dropout(0.5)  # 0.5
        self.fc1 = nn.Linear(self.kernel_num, self.class_num)


    def forward(self, word_index, cluster):
        cluster_onehot = torch.zeros(cluster.size(0), self.task_num).cuda()
        cluster_onehot.scatter_(1, cluster.view(-1, 1), 1)

        word_embedding = self.embed(word_index)        # N*L*D
        word_embedding = self.embed_drop(word_embedding)     # N*L*D

        hidden_value, (_, _) = self.lstm_value(word_embedding)
        # hidden_value = F.relu(self.bn_value(hidden_value))

        hidden_query, (_, _) = self.lstm_query(word_embedding)
        # hidden_query = F.relu(self.bn_value(hidden_query))

        pool_1 = self.att_pool(hidden_value, hidden_query, self.conv_key, cluster_onehot)
        concat_1 = self.fc_drop(pool_1)  # N*K
        logit = self.fc1(concat_1)

        return logit

    def att_pool(self, value, query, key, cluster):

        value = value.transpose(1, 2)   # N*600*L
        # att_matrix = torch.matmul(query, key)  # N*L*1
        att_matrix = key(query, cluster)  # N*L*1
        att_matrix = F.softmax(att_matrix, dim=1)  # N*L*1
        pool = torch.matmul(value, att_matrix).squeeze(2)  # N*K
        return pool

    def load_vector(self, vocab):
        word_vectors = KeyedVectors.load_word2vec_format("../dataset/GoogleNews-vectors-negative300.bin", binary=True)
        wv_matrix = []
        for i in range(len(vocab)-1):
            word = vocab[i]
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

        # one for UNK and one for zero padding
        # wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        return wv_matrix


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Text_CNN_CBAM(nn.Module):

    def __init__(self, args):
        super(Text_CNN_CBAM, self).__init__()
        self.args = args

        self.vocab_num = args.vocab_num
        self.embed_dim = args.embed_dim
        self.kernel_num = args.kernel_num
        self.kernel_sizes = args.kernel_sizes
        self.class_num = args.class_num

        self.embed = nn.Embedding(self.vocab_num, self.embed_dim, padding_idx=args.vocab_num-1)
        self.embed_drop = nn.Dropout(0.5)  # 0.1-0.2
        self.conv = nn.Conv2d(1, self.kernel_num, (3, self.embed_dim))
        self.bn2 = nn.BatchNorm2d(self.kernel_num)

        self.ca = ChannelAttention(self.kernel_num)
        self.sa = SpatialAttention()

        self.fc_drop = nn.Dropout(0.5)  # 0.5
        self.fc1 = nn.Linear(len(self.kernel_sizes) * self.kernel_num, self.class_num)

    def forward(self, word_index):

        word_embedding = self.embed(word_index)  # N*L*D
        word_embedding = self.embed_drop(word_embedding)
        word_embedding = word_embedding.unsqueeze(1)  # N*1*L*D

        conv_1 = F.relu(self.bn2(self.conv(word_embedding)))

        conv_1 = self.ca(conv_1) * conv_1
        conv_1 = self.sa(conv_1) * conv_1

        pools_1 = [F.max_pool2d(conv_1, (conv_1.size(2), 1)).squeeze(3).squeeze(2)]   # N*K
        concat_1 = torch.cat(pools_1, 1)  # N*K
        concat_1 = self.fc_drop(concat_1)  # N*K
        logit = self.fc1(concat_1)

        return logit

class Text_CNN(nn.Module):

    def __init__(self, args):
        super(Text_CNN, self).__init__()
        self.args = args

        self.vocab_num = args.vocab_num
        self.embed_dim = args.embed_dim
        self.kernel_num = args.kernel_num
        self.kernel_sizes = args.kernel_sizes
        self.class_num = args.class_num

        self.embed = nn.Embedding(self.vocab_num, self.embed_dim, padding_idx=args.vocab_num-1)
        self.embed_drop = nn.Dropout(0.5)  # 0.1-0.2
        self.convs = nn.ModuleList([nn.Conv2d(1, self.kernel_num, (K, self.embed_dim)) for K in self.kernel_sizes])
        self.bn2s = nn.ModuleList([nn.BatchNorm2d(self.kernel_num) for K in self.kernel_sizes])

        self.fc_drop = nn.Dropout(0.5)  # 0.5
        self.fc1 = nn.Linear(len(self.kernel_sizes) * self.kernel_num, self.class_num)

    def forward(self, word_index, cluster):

        word_embedding = self.embed(word_index)  # N*L*D
        word_embedding = self.embed_drop(word_embedding)
        word_embedding = word_embedding.unsqueeze(1)  # N*1*L*D

        convs_1 = []
        for i in range(len(self.kernel_sizes)):
            conv_1 = F.relu(self.bn2s[i](self.convs[i](word_embedding)))   # N*K*L*1
            convs_1.append(conv_1)

        pools_1 = [F.max_pool2d(conv_1, (conv_1.size(2), 1)).squeeze(3).squeeze(2) for conv_1 in convs_1]  # N*K
        concat_1 = torch.cat(pools_1, 1)  # N*K
        concat_1 = self.fc_drop(concat_1)  # N*K
        logit = self.fc1(concat_1)

        return logit

class Text_CNN_MTL(nn.Module):

    def __init__(self, args):
        super(Text_CNN_MTL, self).__init__()
        self.args = args

        self.vocab_num = args.vocab_num
        self.embed_dim = args.embed_dim
        self.kernel_num = args.kernel_num
        self.kernel_sizes = args.kernel_sizes
        self.class_num = args.class_num

        self.embed = nn.Embedding(self.vocab_num, self.embed_dim, padding_idx=args.vocab_num-1)
        self.embed_drop = nn.Dropout(0.5)  # 0.1-0.2
        self.convs = nn.ModuleList([nn.Conv2d(1, self.kernel_num, (K, self.embed_dim)) for K in self.kernel_sizes])
        self.bn2s = nn.ModuleList([nn.BatchNorm2d(self.kernel_num) for K in self.kernel_sizes])

        self.fc_drop = nn.Dropout(0.5)  # 0.5
        self.fc1 = nn.Linear(len(self.kernel_sizes) * self.kernel_num, self.class_num)

    def forward(self, word_index, cluster):

        word_embedding = self.embed(word_index)  # N*L*D
        word_embedding = self.embed_drop(word_embedding)
        word_embedding = word_embedding.unsqueeze(1)  # N*1*L*D

        convs_1 = []
        for i in range(len(self.kernel_sizes)):
            conv_1 = F.relu(self.bn2s[i](self.convs[i](word_embedding)))   # N*K*L*1
            convs_1.append(conv_1)

        pools_1 = [F.max_pool2d(conv_1, (conv_1.size(2), 1)).squeeze(3).squeeze(2) for conv_1 in convs_1]  # N*K
        concat_1 = torch.cat(pools_1, 1)  # N*K
        concat_1 = self.fc_drop(concat_1)  # N*K
        logit = self.fc1(concat_1)

        return logit


class WV_Text_CNN(nn.Module):

    def __init__(self, args):
        super(WV_Text_CNN, self).__init__()
        self.args = args

        self.vocab_num = args.vocab_num
        self.embed_dim = args.embed_dim
        self.kernel_num = args.kernel_num
        self.kernel_sizes = args.kernel_sizes
        self.class_num = args.class_num
        self.index2word = args.index2word

        self.wv_matrix = self.load_vector(self.index2word)
        self.embed = nn.Embedding(self.vocab_num, self.embed_dim, padding_idx=args.vocab_num - 1)
        self.embed.weight.data.copy_(torch.from_numpy(self.wv_matrix))

        self.embed_drop = nn.Dropout(0.5)  # 0.1-0.2
        self.convs = nn.ModuleList([nn.Conv2d(1, self.kernel_num, (K, self.embed_dim)) for K in self.kernel_sizes])
        self.bn2s = nn.ModuleList([nn.BatchNorm2d(self.kernel_num) for K in self.kernel_sizes])

        self.fc_drop = nn.Dropout(0.5)  # 0.5
        self.fc1 = nn.Linear(len(self.kernel_sizes) * self.kernel_num, self.class_num)

    def load_vector(self, vocab):
        word_vectors = KeyedVectors.load_word2vec_format("dataset/GoogleNews-vectors-negative300.bin", binary=True)
        wv_matrix = []
        for i in range(len(vocab) - 1):
            word = vocab[i]
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

        # one for UNK and one for zero padding
        # wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        return wv_matrix

    def forward(self, word_index):

        word_embedding = self.embed(word_index)  # N*L*D
        word_embedding = self.embed_drop(word_embedding)
        word_embedding = word_embedding.unsqueeze(1)  # N*1*L*D

        convs_1 = []
        for i in range(len(self.kernel_sizes)):
            conv_1 = F.relu(self.bn2s[i](self.convs[i](word_embedding)))   # N*K*L*1
            convs_1.append(conv_1)

        pools_1 = [F.max_pool2d(conv_1, (conv_1.size(2), 1)).squeeze(3).squeeze(2) for conv_1 in convs_1]  # N*K
        concat_1 = torch.cat(pools_1, 1)  # N*K
        concat_1 = self.fc_drop(concat_1)  # N*K
        logit = self.fc1(concat_1)

        return logit


class AdaFM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AdaFM, self).__init__()

        self.style_gama = nn.Parameter(torch.ones(in_channel, out_channel, 1, 1))
        self.style_beta = nn.Parameter(torch.zeros(in_channel, out_channel, 1, 1))

    def forward(self, input, W, b, bi):
        W_i = W * self.style_gama + self.style_beta
        out = F_conv(input, W_i, bias=b + bi, stride=1, padding=1)
        return out

F_conv = torch.nn.functional.conv2d

class Text_CNN_Style(nn.Module):

    def __init__(self, args):
        super(Text_CNN_Style, self).__init__()
        self.args = args

        self.vocab_num = args.vocab_num
        self.embed_dim = args.embed_dim
        self.kernel_num = args.kernel_num
        self.kernel_sizes = args.kernel_sizes
        self.class_num = args.class_num

        self.embed = nn.Embedding(self.vocab_num, self.embed_dim, padding_idx=args.vocab_num-1)
        self.embed_drop = nn.Dropout(0.5)  # 0.1-0.2
        self.convs = nn.ModuleList([nn.Conv2d(1, self.kernel_num, (K, self.embed_dim)) for K in self.kernel_sizes])
        self.bn2s = nn.ModuleList([nn.BatchNorm2d(self.kernel_num) for K in self.kernel_sizes])

        self.AdaFM_class_bias = nn.Embedding(20, self.kernel_num)

        self.fc_drop = nn.Dropout(0.5)  # 0.5
        self.fc1 = nn.Linear(len(self.kernel_sizes) * self.kernel_num, self.class_num)



    def forward(self, word_index, cluster):

        word_embedding = self.embed(word_index)  # N*L*D
        word_embedding = self.embed_drop(word_embedding)
        word_embedding = word_embedding.unsqueeze(1)  # N*1*L*D

        convs_1 = []
        for i in range(len(self.kernel_sizes)):
            W_conv = self.convs[i].weight
            b_conv = self.convs[i].bias
            bi = self.AdaFM_class_bias(cluster).squeeze(0)
            b = b_conv + bi
            conv_1 = F_conv(word_embedding, W_conv, bias=b, stride=1)
            conv_1 = F.relu(self.bn2s[i](conv_1))   # N*K*L*1
            convs_1.append(conv_1)

        pools_1 = [F.max_pool2d(conv_1, (conv_1.size(2), 1)).squeeze(3).squeeze(2) for conv_1 in convs_1]  # N*K
        concat_1 = torch.cat(pools_1, 1)  # N*K
        concat_1 = self.fc_drop(concat_1)  # N*K
        logit = self.fc1(concat_1)

        return logit

class Text_LSTM(nn.Module):
    def __init__(self, args):
        super(Text_LSTM, self).__init__()
        self.args = args

        self.vocab_num = args.vocab_num
        self.embed_dim = args.embed_dim
        self.kernel_num = args.kernel_num
        self.kernel_sizes = args.kernel_sizes
        self.class_num = args.class_num
        self.index2word = args.index2word

        self.wv_matrix = self.load_vector(self.index2word)
        self.embed = nn.Embedding(self.vocab_num, self.embed_dim, padding_idx=args.vocab_num-1)
        self.embed.weight.data.copy_(torch.from_numpy(self.wv_matrix))

        self.embed_drop = nn.Dropout(0.5)  # 0.1-0.2
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.kernel_num, num_layers=1, batch_first=True, dropout=0)
        self.fc_drop = nn.Dropout(0.5)  # 0.5
        self.fc1 = nn.Linear(len(self.kernel_sizes) * self.kernel_num, self.class_num)

    def forward(self, word_index, cluster):

        word_embedding = self.embed(word_index)  # N*L*D
        word_embedding = self.embed_drop(word_embedding)
        word_embedding = word_embedding # N*L*D

        hidden_units, (_, _) = self.lstm(word_embedding)

        pool = F.max_pool1d(hidden_units.transpose(1, 2), (hidden_units.size(1))).squeeze(2) # N*K
        pool = self.fc_drop(pool)
        logit = self.fc1(pool)

        return logit

    def load_vector(self, vocab):
        word_vectors = KeyedVectors.load_word2vec_format("../dataset/GoogleNews-vectors-negative300.bin", binary=True)
        wv_matrix = []
        for i in range(len(vocab)-1):
            word = vocab[i]
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

        # one for UNK and one for zero padding
        # wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        return wv_matrix


class Text_LSTM_Style(nn.Module):
    def __init__(self, args):
        super(Text_LSTM_Style, self).__init__()
        self.args = args

        self.vocab_num = args.vocab_num
        self.embed_dim = args.embed_dim
        self.kernel_num = args.kernel_num
        self.kernel_sizes = args.kernel_sizes
        self.class_num = args.class_num
        self.index2word = args.index2word

        # self.wv_matrix = self.load_vector(self.index2word)
        self.embed = nn.Embedding(self.vocab_num, self.embed_dim, padding_idx=args.vocab_num-1)
        # self.embed.weight.data.copy_(torch.from_numpy(self.wv_matrix))
        self.domain_embed = nn.Embedding(14, self.embed_dim)
        self.embed_drop = nn.Dropout(0.5)  # 0.1-0.2
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.kernel_num, num_layers=2,
                            batch_first=True, dropout=0, bidirectional=True)
        self.fc_drop = nn.Dropout(0.5)  # 0.5
        self.fc1 = nn.Linear(len(self.kernel_sizes) * self.kernel_num, self.class_num)

    def forward(self, word_index, cluster):

        word_embedding = self.embed(word_index)  # N*L*D
        domain_embedding = self.domain_embed(cluster).unsqueeze(1)
        word_embedding = word_embedding + domain_embedding
        word_embedding = self.embed_drop(word_embedding)
        word_embedding = word_embedding # N*L*D

        hidden_units, (_, _) = self.lstm(word_embedding)

        pool = F.max_pool1d(hidden_units.transpose(1, 2), (hidden_units.size(1))).squeeze(2) # N*K
        pool = self.fc_drop(pool)
        logit = self.fc1(pool)

        return logit

    def load_vector(self, vocab):
        word_vectors = KeyedVectors.load_word2vec_format("../dataset/GoogleNews-vectors-negative300.bin", binary=True)
        wv_matrix = []
        for i in range(len(vocab)-1):
            word = vocab[i]
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

        # one for UNK and one for zero padding
        # wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        return wv_matrix



class Text_MLP(nn.Module):
    def __init__(self, args):
        super(Text_MLP, self).__init__()
        self.args = args

        self.vocab_num = args.vocab_num
        self.embed_dim = args.embed_dim
        self.kernel_num = args.kernel_num
        self.kernel_sizes = args.kernel_sizes
        self.class_num = args.class_num

        self.fc1 = nn.Linear(args.vocab_num, self.kernel_num)
        self.fc2 = nn.Linear(len(self.kernel_sizes) * self.kernel_num, self.class_num)

    def forward(self, doc_bow):
        hidden = F.dropout(F.relu(self.fc1(doc_bow.float())), 0.1, training=self.training)
        logit = self.fc2(hidden)
        return logit


class Text_CNN_att(nn.Module):

    def __init__(self, args):
        super(Text_CNN_att, self).__init__()
        self.args = args

        self.vocab_num = args.vocab_num
        self.embed_dim = args.embed_dim
        self.kernel_num = args.kernel_num
        self.kernel_sizes = args.kernel_sizes
        self.class_num = args.class_num

        self.embed = nn.Embedding(self.vocab_num, self.embed_dim, padding_idx=args.vocab_num-1)
        self.embed_drop = nn.Dropout(0.5)  # 0.1-0.2
        self.conv_value = nn.Conv2d(1, self.kernel_num, (3, self.embed_dim))  # kernel_size = 3
        self.conv_query = nn.Conv2d(1, 64, (3, self.embed_dim))
        self.conv_key = nn.Parameter(torch.rand(64, 1))
        self.conv_key.data.uniform_(-0.25, 0.25)

        self.bn_value = nn.BatchNorm2d(self.kernel_num)
        self.bn_query = nn.BatchNorm2d(64)

        self.fc_drop = nn.Dropout(0.5)  # 0.5
        self.fc1 = nn.Linear(self.kernel_num, self.class_num)

    def forward(self, word_index, cluster):

        word_embedding = self.embed(word_index)  # N*L*D
        word_embedding = self.embed_drop(word_embedding)
        word_embedding = word_embedding.unsqueeze(1)  # N*1*L*D

        conv_value = F.relu(self.bn_value(self.conv_value(word_embedding))).squeeze(3)  # N*K*L
        conv_query = F.relu(self.bn_query(self.conv_query(word_embedding))).squeeze(3)  # N*64*L

        pool_1 = self.att_pool(conv_value, conv_query, self.conv_key)
        concat_1 = self.fc_drop(pool_1)  # N*K
        logit = self.fc1(concat_1)

        return logit

    def att_pool(self, value, query, key):

        query = query.transpose(1, 2)   # N*L*64
        att_matrix = torch.matmul(query, key)  # N*L*1
        att_matrix = F.softmax(att_matrix, dim=1)  # N*L*1
        pool = torch.matmul(value, att_matrix).squeeze(2)  # N*K

        return pool


import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model,  max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * - (math.log(1e4) / d_model))
        pe[:, ::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x

class DPCNN(nn.Module):
    def __init__(self, args):
        super(DPCNN, self).__init__()
        self.args = args

        self.vocab_num = args.vocab_num
        self.embed_dim = args.embed_dim
        self.kernel_num = args.kernel_num
        self.query_kernel_num = args.query_kernel_num
        self.kernel_sizes = args.kernel_sizes
        self.layer_num = args.layer_num
        self.class_num = args.class_num
        self.dropout = args.dropout

        self.embed = nn.Embedding(self.vocab_num, self.embed_dim, padding_idx=args.vocab_num-1)
        self.embed_drop = nn.Dropout(0.5)  # 0.1-0.2
        self.conv_embedding = nn.Conv2d(1, self.kernel_num, (3, self.embed_dim), stride=1)

        # DPCNN
        self.conv_1 = nn.Conv2d(self.kernel_num, self.kernel_num, (3, 1), stride=1)
        self.classifier = nn.Linear(self.kernel_num, args.class_num)

    def forward(self, word_index):
        # word embedding
        word_embedding = self.embed(word_index)  # N*L*D
        word_embedding = self.embed_drop(word_embedding)
        word_embedding = word_embedding.unsqueeze(1)  # N*1*L*D

        conv_1 = F.relu(self.conv_embedding(word_embedding))
        conv_2 = self.block(conv_1)

        doc_embed = F.max_pool2d(conv_2, (conv_2.size(2), 1)).squeeze(3).squeeze(2)  # N*K
        doc_embed = F.dropout(doc_embed, self.dropout, training=self.training)
        logit = self.classifier(doc_embed)

        return logit

    def block(self, x):
        # Pooling
        x = F.pad(x, (0, 0, 0, 1), mode='constant', value=0)
        px = F.max_pool2d(x, (3, 1), 2)

        # Convolution
        x = F.pad(px, (0, 0, 1, 1), mode='constant', value=0)
        x = F.relu(x)
        x = self.conv_1(x)
        # x = bn(x)

        x = F.pad(x, (0, 0, 1, 1), mode='constant', value=0)
        x = F.relu(x)
        x = self.conv_1(x)

        # Short Cut
        x = x + px

        return x

class DPCNN_CBAM(nn.Module):
    def __init__(self, args):
        super(DPCNN_CBAM, self).__init__()
        self.args = args

        self.vocab_num = args.vocab_num
        self.embed_dim = args.embed_dim
        self.kernel_num = args.kernel_num
        self.query_kernel_num = args.query_kernel_num
        self.kernel_sizes = args.kernel_sizes
        self.layer_num = args.layer_num
        self.class_num = args.class_num
        self.dropout = args.dropout

        self.embed = nn.Embedding(self.vocab_num, self.embed_dim, padding_idx=args.vocab_num-1)
        self.embed_drop = nn.Dropout(0.5)  # 0.1-0.2
        self.conv_embedding = nn.Conv2d(1, self.kernel_num, (3, self.embed_dim), stride=1)

        # DPCNN
        self.conv_1 = nn.Conv2d(self.kernel_num, self.kernel_num, (3, 1), stride=1)
        self.classifier = nn.Linear(self.kernel_num, args.class_num)

    def forward(self, word_index):
        # word embedding
        word_embedding = self.embed(word_index)  # N*L*D
        word_embedding = self.embed_drop(word_embedding)
        word_embedding = word_embedding.unsqueeze(1)  # N*1*L*D

        conv_1 = F.relu(self.conv_embedding(word_embedding))
        conv_2 = self.block(conv_1)

        doc_embed = F.max_pool2d(conv_2, (conv_2.size(2), 1)).squeeze(3).squeeze(2)  # N*K

        doc_embed = F.dropout(doc_embed, self.dropout, training=self.training)
        logit = self.classifier(doc_embed)

        return logit

    def block(self, x):
        # Pooling
        x = F.pad(x, (0, 0, 0, 1), mode='constant', value=0)
        px = F.max_pool2d(x, (3, 1), 2)
        # px = x

        # Convolution
        x = F.pad(px, (0, 0, 1, 1), mode='constant', value=0)
        x = F.relu(x)
        x = self.conv_1(x)
        # x = bn(x)

        x = F.pad(x, (0, 0, 1, 1), mode='constant', value=0)
        x = F.relu(x)
        x = self.conv_1(x)

        # Short Cut
        x = x + px

        return x



class DPCNN_att(nn.Module):

    def __init__(self, args):
        super(DPCNN_att, self).__init__()
        self.args = args

        self.vocab_num = args.vocab_num
        self.embed_dim = args.embed_dim
        self.kernel_num = args.kernel_num
        self.query_kernel_num = args.query_kernel_num
        self.kernel_sizes = args.kernel_sizes
        self.layer_num = args.layer_num
        self.class_num = args.class_num
        self.dropout = args.dropout

        self.embed = nn.Embedding(self.vocab_num, self.embed_dim, padding_idx=args.vocab_num-1)
        self.embed_drop = nn.Dropout(0.5)  # 0.1-0.2

       # conv_1
        self.conv_value_1 = nn.Conv2d(1, self.kernel_num, (3, self.embed_dim), stride=1)
        self.conv_query_1 = nn.Conv2d(1, 32, (3, self.embed_dim), stride=1)
        self.conv_key_1 = nn.Parameter(torch.rand(32, 1))
        self.conv_key_1.data.uniform_(-0.25, 0.25)

        # DPCNN
        self.conv_value_2 = nn.Conv2d(self.kernel_num, self.kernel_num, (3, 1), stride=1)
        self.conv_query_2 = nn.Conv2d(32, 32, (3, 1), stride=1)
        self.conv_key_2 = nn.Parameter(torch.rand(32, 1))
        self.conv_key_2.data.uniform_(-0.25, 0.25)

        self.classifier = nn.Linear(self.kernel_num * 2, args.class_num)

    def forward(self, word_index):
        # word embedding
        word_embedding = self.embed(word_index)  # N*L*D
        word_embedding = self.embed_drop(word_embedding)
        word_embedding = word_embedding.unsqueeze(1)  # N*1*L*D

        value_1 = F.relu(self.conv_value_1(word_embedding))
        query_1 = F.relu(self.conv_query_1(word_embedding))
        doc_embed_1 = self.att_pool(value_1, query_1, self.conv_key_1)  # [batch_size, channel_size]

        value_2 = self.value_block(value_1)
        query_2 = self.query_block(query_1)
        doc_embed_2 = self.att_pool(value_2, query_2, self.conv_key_2)  # [batch_size, channel_size]

        doc_embed = torch.cat([doc_embed_1, doc_embed_2], 1)
        doc_embed = F.dropout(doc_embed, self.dropout, training=self.training)
        logit = self.classifier(doc_embed)

        return logit

    def att_pool(self, value, query, key):
        value = value.squeeze(3)  # N*K*L
        query = query.squeeze(3)  # N*64*L
        query = query.transpose(1, 2)  # N*L*64
        att_matrix = torch.matmul(query, key)  # N*L*1
        att_matrix = F.softmax(att_matrix, dim=1)  # N*L*1
        pool = torch.matmul(value, att_matrix).squeeze(2)  # N*K
        return pool

    def value_block(self, x):
        # Pooling
        x = F.pad(x, (0, 0, 0, 1), mode='constant', value=0)
        px = F.max_pool2d(x, (3, 1), 2)

        # Convolution
        x = F.pad(px, (0, 0, 1, 1), mode='constant', value=0)
        x = F.relu(x)
        x = self.conv_value_2(x)
        # x = bn(x)

        x = F.pad(x, (0, 0, 1, 1), mode='constant', value=0)
        x = F.relu(x)
        x = self.conv_value_2(x)

        # Short Cut
        x = x + px

        return x

    def query_block(self, x):
        # Pooling
        x = F.pad(x, (0, 0, 0, 1), mode='constant', value=0)
        px = F.max_pool2d(x, (3, 1), 2)

        # Convolution
        x = F.pad(px, (0, 0, 1, 1), mode='constant', value=0)
        x = F.relu(x)
        x = self.conv_query_2(x)
        # x = bn(x)

        x = F.pad(x, (0, 0, 1, 1), mode='constant', value=0)
        x = F.relu(x)
        x = self.conv_query_2(x)

        # Short Cut
        x = x + px

        return x


class DPCNN_multi(nn.Module):
    def __init__(self, args):
        super(DPCNN_multi, self).__init__()
        self.args = args

        self.vocab_num = args.vocab_num
        self.embed_dim = args.embed_dim
        self.kernel_num = args.kernel_num
        self.query_kernel_num = args.query_kernel_num
        self.kernel_sizes = args.kernel_sizes
        self.layer_num = args.layer_num
        self.class_num = args.class_num
        self.dropout = args.dropout

        self.embed = nn.Embedding(self.vocab_num, self.embed_dim, padding_idx=args.vocab_num-1)
        self.embed_drop = nn.Dropout(0.5)  # 0.1-0.2
        self.conv_embedding = nn.Conv2d(1, self.kernel_num, (3, self.embed_dim), stride=1)

        # DPCNN

        self.convs = nn.ModuleList([nn.Conv2d(self.kernel_num, self.kernel_num, (3, 1)) for K in range(self.layer_num -1)])
        self.classifier = nn.Linear(self.kernel_num, args.class_num)

    def forward(self, word_index):
        # word embedding
        word_embedding = self.embed(word_index)  # N*L*D
        word_embedding = self.embed_drop(word_embedding)
        word_embedding = word_embedding.unsqueeze(1)  # N*1*L*D

        conv_1 = F.relu(self.conv_embedding(word_embedding))

        for conv in self.convs:
            conv_1 = self.block(conv_1, conv)

        doc_embed = F.max_pool2d(conv_1, (conv_1.size(2), 1)).squeeze(3).squeeze(2)  # N*K
        doc_embed = F.dropout(doc_embed, self.dropout, training=self.training)
        logit = self.classifier(doc_embed)

        return logit

    def block(self, x, block_conv):
        # Pooling
        x = F.pad(x, (0, 0, 0, 1), mode='constant', value=0)
        px = F.max_pool2d(x, (3, 1), 2)

        # Convolution
        x = F.pad(px, (0, 0, 1, 1), mode='constant', value=0)
        x = F.relu(x)
        x = block_conv(x)
        # x = bn(x)

        x = F.pad(x, (0, 0, 1, 1), mode='constant', value=0)
        x = F.relu(x)
        x = block_conv(x)

        # Short Cut
        x = x + px

        return x


# class DPCNN_att(nn.Module):
#
#     def __init__(self, args):
#         super(DPCNN_att, self).__init__()
#         self.args = args
#
#         self.vocab_num = args.vocab_num
#         self.embed_dim = args.embed_dim
#         self.kernel_num = args.kernel_num
#         self.query_kernel_num = args.query_kernel_num
#         self.kernel_sizes = args.kernel_sizes
#         self.layer_num = args.layer_num
#         self.class_num = args.class_num
#
#         self.embed = nn.Embedding(self.vocab_num, self.embed_dim, padding_idx=args.vocab_num-1)
#         self.embed_drop = nn.Dropout(0.5)  # 0.1-0.2
#
#         self.conv_value_embedding = nn.Conv2d(1, self.kernel_num, (3, self.embed_dim), stride=1)
#         self.conv_query_embedding = nn.Conv2d(self.kernel_num, 32, (3, 1), stride=1)
#
#         # DPCNN
#         self.conv_value_1 = nn.Conv2d(self.kernel_num, self.kernel_num, (3, 1), stride=1)
#         self.conv_query_1 = nn.Conv2d(32, 32, (3, 1), stride=1)
#
#         self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
#         self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
#         self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
#
#         # word han attenion
#         self.key1 = nn.Parameter(torch.rand(64, 1))
#         self.key1.data.uniform_(-0.25, 0.25)
#
#         self.key2 = nn.Parameter(torch.rand(32, 1))
#         self.key2.data.uniform_(-0.25, 0.25)
#
#         self.dropout = args.dropout
#         self.classifier = nn.Linear(self.kernel_num, args.class_num)
#
#     def forward(self, word_index):
#         # word embedding
#         word_embedding = self.embed(word_index)  # N*L*D
#         word_embedding = self.embed_drop(word_embedding)
#         word_embedding = word_embedding.unsqueeze(1)  # N*1*L*D
#
#         value = F.relu(self.conv_value_embedding(word_embedding))
#         query = F.relu(self.conv_query_embedding(self.padding_conv(value)))
#         # doc_embed_1 = self.att_pool(value, query, self.key1)
#
#         value = self.value_block(value)
#         query = self.query_block(query)
#         doc_embed = self.att_pool(value, query, self.key2)  # [batch_size, channel_size]
#
#         # doc_embed = torch.cat((doc_embed_1, doc_embed_2), 1)
#         doc_embed = F.dropout(doc_embed, self.dropout, training=self.training)
#         logit = self.classifier(doc_embed)
#
#         return logit
#
#     def att_pool(self, value, query, key):
#         value = value.squeeze(3)  # N*K*L
#         query = query.squeeze(3)  # N*64*L
#         query = query.transpose(1, 2)  # N*L*64
#         att_matrix = torch.matmul(query, key)  # N*L*1
#         att_matrix = F.softmax(att_matrix, dim=1)  # N*L*1
#         pool = torch.matmul(value, att_matrix).squeeze(2)  # N*K
#         return pool
#
#     def block(self, value, query, conv_value, conv_query, conv_key):
#         value = self.res_block(value, conv_value)
#         query = self.res_block(query, conv_query)
#         pool = self.att_pool(value, query, conv_key)
#
#         return value, query, pool
#
#     def res_block(self, x, conv):
#         # Pooling
#         x = self.padding_pool(x)
#         px = self.pooling(x)
#
#         # Convolution
#         x = self.padding_conv(px)
#         x = F.relu(x)
#         x = conv(x)
#         # x = bn(x)
#
#         x = self.padding_conv(x)
#         x = F.relu(x)
#         x = conv(x)
#
#         # Short Cut
#         x = x + px
#
#         return x
#
#
#     def value_block(self, x):
#         # Pooling
#         x = self.padding_pool(x)
#         px = self.pooling(x)
#
#         # Convolution
#         x = self.padding_conv(px)
#         x = F.relu(x)
#         x = self.conv_value_1(x)
#
#         x = self.padding_conv(x)
#         x = F.relu(x)
#         x = self.conv_value_1(x)
#
#         # Short Cut
#         x = x + px
#
#         return x
#
#     def query_block(self, x):
#         # Pooling
#         x = self.padding_pool(x)
#         px = self.pooling(x)
#
#         # Convolution
#         x = self.padding_conv(px)
#         x = F.relu(x)
#         x = self.conv_query_1(x)
#
#         x = self.padding_conv(x)
#         x = F.relu(x)
#         x = self.conv_query_1(x)
#
#         # Short Cut
#         x = x + px
#
#         return x

# class DPCNN(nn.Module):
#     def __init__(self, args):
#         super(DPCNN, self).__init__()
#         self.args = args
#
#         self.vocab_num = args.vocab_num
#         self.embed_dim = args.embed_dim
#         self.kernel_num = args.kernel_num
#         self.query_kernel_num = args.query_kernel_num
#         self.kernel_sizes = args.kernel_sizes
#         self.layer_num = args.layer_num
#         self.class_num = args.class_num
#
#         self.embed = nn.Embedding(self.vocab_num, self.embed_dim, padding_idx=args.vocab_num-1)
#         self.embed_drop = nn.Dropout(0.1)  # 0.1-0.2
#         self.conv_embedding = nn.Conv2d(1, self.kernel_num, (3, self.embed_dim), stride=1)
#
#         # DPCNN
#         self.conv_1 = nn.Conv2d(self.kernel_num, self.kernel_num, (3, 1), stride=1)
#         self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
#         self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
#         self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
#
#         self.dropout = args.dropout
#         self.classifier = nn.Linear(self.kernel_num, args.class_num)
#
#     def forward(self, word_index):
#         # word embedding
#         word_embedding = self.embed(word_index)  # N*L*D
#         word_embedding = self.embed_drop(word_embedding)
#         word_embedding = word_embedding.unsqueeze(1)  # N*1*L*D
#
#         conv_1 = F.relu(self.conv_embedding(word_embedding))
#         conv_2 = self.block(conv_1)
#
#         doc_embed = F.max_pool2d(conv_2, (conv_2.size(2), 1)).squeeze(3).squeeze(2) # N*K
#         doc_embed = F.dropout(doc_embed, self.dropout, training=self.training)
#         logit = self.classifier(doc_embed)
#
#         return logit
#
#     def block(self, x):
#         # Pooling
#         x = self.padding_pool(x)
#         px = self.pooling(x)
#
#         # Convolution
#         x = self.padding_conv(px)
#         x = F.relu(x)
#         x = self.conv_1(x)
#         # x = bn(x)
#
#         x = self.padding_conv(x)
#         x = F.relu(x)
#         x = self.conv_1(x)
#
#         # Short Cut
#         x = x + px
#
#         return x



# class DPCNN_att(nn.Module):
#
#     def __init__(self, args):
#         super(DPCNN_att, self).__init__()
#         self.args = args
#
#         self.vocab_num = args.vocab_num
#         self.embed_dim = args.embed_dim
#         self.kernel_num = args.kernel_num
#         self.query_kernel_num = args.query_kernel_num
#         self.kernel_sizes = args.kernel_sizes
#         self.layer_num = args.layer_num
#         self.class_num = args.class_num
#
#         # Position embdedding
#         self.PositionalEmbedding = args.PositionalEmbedding
#         self.PositionalEncoding = PositionalEncoding(self.embed_dim)
#         self.embed = nn.Embedding(self.vocab_num, self.embed_dim, padding_idx=args.vocab_num-1)
#         self.embed_drop = nn.Dropout(0.5)  # 0.1-0.2
#
#         # Region Conv
#         self.region_conv = nn.Conv2d(1, self.kernel_num, (3, self.embed_dim))
#         self.bn_region = nn.BatchNorm2d(self.kernel_num)
#         # self.conv_value = nn.Conv2d(1, self.kernel_num, (3, self.embed_dim))  # kernel_size = 3
#         # self.conv_query = nn.Conv2d(1, self.query_kernel_num, (3, self.embed_dim))
#         # self.conv_key = nn.Parameter(torch.rand(self.query_kernel_num, 1))
#         # self.conv_key.data.uniform_(-0.25, 0.25)
#         # self.conv_key = nn.Linear(self.query_kernel_num, 1, bias=False)
#         # self.conv_key.weight.uniform_(-0.25, 0.25)
#
#         # self.bn_value = nn.BatchNorm2d(self.kernel_num)
#         # self.bn_query = nn.BatchNorm2d(self.query_kernel_num)
#
#         # Dpcnn block
#         self.conv_values = nn.Conv2d(self.kernel_num, self.kernel_num, (3, 1))
#         self.conv_querys = nn.Conv2d(self.query_kernel_num, self.query_kernel_num, (3, 1))
#         self.conv_keys = nn.Parameter(torch.rand(self.query_kernel_num, 1))
#         self.conv_keys.data.uniform_(-0.25, 0.25)
#
#         # self.conv_keys = nn.ModuleList([nn.Linear(self.query_kernel_num, 1, bias=False)])
#         # self.conv_keys = nn.ModuleList(conv_key.weight.uniform_(-0.25, 0.25) for conv_key in self.conv_keys)
#
#         self.bn_values = nn.ModuleList([nn.BatchNorm2d(self.kernel_num) for K in range(self.layer_num)])
#         self.bn_querys = nn.ModuleList([nn.BatchNorm2d(self.query_kernel_num) for K in range(self.layer_num)])
#
#         self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
#         self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
#         self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
#
#         self.fc_drop = nn.Dropout(0.5)  # 0.5
#         self.fc1 = nn.Linear(self.kernel_num * (self.layer_num), self.class_num)
#
#     def forward(self, word_index):
#
#         word_embedding = self.embed(word_index)             # N*L*D
#         if self.PositionalEmbedding:
#            word_embedding = self.PositionalEncoding(word_embedding)
#
#         word_embedding = self.embed_drop(word_embedding)
#         word_embedding = word_embedding.unsqueeze(1)        # N*1*L*D
#
#         #Region cnn
#         # value = F.relu(self.bn_value(self.conv_value(word_embedding)))             # N*K*L*1
#         # query = F.relu(self.bn_query(self.conv_query(word_embedding)))             # N*64*L*1
#         # pool = self.att_pool(value, query, self.conv_key)
#         # pools = [pool]
#         hidden = F.relu(self.region_conv(word_embedding))
#         value = hidden
#         query = hidden
#
#         pools = []
#         #Dpcnn
#         for i in range(self.layer_num):
#             value, query, pool = self.block(value, query, self.conv_values, self.conv_querys,
#                                             self.conv_keys, self.bn_values[i], self.bn_querys[i])
#             pools.append(pool)
#
#         concat_1 = torch.cat(pools, 1)     # N*K
#         concat_1 = self.fc_drop(concat_1)  # N*K
#         logit = self.fc1(concat_1)
#
#         return logit
#
#     def att_pool(self, value, query, key):
#         value = value.squeeze(3)                   # N*K*L
#         query = query.squeeze(3)                   # N*64*L
#         query = query.transpose(1, 2)              # N*L*64
#         att_matrix = torch.matmul(query, key)      # N*L*1
#         #att_matrix = key(query)
#         att_matrix = F.softmax(att_matrix, dim=1)  # N*L*1
#         pool = torch.matmul(value, att_matrix).squeeze(2)       # N*K
#
#         return pool
#
#     def block(self, value, query, conv_value, conv_query, conv_key, bn_value, bn_query):
#
#         value = self.res_block(value, conv_value, bn_value)
#         query = self.res_block(query, conv_query, bn_query)
#         pool = self.att_pool(value, query, conv_key)
#
#         return value, query, pool
#
#     def res_block(self, x, conv, bn):
#         # Pooling
#         x = self.padding_pool(x)
#         px = self.pooling(x)
#
#         # Convolution
#         x = self.padding_conv(px)
#         x = F.relu(x)
#         x = conv(x)
#         # x = bn(x)
#
#         x = self.padding_conv(x)
#         x = F.relu(x)
#         x = conv(x)
#
#         # Short Cut
#         x = x + px
#
#         return x
#
#
