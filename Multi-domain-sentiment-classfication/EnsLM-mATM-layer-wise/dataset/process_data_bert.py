from transformers import BertTokenizer, BertModel
import pickle
import numpy as np
import torch
import logging
logging.basicConfig(level=logging.ERROR)


def precess_data(data_file, save_data_file,  Maxlength = 256):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    with open(data_file, 'rb') as f:
        imdb_data = pickle.load(open(data_file, "rb"), encoding='iso-8859-1')

    train_word_list = imdb_data['train_sentence']
    test_word_list = imdb_data['test_sentence']

    maxlength = 0
    train_sen_index_list = []
    for sen in train_word_list:
        str = ' '
        sen = str.join(sen)
        sen_index = tokenizer.encode(sen, return_tensors='pt', max_length=Maxlength).numpy()
        train_sen_index_list.append(sen_index)
        maxlength = max(sen_index.shape[1], maxlength)


    if maxlength > Maxlength:
        maxlength = Maxlength

    train_sen_index = np.zeros([len(train_sen_index_list), maxlength])

    for num in range(len(train_sen_index_list)):
        sen_index = train_sen_index_list[num]
        length = sen_index.shape[1]
        if length > Maxlength:
            train_sen_index[num, :] = sen_index[0, 0:Maxlength]
        else:
            train_sen_index[num, 0:length] = sen_index[0, :]

    test_sen_index_list = []
    for sen in test_word_list:
        str = ' '
        sen = str.join(sen)
        sen_index = tokenizer.encode(sen, return_tensors='pt', max_length=Maxlength).numpy()
        test_sen_index_list.append(sen_index)


    test_sen_index = np.zeros([len(test_sen_index_list), maxlength])

    for num in range(len(test_sen_index_list)):
        sen_index = test_sen_index_list[num]
        length = sen_index.shape[1]
        if length > Maxlength:
            test_sen_index[num, :] = sen_index[0, 0:Maxlength]
        else:
            test_sen_index[num, 0:length] = sen_index[0, :]

    imdb_data['train_doc_index'] = train_sen_index
    imdb_data['test_doc_index'] = test_sen_index

    pickle.dump(imdb_data, open(save_data_file, 'wb'))

if __name__ == '__main__':
    Maxlength = 256
    # data_file = 'ag_news_csv/ag_news.pkl'
    # save_data_file = 'ag_news_csv/ag_news_bert.pkl'
    # data_file = 'dbpedia_csv/dbpedia.pkl'
    # save_data_file = 'dbpedia_csv/dbpedia_bert.pkl'
    # data_file = 'sogou_news_csv/sogou_news.pkl'
    # save_data_file = 'sogou_news_csv/sogou_news_bert.pkl'

    data_file = 'dataset/mtl_dataset.pkl'
    save_data_file = 'dataset/mtl_bert_dataset.pkl'
    precess_data(data_file, save_data_file, Maxlength)
