import os

# Let keras model to run on cpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras.models
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, CuDNNLSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import tensorflow as tf
import jieba as jb
import re
import pickle
import os.path as osp

from data_process import *


def onehot_encoder(labels, label_map=None):
    label_set = set()
    for lab in labels:
        label_set.add(lab)

    label_id = list(range(len(label_set)))
    if label_map is None:
        label_map = dict()
        for lab, lab_id in zip(label_set, label_id):
            label_map.update({lab: lab_id})
    onehot_len = len(label_map)
    label_onehot = np.zeros((len(labels), len(label_map)))
    for i in range(len(labels)):
        label_id = label_map[labels[i]]
        label_onehot[i, label_id] = 1

    return label_onehot, label_map


def get_model(max_nb_words=50000, max_seq_len=150, embedding_dim=32, classification_num=20):
    tokenizer = Tokenizer(num_words=max_nb_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    model = Sequential()
    model.add(Embedding(max_nb_words, embedding_dim, input_length=max_seq_len, mask_zero=True))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(embedding_dim, dropout=0.2, recurrent_dropout=0.2, activation='tanh'))
    model.add(Dense(classification_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model_zip = {'model': model, 'tokenizer': tokenizer}
    return model_zip


def main():
    model_path = 'models/LSTM_level2'
    random_state = 42

    # load data
    train_label_list, train_text_list, val_label_list, val_text_list, test_label_list, test_text_list, class_table = get_data(
        1)

    label_list = train_label_list + val_label_list + test_label_list
    text_list = train_text_list + val_text_list + test_text_list

    Y, label_map = onehot_encoder(label_list)

    count_list = np.zeros(len(label_map))
    for lab in label_list:
        count_list[label_map[lab]] += 1
    x = list(range(len(label_map)))
    plt.subplots(figsize=(14, 10))
    plt.bar(x, count_list, width=1)
    plt.show()
    assert 0

    # 设置最频繁使用的25000个词
    MAX_NB_WORDS = 25000
    # 每条cut_review最大的长度
    MAX_SEQUENCE_LENGTH = 150
    # 设置Embeddingceng层的维度
    EMBEDDING_DIM = 32

    # get model
    model_zip = get_model(max_nb_words=MAX_NB_WORDS,
                          max_seq_len=MAX_SEQUENCE_LENGTH,
                          embedding_dim=EMBEDDING_DIM,
                          classification_num=len(label_map))

    model, tokenizer = model_zip.values()

    tokenizer.fit_on_texts(text_list)
    word_index = tokenizer.word_index
    print('共有 %s 个不相同的词语.' % len(word_index))

    X = tokenizer.texts_to_sequences(text_list)

    # 填充X,让X的各个列的长度统一
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

    print(X.shape)
    # print(Y.shape)

    # 拆分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=random_state)

    epochs = 5
    batch_size = 64

    model, history = train(model, X_train, Y_train, epochs=epochs, batch_size=batch_size)

    loss, acc = evaluate(model, X_test, Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(loss, acc))

    model_zip = {'model': model, 'tokenizer': tokenizer, 'label_map': label_map, 'history': history}
    # save model
    save_model(model_path, model_zip)


def train(model, train_set, train_label, epochs=5, batch_size=64):
    history = model.fit(train_set, train_label, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    return model, history


def evaluate(model, test_set, test_label):
    accr = model.evaluate(test_set, test_label)
    loss = accr[0]
    acc = accr[1]
    return loss, acc


def save_model(save_path, model_zip):
    if not osp.exists(save_path):
        os.makedirs(save_path)
    model = model_zip.get('model')
    tokenizer = model_zip.get('tokenizer')
    label_map = model_zip.get('label_map')
    history = model_zip.get(tokenizer)
    model.save(osp.join(save_path, 'model'))
    with open(osp.join(save_path, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(osp.join(save_path, 'label_map.npy'), label_map, allow_pickle=True)
    with open(osp.join(save_path, 'history.pickle'), 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(model_path):
    model = keras.models.load_model(osp.join(model_path, 'model'))
    with open(osp.join(model_path, 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)
    label_map = np.load(osp.join(model_path, 'label_map.npy'), allow_pickle=True).item()
    with open(osp.join(model_path, 'history.pickle'), 'rb') as handle:
        history = pickle.load(handle)
    model_zip = {'model': model, 'tokenizer': tokenizer, 'label_map': label_map, 'history': history}
    return model_zip


def tsne_visualization(model, tokenizer):
    word_index = tokenizer.word_index
    index_word = dict(zip(word_index.values(), word_index.keys()))
    # model.summary()
    embedding_weights = model.get_layer("embedding").get_weights()[0]
    embedding_weights = embedding_weights[:len(word_index), :]

    tsne = TSNE(n_components=2)
    tsne_embedding = tsne.fit_transform(embedding_weights)

    # print(tsne_embedding.shape)

    # plot
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.subplots(figsize=(14, 10))
    x = tsne_embedding[1:201, 0]
    y = tsne_embedding[1:201, 1]
    plt.scatter(x, y)

    plt.show()

    plt.subplots(figsize=(14, 10))
    for i in range(200):
        plt.text(x[i], y[i], index_word[i + 1])

    plt.scatter(x, y)
    plt.show()


def get_dataset(mode='all'):
    # load data
    train_label_list, train_text_list, val_label_list, val_text_list, test_label_list, test_text_list, class_table = get_data(
        1)
    if mode == 'all':
        text_list = train_text_list + val_text_list + test_text_list
        label_list = train_label_list + val_label_list + test_label_list
        return text_list, label_list
    elif mode == 'train':
        text_list = train_text_list + val_text_list
        label_list = train_label_list + val_label_list
        return text_list, label_list
    else:
        return test_text_list, test_label_list


def draw_dataset_len_histogram(bins=10, max_len=None):
    text_list, label_list = get_dataset(mode='all')
    text_lens = []
    over_max_len_count = 0
    for text in text_list:
        if max_len is not None and len(text) > max_len:
            over_max_len_count += 1
            continue
        text_lens.append(len(text))
    plt.hist(text_lens, bins=bins)
    plt.show()

    if max_len is not None:
        print("Max len is set to " + str(max_len))
        print("Longer than max_len = " + str(over_max_len_count))
        print("Total text count = " + str(len(text_list)))


def load_example():
    # load model
    model_zip = load_model('models/LSTM')

    # 解压model_zip
    m, token, label_m, his = model_zip.values()

    # load data
    train_label_list, train_text_list, val_label_list, val_text_list, test_label_list, test_text_list, class_table = get_data(
        1)

    # 将数据合成一个大的数据集
    label_list = train_label_list + val_label_list + test_label_list
    text_list = train_text_list + val_text_list + test_text_list

    # label encoder 如果指定了label_map，则不会新统计出一个label_map
    Y, label_map = onehot_encoder(label_list, label_m)
    X = token.texts_to_sequences(text_list)
    X = pad_sequences(X, maxlen=150)

    # 拆分测试训练集，因为data preprocess里面不能固定随机数，所以这里测试集里面会有训练集的数据，正确率偏高
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

    # 测试使用方法
    loss, acc = evaluate(m, X_train, Y_train)
    print('Loss : {:0.3f}, Accuracy : {:0.3f}'.format(loss, acc))


if __name__ == "__main__":
    # main
    main()
    # load model example
    # load_example()
