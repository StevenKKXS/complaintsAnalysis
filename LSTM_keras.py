import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import jieba as jb
import re

from data_process import *


def onehot_encoder(labels):
    label_set = set()
    for lab in labels:
        label_set.add(lab)

    label_id = list(range(len(label_set)))
    label_map = dict()
    for lab, lab_id in zip(label_set, label_id):
        label_map.update({lab: lab_id})

    onehot_len = len(label_map)
    label_onehot = np.zeros((len(labels), len(label_map)))
    for i in range(len(labels)):
        label_id = label_map[labels[i]]
        label_onehot[i, label_id] = 1

    return label_onehot, label_map


def train_and_eval(model_path='models/LSTM', random_state=42):
    # load data
    train_label_list, train_text_list, val_label_list, val_text_list, test_label_list, test_text_list, class_table = get_data(
        1)

    label_list = train_label_list + val_label_list + test_label_list
    text_list = train_text_list + val_text_list + test_text_list

    Y, label_map = onehot_encoder(label_list)

    # 设置最频繁使用的50000个词
    MAX_NB_WORDS = 50000
    # 每条cut_review最大的长度
    MAX_SEQUENCE_LENGTH = 150
    # 设置Embeddingceng层的维度
    EMBEDDING_DIM = 32

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
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
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(EMBEDDING_DIM, dropout=0.2, recurrent_dropout=0.2, activation='tanh'))
    model.add(Dense(len(label_map), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    epochs = 5
    batch_size = 64

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    accr = model.evaluate(X_test, Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

    # save model
    model.save(model_path)

    return model


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


if __name__ == "__main__":
    train_and_eval()
