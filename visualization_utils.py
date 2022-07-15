import os.path as osp
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px

from data_process import get_data
from LSTM_keras import load_model


def draw_dataset_len_histogram(bins=10, max_len=None, label_level=1):
    label_list, text_list = get_data(label_level=label_level)
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


def tsne_visualization_3d(model_path, show_head_num=None, is_use_cache=True):
    pattern = re.compile(r'[\\/]')
    model_name = re.split(pattern, model_path)[-1]
    cache_dir = 'tsne_cache'
    cache_file = 'tsne_3d_' + model_name + '.csv'
    if is_use_cache and osp.exists(osp.join(cache_dir, cache_file)):
        df = pd.read_csv(osp.join(cache_dir, cache_file))
    else:
        model_zip = load_model(model_path)
        model, tokenizer, label_map, his = model_zip.values()
        word_index = tokenizer.word_index
        index_word = dict(zip(word_index.values(), word_index.keys()))
        # model.summary()
        embedding_weights = model.get_layer("embedding").get_weights()[0]
        embedding_weights = embedding_weights[1:len(word_index) + 1, :]

        tsne = TSNE(n_components=3)
        tsne_embedding = tsne.fit_transform(embedding_weights)

        projection_lens = np.zeros(tsne_embedding.shape[0])
        for i in range(tsne_embedding.shape[0]):
            projection_lens[i] = np.dot(tsne_embedding[i, :], np.array([1, 1, 1]))
        projection_lens = projection_lens / np.sqrt(3)

        # convert to data frame in pandas
        tsne_dict = dict()
        tsne_dict.update({'word': list(word_index.keys())})
        tsne_dict.update({'x': list(tsne_embedding[:, 0])})
        tsne_dict.update({'y': list(tsne_embedding[:, 1])})
        tsne_dict.update({'z': list(tsne_embedding[:, 2])})
        tsne_dict.update({'project_len': list(projection_lens)})

        df = pd.DataFrame(tsne_dict)
        if is_use_cache:
            # save cache
            df.to_csv(osp.join(cache_dir, cache_file))
    if show_head_num is not None:
        df = df.head(show_head_num)
    # visualization
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='project_len', hover_name='word')
    fig.show()
