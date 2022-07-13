import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from data_process import get_data


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
