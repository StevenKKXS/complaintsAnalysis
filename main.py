import os.path as osp
import re

import jieba as jb
import pandas as pd

from visualization_utils import tsne_visualization_3d


def split_by_punc(str_in):
    return str(str_in).split('，')


if __name__ == '__main__':
    tsne_visualization_3d('models/LSTM', show_head_num=None, is_use_cache=True)
