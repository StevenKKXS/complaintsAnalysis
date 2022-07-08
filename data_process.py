from pydoc import describe
from tracemalloc import stop
import openpyxl
import numpy as np
import os, re
import random
import time
import jieba
t = time.time()


class Info:
    def __init__(self, serial_number, time, city_num, business_num, city_name, text_type, describe):
        self.serial_number = serial_number
        self.time = time
        self.city_num = city_num
        self.city_name = city_name
        self.business_num = business_num
        self.text_type = text_type.split('.')
        self.describe = cut_text(describe)
    
    def is_use(self):  # 过滤无效信息
        if self.describe != None and self.describe != '' \
            and not (len(self.describe) < 15 and '测试' in self.describe):
            return True
        else:
            return False


def read_stopwords(stopword_path='cn_stopwords.txt'):
    stopword_file = open(stopword_path, 'r', encoding='utf-8')
    stopwords = stopword_file.read().split("\n")
    return stopwords


stopwords = read_stopwords()


def cut_text(text):
    try:
        remove_chars = "[a-zA-Z0-9!%&()*+,-./:;<=>?@，。?★（）、…【】《》？“”‘’！[\]^_`{|}~\s]"
        text_ = re.sub(remove_chars, "",text)  # 去除标点
        cut_text = jieba.lcut(text_)  # 分词
        cut_text_ = [word for word in cut_text if word not in stopwords]  # 去除停用词
        return cut_text_
    except:
        return None


def read_data(data_path='工单信息19-29.xlsx'):
    data = openpyxl.load_workbook(data_path)
    sheet = data['Sheet2']
    is_first_line = True
    data_list = []
    for row in sheet.iter_rows():
        if is_first_line:  # 去掉第一行的类别目录
            is_first_line = False
        elif '热点' in row[5].value:  # 特殊情况去除
            describe = re.sub(r'[0-9]、', '\t', row[6].value)
            describe = describe.split('\t')
            for num in range(len(describe)):
                information = Info(row[0].value, row[1].value, row[2].value, 
                               row[3].value, row[4].value, row[5].value, describe[num])
                if information.is_use():
                    data_list.append(information) 
        else:
            information = Info(row[0].value, row[1].value, row[2].value, 
                               row[3].value, row[4].value, row[5].value, row[6].value)
            if information.is_use():
                data_list.append(information) 
    np.save('datalist.npy', data_list, allow_pickle=True, fix_imports=True)
    random.seed(10)
    random.shuffle(data_list)
    return data_list


def get_data(label_level=1, data_path='工单信息19-29.xlsx'):
    if os.path.exists('datalist.npy'):
        data_list = np.load('datalist.npy', allow_pickle=True)
    else:
        data_list = read_data(data_path)
    label_list = []
    text_list = []
    for i in range(len(data_list)):
        # 标签数目大于需要的数目级别,截取
        if label_level <= len(data_list[i].text_type):  
            label = data_list[i].text_type[0]
            for j in range(1, label_level):
                if data_list[i].text_type[j] != '':
                    label = label + '_' + data_list[i].text_type[j]
            label_list.append(label)
        # 标签数目小于需要的数目级别，直接取全部的标签
        else: 
            label = data_list[i].text_type[0]
            for j in range(1, len(data_list[i].text_type)):
                if data_list[i].text_type[j] != '':
                    label = label + '_' + data_list[i].text_type[j]
            label_list.append(label)
        text_list.append(data_list[i].describe)
    train_label_list = label_list[0:int(len(label_list)*0.6)]  # 分割
    train_text_list = text_list[0:int(len(text_list)*0.6)]
    val_label_list = label_list[int(len(label_list)*0.6)+1:int(len(label_list)*0.8)]
    val_text_list = text_list[int(len(text_list)*0.6)+1:int(len(text_list)*0.8)]
    test_label_list = label_list[int(len(label_list)*0.8)+1:len(label_list)-1]
    test_text_list = text_list[int(len(text_list)*0.8)+1:len(text_list)-1]
    class_table = list(set(label_list))
    return train_label_list, train_text_list, val_label_list, \
        val_text_list, test_label_list, test_text_list, class_table



if __name__ =='__main__':
    train_label_list, train_text_list, val_label_list, \
        val_text_list, test_label_list, test_text_list, class_table = get_data(label_level=1)
    print(time.time()-t)
    print(class_table)
    print(len(train_label_list))
    tmp_label = train_label_list + val_label_list + test_label_list
    tmp_describe = train_text_list + val_text_list + test_text_list
    max_len = max([len(tmp_) for tmp_ in tmp_describe])
    print(len(tmp_label), max_len)
        