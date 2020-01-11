from __future__ import print_function, absolute_import
import glob
import random
import os
import re
import sys
import json
import pickle
import numpy as np
from scipy.misc import imsave
import random
import pprint
from time import time

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class Flick30k_triplet_dataset(Dataset):
    def __init__(self,
                 data_folder = 'flick30k_data',
                 descriptions = 'annotations.json',
                 images = 'flick30k_image_256',
                 word2idxs = 'text_info.json',
                 train_id = "flick10k_train.txt",
                 val_id = "flick10k_val.txt",
                 test_id = "flick10k_test.txt",
                 max_sentence_len = 32,
                 shift = True,
                 transfroms_list = None,
                 samples = None,
                 randomTriplet = True, # 是不是随机构成三元组
                 mode = 'train'):
        if mode == 'train':
            self.id_file = train_id
        elif mode == 'test':
            self.id_file = test_id
        else:
            self.id_file = val_id
        self.position_shift = shift
        self.max_words_len = max_sentence_len
        file_path = os.path.join(data_folder, self.id_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            self.ids = file.readlines()
        id_labels = [x.strip('\n').split() for x in self.ids]
        self.ids = [x[0] for x in id_labels]
        self.ids.sort()
        if samples:
            self.ids = self.ids[0:samples]
        self.id_dict = {}
        for x in id_labels:
            self.id_dict[x[0]] = int(x[1]) - 1 # 从0开始

        desciption_path = os.path.join(data_folder,descriptions)
        with open(desciption_path, 'r') as file:
            self.annotation = json.load(file)

        word2idx_path = os.path.join(data_folder, word2idxs)
        with open(word2idx_path, 'r') as file:
            self.word2id = json.load(file)
            self.word2id = self.word2id['word2id']
        self.image_path = os.path.join(data_folder, images)
        if transfroms_list:
            self.transform = transforms.Compose(transfroms_list)
        else:
            self.transform = None

        # 创建一个已经取过的序列
        self.candiate = [ random.randrange(0,len(self.ids)) for i in range(len(self.ids))]
        self.randomTriplet = randomTriplet
    def __getitem__(self, index):
        first_id = self.ids[index]
        first_image = os.path.join(self.image_path, first_id + '.jpg')
        first_text = np.random.choice(self.annotation[first_id + '.jpg'])

        if self.randomTriplet:
            second_id = np.random.choice([x for x in self.ids if x != first_id])
        else:
            second_id = self.ids[self.candiate[index]]
            if second_id == first_id:
                self.candiate[index] = (self.candiate[index] + 1) % len(self.ids)
                second_id = self.ids[self.candiate[index]]
            # Next Candiate
            self.candiate[index] = (self.candiate[index] + 1) % len(self.ids)
        second_image = os.path.join(self.image_path, second_id + '.jpg')
        second_text = np.random.choice(self.annotation[second_id + '.jpg'])

        first_label = np.array(self.id_dict[first_id])
        first_image = Image.open(first_image)
        first_text = self._sentence_encode(first_text)

        second_label = np.array(self.id_dict[second_id])
        second_image = Image.open(second_image)
        second_text = self._sentence_encode(second_text)

        # 对描述进行数据增广
        # 1. 固定长度
        # 2. 首尾随机填0
        first_text = first_text[0:self.max_words_len]
        second_text = second_text[0:self.max_words_len]
        if len(first_text) < self.max_words_len:
            first_text = self._random_padding_zero(first_text)
        if len(second_text) < self.max_words_len:
            second_text = self._random_padding_zero(second_text)
        first_text = np.array(first_text, dtype=np.long)
        second_text = np.array(second_text, dtype=np.long)

        if not self.transform is None:
            first_image = self.transform(first_image)
            second_image = self.transform(second_image)

        modality_image = torch.tensor([1,0]).float()
        modality_text = torch.tensor([0,1]).float()

        return first_image,first_text, first_label, \
               second_image, second_text, second_label,modality_image,modality_text
    def _random_padding_zero(self, code):
        if self.position_shift is False:
            code = np.pad(code, (0,self.max_words_len - len(code)), 'constant')
        else:
            left = np.random.randint(0,self.max_words_len - len(code))
            right = self.max_words_len - len(code) - left
            code = np.pad(code, (left, right), 'constant')
        return code
    def _sentence_encode(self, sentence):
        words = [word.lower() for word in sentence.split(' ')]
        code = [0] * len(words)
        for i,word in enumerate(words):
            if word in self.word2id.keys():
                code[i] = self.word2id[word]
            else:
                code[i] = self.word2id['<UNK>']
        return code
    def __len__(self):
        return len(self.ids)

class Flick303k_eval_dataset(Dataset):
    """
    生成测试用的数据：图像检索文本 + 文本检索图像
    只返回图像路径和文本编码
    """
    def __init__(self,
                 data_folder = 'flick30k_data',
                 descriptions = 'annotations.json',
                 images = 'flick30k_image_256',
                 word2idxs = 'text_info.json',
                 max_sentence_len = 32,
                 train_id="flick10k_train.txt",
                 val_id="flick10k_val.txt",
                 test_id="flick10k_test.txt",
                 mode = 'val',
                 samples = None):
        if mode == 'train':
            self.id_file = train_id
        elif mode == 'test':
            self.id_file = test_id
        else:
            self.id_file = val_id

        self.max_words_len = max_sentence_len
        file_path = os.path.join(data_folder, self.id_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            self.ids = file.readlines()
        id_labels = [x.strip('\n').split() for x in self.ids]
        self.ids = [x[0] for x in id_labels]
        self.ids.sort()
        if samples:
            self.ids = self.ids[0:samples]
        self.id_dict = {}
        for x in id_labels:
            self.id_dict[x[0]] = int(x[1]) - 1

        desciption_path = os.path.join(data_folder, descriptions)
        with open(desciption_path, 'r') as file:
            self.annotation = json.load(file)

        word2idx_path = os.path.join(data_folder, word2idxs)
        with open(word2idx_path, 'r') as file:
            self.word2id = json.load(file)
            self.word2id = self.word2id['word2id']

        self.image_path = os.path.join(data_folder, images)

        query_images, images_ids,query_texts, text_ids = self._process_query_data(self.ids)
        print("Dataset statistics")
        print("  -----------------------")
        print("  subset | # images | # texts ")
        print("  {}  |  {}   | {}".format(mode,len(query_images),len(query_texts)))
        print("  -----------------------")
        self.images = query_images
        self.imageslabels = images_ids
        self.texts = query_texts
        self.textslabels = text_ids

    def get_data(self):
        return self.images,self.imageslabels,self.texts,self.textslabels

    def _process_query_data(self,ids, relabel = False):
        images = []
        images_labels = []
        texts = []
        texts_labels = []
        for i,id in enumerate(ids):
            image_path = os.path.join(self.image_path,id + '.jpg')
            images.append(image_path)
            if relabel:
                images_labels.append(i)
            else:
                images_labels.append(id)

            descriptions = self.annotation[id + '.jpg']
            for desc in descriptions:
                texts.append(self._sentence_encode(desc))
                texts_labels.append(id)
        return images,images_labels,texts,texts_labels
    def _sentence_encode(self, sentence):
        words = [word.lower() for word in sentence.split(' ')]
        code = [0] * len(words)
        for i,word in enumerate(words):
            if word in self.word2id.keys():
                code[i] = self.word2id[word]
            else:
                code[i] = self.word2id['<UNK>']
        if len(code) < self.max_words_len:
            code = np.pad(code, (0,self.max_words_len - len(code)), 'constant')
        elif len(code) > self.max_words_len:
            code = code[0:self.max_words_len]

        return code


if __name__ == '__main__':
    transforms_list = [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    Flick10k = Flick30k_triplet_dataset(
        data_folder= '../Data/flick10k_data',
        transfroms_list = transforms_list)

    first_image, first_text, first_label, \
    second_image, second_text, second_label, modality_image, modality_text = Flick10k[0]
    print(first_image.size())
   
    trainLoader = DataLoader(Flick10k, shuffle = True, batch_size = 4)
    for i, batch in enumerate(trainLoader):
        if i > 100 : break
        first_image, first_text, first_label, \
        second_image, second_text, second_label, modality_image, modality_text = batch
        print(first_image.size())
        print(first_label.size())
        first_label = first_label.long()
        first_label = first_label.unsqueeze(-1)
        ont_hot = torch.zeros(4, 12000).scatter_(1,first_label,1)
        print(first_label)
        print(torch.argmax(ont_hot, dim=1))





