"""
统计词频
"""
import os
import nltk
from nltk.tokenize import word_tokenize
import gensim
import pprint

#root_dir = "D:\\ImageData\\flick30k"

output_vocab_file = 'flick30k_vocab.txt' # 词频
output_sentence_file = 'flick30k_sentence.txt' # 句频
root_dir = "/root/fly2fly/median_compute/flick30k"
img_dir = "flickr30k-images"
text_file = 'results_20130124.token'
word2VecBin_path = 'Download/GoogleNews-vectors-negative300.bin'

def getfullpath(subdir):
    return os.path.join(root_dir,subdir)

word2Vec = gensim.models.KeyedVectors.load_word2vec_format(
    getfullpath(word2VecBin_path), binary=True)
print(len(word2Vec.vocab)) # 3,000,000 个单词

def count_vocab(input_description_file):
    with open(input_description_file) as f:
        lines = f.readlines()
    max_length_of_sentences = 0 # 所有句子中 最长长度
    max_sentence = None
    total_length = 0
    length_dict = {} #　统计　句子长度字典　｛长度：句子总数｝
    vocab_dict = {} # 词表字典　｛词：词频｝
    for line in lines:
        image_id, description = line.strip('\n').split('\t')
        words = description.strip(' ').split() # 分词
        # words 的 格式 ['Two', 'young', 'guys', 'with', 'shaggy', 'hair', ……]

        max_length_of_sentences = max(max_length_of_sentences, len(words)) # 选择一个最大值放入
        total_length += len(words)
        if len(words) == max_length_of_sentences:
            max_sentence = image_id
        length_dict.setdefault(len(words), 0)
        length_dict[len(words)] += 1

        # 词表 统计
        for word in words:
            word = word.lower()
            if word != '.' and word not in word2Vec: continue # 过滤
            vocab_dict.setdefault(word, 0)
            vocab_dict[word] += 1
    print("最长句子",max_sentence)
    print("词表长度", len(vocab_dict.keys()))
    print("句子总长度",total_length)
    print("最长句子长度",max_length_of_sentences)
    pprint.pprint(length_dict)
    return vocab_dict, length_dict

vocab_dict, length_dict = count_vocab(getfullpath(text_file))
sorted_vocab_dict = sorted(vocab_dict.items(), key = lambda d:d[1], reverse=True) #对 词表进行排序
sorted_length_dict = sorted(length_dict.items(), key = lambda d : d[0], reverse = True)

with open(getfullpath(output_vocab_file), 'w') as f:
    f.write('<UNK>\t1000000\n')
    for item in sorted_vocab_dict:
        f.write('%s\t%d\n' % item)
with open(getfullpath(output_sentence_file), 'w') as f:
    for item in sorted_length_dict:
        f.write('%d,%d\n' % item )
