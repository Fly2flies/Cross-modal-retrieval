# 读取描述，并建立字典和索引，索引到字典的映射
class Vocab(object):
    '''
    构建词表
    '''

    def __init__(self, vocab_file, word2vec,word_num_threshlod):
        self._id_to_word = {}
        self._word_to_id = {}  # 单词 到 索引 的映射，用于网络输入

        self._unk = -1  # unknown
        self._eos = -1  # 句号

        self._word_num_threshlod = word_num_threshlod  # 单词的词频阈值
        self._read_dict(vocab_file)

        self.word2vec = word2vec # 预训练模型
    def _read_dict(self, filename):
        # 将 词表 转换成 字典
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            word, occurence = line.strip('\r\n').split('\t')
            occurence = int(occurence)
            if word != '<UNK>' and word != '.' and occurence < self._word_num_threshlod:
                continue  # 过滤
            if word != '<UNK>' and word != '.' and word not in self.word2vec:
                continue  # 过滤
            # 按照进入字典的顺序 分配索引
            idx = len(self._id_to_word)

            if word == '<UNK>':
                self._unk = idx
            elif word == '.':
                self._eos = idx
            if idx in self._id_to_word or word in self._word_to_id:
                raise Exception('duplicate words in vocab file')

            # 构建两个索引
            self._word_to_id[word] = idx
            self._id_to_word[idx] = word

    @property
    def unk(self):
        return self._unk

    @property
    def eos(self):
        return self._eos

    @property
    def Word2ID(self):
        return self._word_to_id

    @property
    def ID2Word(self):
        return self._id_to_word

    def word_to_id(self, word):
        return self._word_to_id(word, self.unk)

    def id_to_word(self, id):
        return self._id_to_word(id, '<UNK>')

    def size(self):
        # 词表长度
        return len(self._word_to_id)

    def encode(self, sentence):
        '''
        将一个描述中的单词，映射成 id 表示
        :param sentence: 描述语句
        :return: 词id句子
        '''
        word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split(' ')]
        return word_ids

    def encode(self, sentence_code):
        '''
        将一个 id 句子，转化为 单词句子
        :param sentence_code:
        :return:
        '''
        words = [self.id_to_word(word_id) for word_id in sentence_code]
        return words
