"""
处理文本的CNN
"""
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

DEBUG = 0
# The Text ResBlock
class BasicResBlock(nn.Module):
    def __init__(self,input_channels, hidden_channels):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels,
                               hidden_channels,
                               kernel_size= (1,1),
                               stride = (1,1),
                               padding = (0,0),
                               bias = False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(hidden_channels,
                               hidden_channels,
                               kernel_size = (1,2),
                               stride = (1,1),
                               padding = (0,1),
                               bias = False,
                               dilation = (1,2))
        self.bn2 = nn.BatchNorm2d(hidden_channels)

        self.conv3 = nn.Conv2d(hidden_channels,
                               input_channels,
                               kernel_size = (1,1),
                               stride=(1, 1),
                               padding=(0, 0),
                               bias = False)
        self.bn3 = nn.BatchNorm2d(input_channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))
        out += identity

        return self.relu(out)

# The InceptionResBlock to deepen
class BasicInceptionBlock(nn.Module):
    def __init__(self,input_channel, hidden_channel, output_channel,
                 hidden_stride = (1,1)):
        super(BasicInceptionBlock, self).__init__()
        self.relu = nn.ReLU()
        # One head
        self.Conv1 = nn.Conv2d(input_channel, hidden_channel,
                               kernel_size=(1,1),stride=(1,1),
                               padding=(0,0),bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channel)

        self.Conv2 = nn.Conv2d(hidden_channel, hidden_channel,
                               kernel_size=(1,2), stride= hidden_stride,
                               padding=(0,1), bias=False,
                               dilation=(1,2))
        self.bn2 = nn.BatchNorm2d(hidden_channel)

        self.Conv3 = nn.Conv2d(hidden_channel, output_channel,
                               kernel_size=(1,1), stride=(1,1),
                               padding=(0,0),bias=False)
        self.bn3 = nn.BatchNorm2d(output_channel)

        # Two Head
        self.Conv1_1 = nn.Conv2d(input_channel,output_channel,
                                 kernel_size=(1,1),stride = hidden_stride,
                                 padding=(0,0), bias= False)
        self.bn1_1 = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        identity = x
        out1 = self.relu(self.bn1(self.Conv1(x)))
        out1 = self.relu(self.bn2(self.Conv2(out1)))
        out1 = self.bn3(self.Conv3(out1))

        out2 = self.bn1_1(self.Conv1_1(identity))
        out2 += out1
        return self.relu(out2)

class textCNN(nn.Module):
    def __init__(self,
                 input_channel = 300,
                 embedding_dims = 9611,
                 word2vec = 'word2vec.pkl'): # (N,6411,1,32) -> (N,300,1,30)
        super(textCNN, self).__init__()
        self.emb_layer = self.create_embedding_layer(weight_matrix = word2vec,
                                                     input_size = embedding_dims,
                                                     embedding_dims = input_channel)
        # ---- First CNN Block --- 32 * 256
        self.Block1 = BasicInceptionBlock(input_channel,128,256)
        self.layer1 = self.__make_layer(input_channel = 256,hidden_channel = 64, num_blocks = 2)

        # ---- Second CNN Block ---- 16 * 512
        self.Block2 = BasicInceptionBlock(256,512,512,hidden_stride=(2,2))
        self.layer2 = self.__make_layer(input_channel=512, hidden_channel=128, num_blocks=3)

        # ---- Third CNN Block ---- 8 * 1024
        self.Block3 = BasicInceptionBlock(512,1024,1024,hidden_stride=(2,2))
        self.layer3 = self.__make_layer(input_channel=1024,hidden_channel=256,num_blocks=5)

        # ---- Four CNN Block ---- 8 * 2048
        self.Block4 = BasicInceptionBlock(1024,2048,2048)

        # ---- Output Layer ----
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(2048,2048)
        self.fc1_bn = nn.BatchNorm1d(2048)
        self.dp = nn.Dropout(0.75)

        self.relu = nn.ReLU()
    def forward(self, x): # (N , 32)
        # x = torch.LongTensor(x) # 索引的数值类型
        x = self.emb_layer(x) # (N,32,300)
        if DEBUG: print(x[0][0])
        if DEBUG: print(x[0][1])
        x = x.permute(0, 2, 1).unsqueeze(2) #(N, 300, 1, 32)
        # ---- First Layer ----
        x = self.Block1(x) # (N,256,1,32)
        x = self.layer1(x) # (N,256,1,32)
        # ---- Second Layer ----
        x = self.Block2(x) # (N,512,1,16)
        x = self.layer2(x) # (N,512,1,16)
        # ---- Third Layer ----
        x = self.Block3(x) # (N,1024,1,8)
        x = self.layer3(x) # (N,1024,1,8)
        # ---- Four Layer ----
        x = self.Block4(x) # (N,2048,1,8)
        # ---- Output Layer ----
        x = self.avgpool(x) # ([N, 2048, 1, 1])
        x = torch.flatten(x,1) # (N,2048)
        out = self.dp(self.relu(self.fc1_bn(self.fc1(x))))
        return out

    def __make_layer(self,input_channel,hidden_channel,num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(BasicResBlock(input_channel,hidden_channel))
        return nn.Sequential(*layers)
    def create_embedding_layer(self, weight_matrix,input_size, embedding_dims, trainable = False):
        if weight_matrix is not None:
            with open(weight_matrix, 'rb') as file:
                word2vec = pickle.load(file)
        else:
            word2vec = np.random.randn(input_size,embedding_dims)
        word2vec = torch.from_numpy(word2vec)
        num_embeddings, embedding_dims = word2vec.size()
        emb_layer = torch.nn.Embedding(num_embeddings, embedding_dims)
        emb_layer.load_state_dict({'weight': word2vec})
        if trainable is False:
            emb_layer.weight.require_grad = False
        return emb_layer

if __name__ == '__main__':
    x = torch.randint(0,6411,[16,32])
    x = torch.LongTensor(x)
    print(x.size())
    cnn = textCNN(word2vec = None)
    out = cnn(x)
    print(out.size())