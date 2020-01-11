import numpy as np
import torch
import os
from PIL import Image
from torch.autograd import Variable

def rm_dict(dicts, keys):
    # 去除字典中的keys
    ans = dict()
    for k,v in dicts.items():
        if k not in keys:
            ans[k] = v
    return ans
def l2norm(input, p = 2.0, dim = 1, eps =1e-12):
    '''对行向量组进行单位化，方便计算余弦距离'''
    if isinstance(input,torch.Tensor):
        norm = input.norm(p, dim, True).clamp(min=eps).expand_as(input)
    else:
        norm = np.maximum(np.linalg.norm(input,ord=p,axis=dim, keepdims=True),eps)
    return input / norm
def encode_sentences(model, X, verbose = False, batch_size = 32, use_gpu = True):
    """
    Encode sentence into commom embedding space
    """
    features = Variable(torch.zeros(len(X),2048))
    if use_gpu:
        features = features.cuda()
    num_batches = len(X) // batch_size + 1

    for minbatch in range(num_batches):
        x = X[minbatch * batch_size: (minbatch + 1) * batch_size]
        if use_gpu:
            x = x.cuda()
        with torch.no_grad():
            feature = model['textcnn'](x)

        features[minbatch * batch_size: (minbatch + 1) * batch_size] = feature
    return features
def encode_images(model, X,verbose = False, batch_size = 32, use_gpu = True):
    """
    Encode images into commom embedding space
    """
    features = Variable(torch.zeros(len(X), 2048))
    if use_gpu:
        features = features.cuda()

    num_batches = len(X) // batch_size + 1

    for minbatch in range(num_batches):
        x = X[minbatch * batch_size: (minbatch + 1) * batch_size]
        if use_gpu:
            x = x.cuda()
        with torch.no_grad():
            feature = model['imagecnn'](x)

        features[minbatch * batch_size: (minbatch + 1) * batch_size] = feature
    return features