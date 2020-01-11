'''
图文检索的评价准则
Recall@ K
- the possibility that the true match appears in the top K of the rank list, where a higher score is better.
- 正确答案出现在前K个返回结果的样例占总样例的比例（召回率）
Median Rank
- the median rank of the closest ground truth result in the rank list, with a lower index being better.
- 在结果排序中，第一个真实样本出现的位置的中位数
- 也是使得 Recall@K >= 50% 的最小K值
'''
import numpy as np
import torch
import os
from PIL import Image
from torch.autograd import Variable
from tools import encode_images, encode_sentences,l2norm

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def evalRank(model, data,batchsize = 64,transforms_list = None, use_gpu = True,verbose = False):
    """
    Evaluate a trainrd model on val or test dataset
    """
    model['imagecnn'].eval()
    model['textcnn'].eval()

    imgs,labels_i, caps,labels_c= data.get_data()
    images = Variable(torch.empty((len(imgs),3,224,224)))
    captions = Variable(torch.empty(len(caps),32, dtype = torch.long))
    for i,cap in enumerate(caps):
        cap = torch.Tensor(cap)
        captions[i] = cap
    for i,img in enumerate(imgs):
        img = Image.open(img)
        if transforms_list:
            img = transforms_list(img)
        images[i] = img
    del imgs
    del caps
    with torch.no_grad():
        imgs_codes = encode_images(model, images, batch_size=batchsize,use_gpu=True)
        captions_codes = encode_sentences(model, captions, batch_size=batchsize, use_gpu=True)
        imgs_codes = imgs_codes
        captions_codes = captions_codes
        (r1,r5,r10,medr) = image2txt(imgs_codes, captions_codes)
    if verbose:
        print("Image to text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr))
    (r1i, r5i, r10i, medri) = txt2image(captions_codes, imgs_codes)
    if verbose:
        print("Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri))

    model['imagecnn'].train()
    model['textcnn'].train()

    return (r1,r5,r10,medr), (r1i, r5i, r10i, medri)
def image2txt(images, captions, npts = None, verbose = False):
    """ Image <--> 5 * captions
    Image->Text (Image, Annotation)
    :param images:   (N,K) matrix of images
    :param captions:  (5N,K) matrix of captions
    :param npts: numbers of image-text group
    :return: Recall & Median Rank
    """
    if npts is None:
        npts = images.size()[0]

    ranks = np.zeros(npts) # Each Image's first rank
    captions = l2norm(captions) # L2Norm
    for index in range(npts):
        # Get query Image
        im = images[index]
        im = im.unsqueeze(0)
        # Compute scores
        im = l2norm(im)
        d = torch.mm(im, captions.t())
        d_sorted, ins = torch.sort(d, descending=True) # 从大到小的距离和索引
        inds = ins.data.squeeze(0).cpu().numpy()

        # Score
        rank = 1e20
        # find the highest ranking
        for i in range(5*index, 5*index + 5,1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp

        ranks[index] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1

    if verbose:
        print("		* Image to text scores: R@1: %.1f, R@5: %.1f, R@10: %.1f, Medr: %.1f" % (r1, r5, r10, medr))
    return (r1, r5, r10, medr)
def txt2image(captions, images, npts = None, verbose = False):
    """
    :param captions: [5N, K]
    :param images:  [N , K]
    :param npts:
    :param verbose:
    :return:
    """
    if npts is None:
        npts = images.size()[0]

    ranks = np.zeros(5 * npts)
    images = l2norm(images)

    for index in range(npts):
        # Get query captions
        queries = captions[5 * index: 5 * index + 5]
        # Compute scores
        d = torch.mm(queries, images.t())
        for i in range(d.size()[0]):
            d_sorted, inds = torch.sort(d[i], descending=True)
            inds = inds.data.squeeze(0).cpu().numpy()
            ranks[5 * index + i] = np.where(inds == index)[0][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1

    if verbose:
        print("		* Text to image scores: R@1: %.1f, R@5: %.1f, R@10: %.1f, Medr: %.1f" % (r1, r5, r10, medr))
    return (r1, r5, r10, medr)







