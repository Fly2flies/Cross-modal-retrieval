"""
1） 改变图像短边为 256，且保持图像长宽比例不变
2)  计算训练集合的均值和方差（直接跳过，归一化后直接用0.5和0.5代替）
"""
import os
import cv2

#### 设置路径 ####
root_dir = "/root/fly2fly/median_compute/flick30k"
img_dir = "flickr30k-images"
text_file = 'results_20130124.token'
train_fid = 'flick30k_train.txt'
savepath = 'flick30k_image_256'

def getfullpath(subdir):
    return os.path.join(root_dir,subdir)

#### 初始化列表 ####
train_len = 29784
val_len = 1000
test_len = 1000
imglist = os.listdir(getfullpath(img_dir))
imglist.sort()
train_list = imglist[0:train_len]
val_list = imglist[train_len:train_len + val_len]
test_list = imglist[-test_len-1:-1]

#### 预处理数据 ####
# 1) 改变形状
if not os.path.exists(getfullpath(savepath)):
    os.makedirs(getfullpath(savepath))

s = 256
for name in imglist:
    fullpath = os.path.join(getfullpath(img_dir), name)
    img = cv2.imread(fullpath)
    h,w,c = img.shape

    if h < w:
        rate = s / h
        # (w , h)
        img = cv2.resize(img,(round(rate*w), s), interpolation = cv2.INTER_CUBIC)
    else:
        rate = s / w
        img = cv2.resize(img,(s, round(rate*h) ), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(getfullpath(savepath), name),img)

