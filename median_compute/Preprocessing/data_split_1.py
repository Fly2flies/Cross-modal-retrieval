"""子集划分"""
import random
import os

root_dir = "/root/fly2fly/median_compute/flick30k"
img_dir = "flickr30k-images"
text_file = 'results_20130124.token'

def getfullpath(subdir):
    return os.path.join(root_dir,subdir)

train_len = 29784
val_len = 1000
test_len = 1000

train_fid = 'flick30k_train.txt'
val_fid = 'flick30k_val.txt'
test_fid = 'flick30k_test.txt'

imglist = os.listdir(getfullpath(img_dir))
imglist.sort() # 保持每次划分都一样
train_list = imglist[0:train_len]
val_list = imglist[train_len:train_len + val_len]
test_list = imglist[-test_len-1:-1]

ids_dict = {} # 分配类别
for i,img in enumerate(imglist):
    ids_dict[img] = i + 1

print("总的长度",len(set(imglist)))
print("训练集长度",len(train_list))
print("验证集长度",len(val_list))
print("测试集长度",len(test_list))

for i,fid in enumerate([train_fid,val_fid,test_fid]):
    with open(getfullpath(fid),'w') as f:
        for name in [train_list,val_list,test_list][i]:
            f.write(name[:-4] + ' ' + str(ids_dict[name]) + '\n')