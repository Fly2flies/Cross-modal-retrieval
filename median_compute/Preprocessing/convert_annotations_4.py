"""
按照字典转换图片的标注
形如dict(
    图片名: 标注列表
)
"""
import os
import json
import pprint
import gensim

root_dir = "/root/fly2fly/median_compute/flick30k"
# root_dir = "D:\\ImageData\\flick30k"
img_dir = "flickr30k-images"
text_file = 'results_20130124.token'
word2VecBin_path = 'Download/GoogleNews-vectors-negative300.bin'

def getfullpath(subdir):
    return os.path.join(root_dir,subdir)
def parse_token_file(token_file):
    '''
    解析token文件
    :param token_file: 文件路径
    :return: dict 形式如： {'1234.jpg': ['this is a people', 'the people is happy']}
    '''
    img_name_to_token = {}
    with open(token_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:  # image captions
        img_id, description = line.strip('\r\n').split('\t')
        img_name, _ = img_id.split('#')
        img_name_to_token.setdefault(img_name, [])
        img_name_to_token[img_name].append(description)
    return img_name_to_token

annations_json = 'flick30k_annotations.json'
image_to_token = parse_token_file(getfullpath(text_file))

with open(getfullpath(annations_json), 'w') as f:
    f.write(json.dumps(image_to_token, indent= 4))