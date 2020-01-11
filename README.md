# 跨模态检索：图像——文本检索
媒体计算实践作业：图像——文本跨模态搜索

## 数据集下载
本项目使用的是[Flickr30k数据集](http://shannon.cs.illinois.edu/DenotationGraph/data/index.html)，你需要自行先下载。
[百度云地址](https://pan.baidu.com/s/10z2LTaQWzIlfBuQOunf7yA)

## 数据预处理

在Preprocessing下:
- `data_split_1.py` 划分训练集、测试集、验证集
- `resize_data_2.py` 长宽比例不变，将短边拉伸为256
- `count_vocab_3.py` 统计每个单词的词频
- `convert_annotations_4.py` 将.txt格式的标注文件转换为.json
- `build_dictionary_5.py` 构建单词编号，即查询字典

## 模型训练

在数据预处理完成后，在`config.py`中配置各文件的路径以及训练的参数
- `trainStage1.py` 使用分类损失预训练
- `trainStage2.py` 使用三元组损失和对抗损失微调

## 测试界面

在 QueryApp 下的 `图文互搜.exe` 提供简单的测试界面。需要提前下载预训练模型`imgcnn.pth`和`textcnn.pth`到`DATA/Checkpoint`下方便自动初始化，`captions_database.pkl`和`images_database.pkl`事先提取的图像和文本特征以及其索引到`DATA/`下，字典`text_info.json`到`DATA/Flick_10k`，图片数据到`DATA/Flick_10k/flick_image_256`下。也可以自己选择路径，但是后续检索的时候不支持自动初始化。
