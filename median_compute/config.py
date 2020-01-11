config = {
    "train":
        {
            "batchsize":32,
            "num_classes":29784,
            "ranks": [1,5,10,0],
            "epoches":2000,
            "eval_epoch":5,
        },
    "dataset":
        {
            "data_folder":"./QueryApp/DATA/Flick_10k", # 数据根目录
            "descriptions":'annotations.json', # 标注文件
            "images":"flick10k_image_256", # 图片路径
            "word2idxs": "text_info.json", # 事先准备的字典文件
            "max_sentence_len":32, # 截取的句子长度
            "word2vec":"word2vec.pkl",
            "train_id" : "flick10k_train.txt",
            "val_id" : "flick10k_val.txt",
            "test_id" : "flick10k_test.txt",
        },
    "CUDA_VISIBLE_DEVICES": "0"
}