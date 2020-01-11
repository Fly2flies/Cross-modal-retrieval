"""
接着之前的检查点训练
- 更强的正则，因为训练集和验证集的精度差别过大, 去掉随机翻转
"""
from DataSet.flick30k_dataset import Flick30k_triplet_dataset
from DataSet.flick30k_dataset import Flick303k_eval_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from Models.models import IdClassifier
from Models.ImageCNN import imageCNN
from Models.TextCNN import textCNN
from evaluation import evalRank

from tools import  rm_dict
from config import config
from logger import  Logger
import os

os.environ["CUDA_VISIBLE_DEVICES"]= config["CUDA_VISIBLE_DEVICES"]
TRAIN_CONFIG = config["train"]
DATA_CONFIG = config["dataset"]

batchsize = TRAIN_CONFIG["batchsize"]
NUM_CLASSES = TRAIN_CONFIG["num_classes"]
RANKS =TRAIN_CONFIG["ranks"]
######################## Get Datasets & Dataloaders ###########################
transforms_list = [
        transforms.RandomCrop((256,256)),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

train_dataset = Flick30k_triplet_dataset(
    **rm_dict(DATA_CONFIG, ("word2vec")),
    transfroms_list=transforms_list)
train_dataloader = DataLoader(train_dataset, batch_size= batchsize,shuffle=True)

eval_train = Flick303k_eval_dataset(**rm_dict(DATA_CONFIG, ("word2vec")), samples=10000,mode = 'train')
eval_val = Flick303k_eval_dataset(**rm_dict(DATA_CONFIG, ("word2vec")),mode = 'val')

transforms_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

##################################### Import models ###########################
ROOT_PATH = 'saved/S1'
Image_Generator_Path = os.path.join(ROOT_PATH, 'imgcnn_epoch210.pth')
Text_Generator_Path = os.path.join(ROOT_PATH, 'textcnn_epoch210.pth')
Identity_Classifier_Path = os.path.join(ROOT_PATH, 'id_classifier_epoch210.pth')

imagecnn = imageCNN()
textcnn = textCNN(word2vec= None)
id_classifier = IdClassifier()

## Initialization ##
imagecnn.load_state_dict(torch.load(Image_Generator_Path))
textcnn.load_state_dict(torch.load(Text_Generator_Path))
id_classifier.load_state_dict(torch.load(Identity_Classifier_Path))

if torch.cuda.is_available():
    imagecnn.cuda()
    textcnn.cuda()
    id_classifier.cuda()

############################# Get Losses & Optimizers #########################
criterion_identity = torch.nn.CrossEntropyLoss()
criterion_modality = torch.nn.BCEWithLogitsLoss()

# 第一次实验训练的时候，没有加L2正则化
# 实验现象表明已经预训练一部分的ImageCNN收敛要快一点，
# 下次的更新步长是不是可以考虑调小一点
optimizer_G = torch.optim.SGD([
    {'params': imagecnn.parameters(),'lr':1e-3},
    {'params': textcnn.parameters(), 'lr': 2e-3},
    {'params': id_classifier.parameters(), 'lr': 1e-3},
], momentum=0.9, weight_decay = 1e-4)

############################# Hyper-parameters ################################
# the loss combination
alpha = 1.0
beta = 1.0
gamma = 0.05
K = 5
nu = 1

# train parameters
epochs = 2000
eval_epochs = 10
################################  Train  ######################################
# Loss plot
logger = Logger(n_epochs = epochs, eval_epochs = eval_epochs,has_epoch= 210,batches_epoch= len(train_dataloader))

models = {'imagecnn':imagecnn, 'textcnn':textcnn}
val_img2text_rank, val_text2img_rank = evalRank(models, data = eval_val, transforms_list = transforms_test, verbose=True)
train_img2text_rank ,train_text2img_rank = evalRank(models,data= eval_train, transforms_list = transforms_test, verbose= True)

curr_rank = 0.0
curr_medr = 100

prev_scores = None
prev_loss_G = None
noChanged = 0

for epoch in range(210,epochs):
    print("Epoch ----------", epoch + 1)

    for i,batch in enumerate(train_dataloader):
        first_image, first_text, first_label, \
        second_image, second_text, second_label, \
        modality_image, modality_text = batch

        # Adjust data type
        first_text = first_text.long()
        first_label = first_label.long()

        # Change onto One-Hot code
        #first_label = torch.zeros(first_text.size(0),NUM_CLASSES).scatter_(1, first_label,1).long()
        #second_label = torch.zeros(second_text.size(0),NUM_CLASSES).scatter_(1,second_label,1).long()

        if torch.cuda.is_available():
            first_image = first_image.cuda()
            first_text = first_text.cuda()
            first_label = first_label.cuda()

        optimizer_G.zero_grad()
        # Generator
        first_image_features = imagecnn(first_image)
        first_text_features = textcnn(first_text)

        # Compute Loss
        # ID Classifier Loss
        predicted_id_first_image = id_classifier(first_image_features)
        predicted_id_first_text = id_classifier(first_text_features)

        identity_image_loss = criterion_identity(predicted_id_first_image, first_label)
        identity_text_loss = criterion_identity(predicted_id_first_text, first_label)
        identity_loss = identity_image_loss + identity_text_loss

        loss_G = beta * identity_loss

        loss_G.backward()

        optimizer_G.step()
        logger.log(
            # Add loss
            losses={
                'image_loss': identity_image_loss,
                'text_loss': identity_text_loss,
                'loss_G':loss_G,
            },
            ## Add metrics
            metrics={
                'train_img2txt_r5':train_img2text_rank[1],
                'train_txt2img_r5':train_text2img_rank[1],
                'val_img2txt_r5': val_img2text_rank[1],
                'val_txt2img_r5': val_text2img_rank[1],
            }
        )
    if epoch % eval_epochs == 0:
        val_img2text_rank, val_text2img_rank = evalRank(models, data=eval_val, transforms_list=transforms_test)
        train_img2text_rank, train_text2img_rank = evalRank(models, data=eval_train, transforms_list=transforms_test)

        curr_rank_scores = val_img2text_rank[0] + val_img2text_rank[1] + val_img2text_rank[2]
        curr_medr_scores = val_img2text_rank[3]

    # NO CHANGED
    if prev_scores and curr_rank_scores < prev_scores:
        noChanged += 1
    else:
        noChanged = 0
    if noChanged >= 10:
        break
    else:
        prev_scores = curr_rank_scores
        prev_loss_G = loss_G.data.cpu()

    if epoch % eval_epochs == 0:
        if curr_rank_scores > curr_rank or curr_medr_scores < curr_medr:
            curr_rank = curr_rank_scores
            curr_medr = curr_medr_scores
            torch.save(imagecnn.state_dict(), 'saved/S1/imgcnn_epoch' + str(epoch) + '.pth')
            torch.save(textcnn.state_dict(), 'saved/S1/textcnn_epoch' + str(epoch) + '.pth')
            torch.save(id_classifier.state_dict(), 'saved/S1/id_classifier_epoch' + str(epoch) + '.pth')
            # torch.save(mode_classifier.state_dict(), 'saved/S1/mode_classifier_epoch' + str(epoch) + '.pth')
        with open(os.path.join('saved', 'S1_stats_ep' + '.txt'), 'a') as file:
            for rank, item_i2t, item_t2i in zip(RANKS, val_img2text_rank, val_text2img_rank):
                # rank-1 rank-5 rank-10 median-rank
                file.write("{},{},".format(item_i2t, item_t2i))
            file.write("\n")

