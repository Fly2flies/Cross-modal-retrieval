"""
training with all loss
"""
from DataSet.flick30k_dataset import Flick30k_triplet_dataset
from DataSet.flick30k_dataset import Flick303k_eval_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import pprint
import itertools
from Models.models import IdClassifier, ModalityClassifier
from Models.ImageCNN import imageCNN
from Models.TextCNN import textCNN
from evaluation import evalRank

from logger import  Logger
from config import config
from tools import rm_dict
import os

######################## Preprocessing parameter ###########################
STAGE1 = False
os.environ["CUDA_VISIBLE_DEVICES"]= config["CUDA_VISIBLE_DEVICES"]
TRAIN_CONFIG = config["train"]
DATA_CONFIG = config["dataset"]

batchsize = TRAIN_CONFIG["batchsize"]
NUM_CLASSES = TRAIN_CONFIG["num_classes"]
RANKS =TRAIN_CONFIG["ranks"]
eval_epochs = TRAIN_CONFIG["eval_epoch"]

## Model path ##
ROOT_PATH = 'saved/S1'
Image_Generator_Path = os.path.join(ROOT_PATH, 'imgcnn_epoch140.pth')
Text_Generator_Path = os.path.join(ROOT_PATH, 'textcnn_epoch140.pth')
Identity_Classifier_Path = os.path.join(ROOT_PATH, 'id_classifier_epoch140.pth')

######################## Get Datasets & Dataloaders ###########################
transforms_list = [
        transforms.RandomCrop((256,256)),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
transforms_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = Flick30k_triplet_dataset(
    **rm_dict(DATA_CONFIG, ("word2vec")),
    transfroms_list=transforms_list,
    randomTriplet=False,
)
train_dataloader = DataLoader(train_dataset, batch_size= batchsize,shuffle=True,num_workers=4)

eval_train = Flick303k_eval_dataset(**rm_dict(DATA_CONFIG, ("word2vec")), samples=10000,mode = 'train')
eval_val = Flick303k_eval_dataset(**rm_dict(DATA_CONFIG, ("word2vec")),mode = 'val')

##################################### Import models ###########################
imagecnn = imageCNN(StageI = STAGE1)
textcnn = textCNN(word2vec = None)
id_classifier = IdClassifier()
mode_classifier = ModalityClassifier()

## Initialization ##
imagecnn.load_state_dict(torch.load(Image_Generator_Path))
textcnn.load_state_dict(torch.load(Text_Generator_Path))
id_classifier.load_state_dict(torch.load(Identity_Classifier_Path))

if torch.cuda.is_available():
    imagecnn.cuda()
    textcnn.cuda()
    id_classifier.cuda()
    mode_classifier.cuda()

############################# Get Losses & Optimizers #########################
criterion_rank = torch.nn.MarginRankingLoss(margin = 1.0)
criterion_identity = torch.nn.CrossEntropyLoss()
criterion_modality = torch.nn.BCEWithLogitsLoss()

optimizer_G = torch.optim.SGD([
    {'params': imagecnn.parameters(),'lr':1e-3},
    {'params': textcnn.parameters(), 'lr': 1.1e-3},
    {'params': id_classifier.parameters(), 'lr': 1e-3},
], momentum = 0.9,weight_decay = 1e-4)

optimizer_D = torch.optim.SGD(mode_classifier.parameters(), lr = 1e-4, momentum = 0.9, weight_decay = 1e-4)

############################# Hyper-parameters ################################
# the loss combination
alpha = 1.0
beta = 1.0
gamma = 0.05
K = 5
nu = 1

# train parameters
epochs = 2000

################################  Train  ######################################
# Loss plot
logger = Logger(n_epochs = epochs,eval_epochs = eval_epochs,batches_epoch= len(train_dataloader))

models = {'imagecnn':imagecnn, 'textcnn':textcnn}
val_img2text_rank, val_text2img_rank = evalRank(models, data = eval_val, transforms_list = transforms_test, verbose = True)
train_img2text_rank ,train_text2img_rank = evalRank(models,data= eval_train, transforms_list = transforms_test, batchsize=32,verbose= True)

curr_rank = 0.0
curr_medr = 100

prev_scores = None
prev_loss_Total = None
noChanged = 0

for epoch in range(epochs):
    print("Epoch ----------", epoch + 1)

    for i,batch in enumerate(train_dataloader):
        first_image, first_text, first_label, \
        second_image, second_text, second_label, \
        modality_image, modality_text = batch

        # Adjust data type
        first_text = first_text.long()
        second_text = second_text.long()
        first_label = first_label.long()
        second_label = second_label.long()# [N,1]

        if torch.cuda.is_available():
            first_image = first_image.cuda()
            first_text = first_text.cuda()
            second_image = second_image.cuda()
            second_text = second_text.cuda()
            first_label = first_label.cuda()
            second_label = second_label.cuda()
            modality_image = modality_image.cuda()
            modality_text = modality_text.cuda()

        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        # Generator
        first_image_features = imagecnn(first_image)
        second_image_features = imagecnn(second_image)

        first_text_features = textcnn(first_text)
        second_text_features = textcnn(second_text)

        # Compute Loss
        # (1) Rank Loss with cisine distance
        image2text_pos = torch.cosine_similarity(first_image_features,first_text_features, dim = 1)
        image2text_neg = torch.cosine_similarity(first_image_features,second_text_features, dim = 1)

        text_image_pos = torch.cosine_similarity(first_text_features, first_image_features, dim = 1)
        text_image_neg = torch.cosine_similarity(first_text_features, second_image_features, dim = 1)

        labels = torch.ones(image2text_neg.size(0))
        if torch.cuda.is_available():
            labels = labels.cuda()

        triplet_loss_image2text = criterion_rank(image2text_pos, image2text_neg, labels)
        triplet_loss_text2image = criterion_rank(text_image_pos, text_image_neg, labels)

        triplet_loss = triplet_loss_image2text + triplet_loss_text2image

        # (2) ID Classifier Loss
        predicted_id_first_image = id_classifier(first_image_features)
        # predicted_id_second_image = id_classifier(second_image_features)

        predicted_id_first_text = id_classifier(first_text_features)
        # predicted_id_second_text = id_classifier(second_text_features)

        identity_image_loss = criterion_identity(predicted_id_first_image, first_label)
        identity_text_loss = criterion_identity(predicted_id_first_text, first_label)
        identity_loss = identity_image_loss + identity_text_loss

        loss_G = alpha * triplet_loss + beta * identity_loss

        # (3) Modality Classfier Loss
        predicted_first_image_modality = mode_classifier(first_image_features)
        predicted_first_text_modality = mode_classifier(first_text_features)
        modality_confused = (modality_image + modality_text) / 2 # [0.5,0.5]
        if epoch % K:
            # update Generator
            image_modality_loss = criterion_modality(predicted_first_image_modality, modality_confused)
            text_modality_loss = criterion_modality(predicted_first_text_modality, modality_confused)
        else:
            # update Discriminator
            image_modality_loss = criterion_modality(predicted_first_image_modality, modality_image)
            text_modality_loss = criterion_modality(predicted_first_text_modality, modality_text)

        loss_D = image_modality_loss + text_modality_loss

        # if epoch % K:
        #     # update Generator
        #     loss_total = loss_G - gamma * loss_D
        # else:
        #     # update Discriminator
        #     loss_total = nu * (gamma * loss_D - loss_G)

        loss_total = loss_G + gamma * loss_D
        loss_total.backward()

        if epoch % K:
            optimizer_G.step()
        else:
            optimizer_D.step()

        logger.log(
            # Add loss
            losses={
                'loss_D':loss_D,
                'loss_Triplet':triplet_loss,
                'loss_Identity':identity_loss
            },
            ## Add metrics
            metrics={
                'train_img2txt_r5':train_img2text_rank[1],
                'train_txt2img_r5':train_text2img_rank[1],
                'val_img2txt_r5': val_img2text_rank[1],
                'val_txt2img_r5': val_text2img_rank[1],
            }
        )
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
        prev_loss_G = loss_total.data.cpu()

    if not os.path.exists("./saved/S2"):
        os.mkdir("./saved/S2")

    if epoch % eval_epochs == 0:
        if curr_rank_scores > curr_rank or curr_medr_scores < curr_medr:
            curr_rank = curr_rank_scores
            curr_medr = curr_medr_scores
            torch.save(imagecnn.state_dict(), 'saved/S2/imgcnn_epoch' + str(epoch) + '.pth')
            torch.save(textcnn.state_dict(), 'saved/S2/textcnn_epoch' + str(epoch) + '.pth')
            torch.save(id_classifier.state_dict(), 'saved/S2/id_classifier_epoch' + str(epoch) + '.pth')
            torch.save(mode_classifier.state_dict(), 'saved/S2/mode_classifier_epoch' + str(epoch) + '.pth')

        with open(os.path.join('saved', 'S2_stats_ep' + '.txt'), 'a') as file:
            for rank, item_i2t, item_t2i in zip(RANKS, val_img2text_rank, val_text2img_rank):
                # rank-1 rank-5 rank-10 median-rank
                file.write("{},{},".format(item_i2t, item_t2i))
                file.write("\n")

