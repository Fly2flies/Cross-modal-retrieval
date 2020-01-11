import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def ImageGenerator(pretrained = True):
    model = models.resnet50(pretrained = pretrained)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    image_extractor = nn.Sequential(*list(model.children())[:-1])
    return image_extractor

class imageCNN(nn.Module):
    def __init__(self,
                 input_size = 2048,
                 hidden_size = 2048,
                 output_size = 2048,
                 StageI = True):
        super(imageCNN, self).__init__()
        # 除去最后分类层,再添加两层FC层
        self.res = ImageGenerator(pretrained = True)
        if StageI:
            for weights in self.res.parameters():
                weights.requires_grad_(False)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(0.75)

    def forward(self, x):
        x = self.res(x) # (N,2048,1,1)
        x = x.view(x.size(0), -1) # (N,2048)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        return self.dp(x)

if __name__ == '__main__':
    image = torch.rand([2,3,224,224])
    feature_extractor = imageCNN()
    output = feature_extractor(image)
    print(output.size())