import torch
import torch.nn as nn
import torch.nn.functional as F

class IdClassifier(nn.Module):
    """
    Image-Text Group Classifier
   """
    def __init__(self,
                 input_size = 2048,
                 hidden_size = 4096,
                 num_classes = 29784):
        super(IdClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size,num_classes)
    def forward(self, x):
        x = x.view(x.size(0),-1) # [N , 2048]
        x = self.relu(self.bn1(self.fc1(x)))
        out = self.fc(x)
        return out
class ModalityClassifier(nn.Module):
    def __init__(self,input_size = 2048,
                 hidden_size = 1024,
                 num_classes = 2):
        super(ModalityClassifier, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        x = x.view(x.size(0),-1)
        out = self.relu(self.fc1(x))
        out = self.bn1(out)
        out = self.fc2(out)
        return out