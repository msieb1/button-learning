import torch as th
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.autograd import Function
import torchvision.models as models
from copy import deepcopy as copy

from pdb import set_trace as st

class BatchNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BatchNormConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm(x)
        return F.relu(x, inplace=True)

class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x    


class RGBClassifier(nn.Module):
    
    def __init__(self):
        super(RGBClassifier, self).__init__()
        self.Conv2d_1a = BatchNormConv2d(3, 32, kernel_size=3, stride=1)
        self.Conv2d_2a = BatchNormConv2d(32, 32, kernel_size=3, stride=1)
        self.Conv2d_3a = BatchNormConv2d(32, 16, kernel_size=3, stride=1)
        self.Conv2d_4a = BatchNormConv2d(16, 16, kernel_size=3, stride=1)
        self.Conv2d_5a = BatchNormConv2d(16, 8, kernel_size=3, stride=1)
        
        self.FullyConnected_6a = Dense(8 * 90 * 90 , 1)
        
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.shape[1] == 4:
            x = x[:, :-1].clone()
        else:
            x = x.clone()
        x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5       

        x = self.Conv2d_1a(x)
        x = self.Conv2d_2a(x)
        x = self.Conv2d_3a(x)
        x = self.Conv2d_4a(x)
        x = self.Conv2d_5a(x)
        x = x.view(x.size()[0], -1)
        x = self.FullyConnected_6a(x)
        probs = self.Sigmoid(x)
        return x

class DepthClassifier(nn.Module):
    def __init__(self):
        super(DepthClassifier, self).__init__()
        self.Conv2d_1a = BatchNormConv2d(1, 32, kernel_size=3, stride=1)
        self.Conv2d_2a = BatchNormConv2d(32, 64, kernel_size=3, stride=1)
        self.Conv2d_3a = BatchNormConv2d(64, 64, kernel_size=3, stride=1)
        self.FullyConnected_4a = Dense(7 * 7 * 64, 1, activation=nn.Sigmoid())

    def forward(self, x):
        if x.shape[1] == 4:
            x = x[:, :-1].clone()
        else:
            x = x.clone()
        x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5       

        x = self.Conv2d_1a(x)
        x = self.Conv2d_2a(x)
        x = self.Conv2d_3a(x)
        x = x.view(x.size()[0], -1)
        x = self.FullyConnected_4a(x)
        return x


class GreyscaleClassifier(nn.Module):
    def __init__(self):
        super(GreyscaleClassifier, self).__init__()
        self.Conv2d_1a = BatchNormConv2d(1, 32, kernel_size=3, stride=1)
        self.Conv2d_2a = BatchNormConv2d(32, 64, kernel_size=3, stride=1)
        self.Conv2d_3a = BatchNormConv2d(64, 64, kernel_size=3, stride=1)
        self.FullyConnected_4a = Dense(7 * 7 * 64, 1, activation=nn.Sigmoid)

    def forward(self, x):
        if x.shape[1] == 4:
            x = x[:, :-1].clone()
        else:
            x = x.clone()
        x[:, 0] = x[:, 0] * ((0.229 + 0.224 + 0.225) / 3.0 / 0.5) + ((0.485 + 0.456 + 0.406) / 3.0 - 0.5) / 0.5

        x = self.Conv2d_1a(x)
        x = self.Conv2d_2a(x)
        x = self.Conv2d_3a(x)
        x = x.view(x.size()[0], -1)
        x = self.FullyConnected_4a(x)
        return x


class BinaryEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(BinaryEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.Conv2d_1a = BatchNormConv2d(3, 16, kernel_size=3, stride=1)
        self.Conv2d_2a = BatchNormConv2d(16, 32, kernel_size=3, stride=1)
        self.Conv2d_3a = BatchNormConv2d(32, 64, kernel_size=3, stride=1)
        self.FC_4a = Dense(26 * 26 * 64, emb_dim, activation=nn.Sigmoid())

    def forward(self, x):
        if x.shape[1] == 4:
            x = x[:, :-1].clone()
        else:
            x = x.clone()
        x[:, 0] = x[:, 0] * ((0.229 + 0.224 + 0.225) / 3.0 / 0.5) + ((0.485 + 0.456 + 0.406) / 3.0 - 0.5) / 0.5

        x = self.Conv2d_1a(x)
        x = self.Conv2d_2a(x)
        x = self.Conv2d_3a(x)
        x = x.view(x.size()[0], -1)
        x = self.FC_4a(x)
        return x    



class AttributeClassifier(nn.Module):
    def __init__(self, emb_dim, num_attr):
        super(AttributeClassifier, self).__init__()
        self.emb_dim = emb_dim
        self.num_attr = num_attr
        self.Conv2d_1a = BatchNormConv2d(3, 16, kernel_size=3, stride=1)
        self.Conv2d_2a = BatchNormConv2d(16, 32, kernel_size=3, stride=1)
        self.Conv2d_3a = BatchNormConv2d(32, 64, kernel_size=3, stride=1)

        self.FC_4a = Dense(26 * 26 * 64, emb_dim*num_attr)
        self.pool1d = nn.MaxPool1d(kernel_size=emb_dim)
        
        self.FC_5a = Dense(num_attr, 1, activation=nn.Sigmoid())

        # self.FC_5a = Dense(emb_dim*num_attr, 1, activation=nn.Sigmoid())


    # def forward(self, x):
    #     if x.shape[1] == 4:
    #         x = x[:, :-1].clone()
    #     else:
    #         x = x.clone()
    #     x[:, 0] = x[:, 0] * ((0.229 + 0.224 + 0.225) / 3.0 / 0.5) + ((0.485 + 0.456 + 0.406) / 3.0 - 0.5) / 0.5

    #     x = self.Conv2d_1a(x)
    #     x = self.Conv2d_2a(x)
    #     x = self.Conv2d_3a(x)
    #     x = x.view(x.size()[0], -1)
    #     emb = self.FC_4a(x)
    #     x = emb.unsqueeze(-2)
    #     x = self.pool1d(x)
    #     x = x.squeeze(-2)
    #     x = self.FC_5a(x)
    #     return x, emb   


    def forward(self, x):
        if x.shape[1] == 4:
            x = x[:, :-1].clone()
        else:
            x = x.clone()
        x[:, 0] = x[:, 0] * ((0.229 + 0.224 + 0.225) / 3.0 / 0.5) + ((0.485 + 0.456 + 0.406) / 3.0 - 0.5) / 0.5

        x = self.Conv2d_1a(x)
        x = self.Conv2d_2a(x)
        x = self.Conv2d_3a(x)
        x = x.view(x.size()[0], -1)
        emb = self.FC_4a(x)
        emb = emb.view(-1, self.num_attr, self.emb_dim)
        emb = F.normalize(emb, p=2, dim=-1)
        x = emb.view(-1, self.num_attr*self.emb_dim)
   
        x = x.unsqueeze(-2)
        x = self.pool1d(x)
        x = x.squeeze(-2)
    
        x = self.FC_5a(x)
        return x, emb   