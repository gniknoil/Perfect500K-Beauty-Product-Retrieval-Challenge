from pretrainedmodels import models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from pooling import *

__all__=['L2N','resnext101']
#---------------feature extraction------------------#

class L2N(nn.Module):
    
    def __init__(self, eps=1e-6):
        super(L2N,self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'

class model_resnext101(nn.Module):
    def __init__(self):
        super(model_resnext101,self).__init__()

        model=models.resnext101_32x4d()
        self.backbone = nn.Sequential(*list(model.children())[0])
    
    def forward(self, data):
        features = self.backbone(data)

        return features

class resnext101(nn.Module):
    def __init__(self,model_path):
        super(resnext101, self).__init__()
        model = model_resnext101()
        self.backbone=model.backbone

        checkpoint = torch.load(model_path)['net']
        self.backbone.load_state_dict(checkpoint)
        self.norm=L2N()
        
        self.rmac=Rmac_Pooling()
        self.ramac=Ramac_Pooling()
        self.Grmac=Grmac_Pooling(p=3.5)
        self.Mac=Mac_Pooling()
        self.SPoC=SPoC_pooling()
        self.AMAC=AMAC_Pooling()

    def forward(self,data):
        feature=self.backbone(data)
        feature_rmac=self.norm(self.rmac(feature))
        feature_ramac=self.norm(self.ramac(feature))
        feature_Grmac=self.norm(self.Grmac(feature))
        feature_Mac=self.norm(self.Mac(feature))
        feature_SPoC=self.norm(self.SPoC(feature))
        feature_AMAC=self.norm(self.AMAC(feature))
        return feature_rmac,feature_ramac,feature_Grmac,feature_Mac,feature_SPoC,feature_AMAC
