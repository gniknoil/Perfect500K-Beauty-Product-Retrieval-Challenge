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
    def __init__(self,model_path,feature_name):
        super(resnext101, self).__init__()
        model = model_resnext101()
        self.backbone=model.backbone

        checkpoint = torch.load(model_path)['net']
        self.backbone.load_state_dict(checkpoint)
        
        self.norm=L2N()
        
        self.feature_name=feature_name
        if self.feature_name=='rmac':
            self.rmac=Rmac_Pooling()
        if self.feature_name=='ramac':
            self.ramac=Ramac_Pooling()
        if self.feature_name=='Grmac':
            self.Grmac=Grmac_Pooling(p=3.5)
        if self.feature_name=='Mac':
            self.Mac=Mac_Pooling()
        if self.feature_name=='SPoc':
            self.SPoc=SPoC_pooling()
        if self.feature_name=='AMAC':
            self.AMAC=AMAC_Pooling()

    def forward(self,data):
        feature=self.backbone(data)
        if self.feature_name=='rmac':
            feature=self.rmac(feature)
        if self.feature_name=='ramac':
            feature=self.ramac(feature)
        if self.feature_name=='Grmac':
            feature=self.Grmac(feature)
        if self.feature_name=='Mac':
            feature=self.Mac(feature)
        if self.feature_name=='SPoc':
            feature=self.SPoc(feature)
        if self.feature_name=='AMAC':
            feature=self.AMAC(feature)
        feature=self.norm(feature)
        return feature

