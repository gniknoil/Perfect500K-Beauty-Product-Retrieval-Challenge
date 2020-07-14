import os
import argparse
import numpy as np
from PIL import Image
import shutil

from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import pickle
import warnings
from tqdm import tqdm
from pooling import *
from training_dataset import retrieval_dataset
import net2

transform_480 = transforms.Compose([
    transforms.Resize((480,480)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

batch_size=100

model_path={
    'resnext101':'pretrained/resnext101.pth'
}
feature_length={
    'resnext101':2048
}

if __name__ == "__main__":
    os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    name_list=os.listdir('../PF-500K')
    name_list.sort()
    print(len(name_list))
    mode='cuda' # or 'cpu'

    for model_name in ['resnext101']:
        model=net2.__dict__[model_name](model_path[model_name])
        if mode=='cuda':
            model=model.cuda()
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True

        dataset = retrieval_dataset('../PF-500K',transform=transform_480)
        testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        feat_dict={
            'ramac':torch.empty(len(name_list),feature_length[model_name]),
            'Grmac':torch.empty(len(name_list),feature_length[model_name]),
            'AMAC':torch.empty(len(name_list),feature_length[model_name])
        }

        print(feat_dict['ramac'].size())
        img_list=[]
        model.eval()
        with torch.no_grad():
            for i, (inputs, names) in tqdm(enumerate(testloader)):
                inputs = inputs.to(mode)
                feature_rmac,feature_ramac,feature_Grmac,feature_Mac,feature_SPoC,feature_AMAC = model(inputs)
                
                feat_dict['ramac'][i*batch_size:i*batch_size+len(names),:]=feature_ramac.cpu()
                feat_dict['Grmac'][i*batch_size:i*batch_size+len(names),:]=feature_Grmac.cpu()
                feat_dict['AMAC'][i*batch_size:i*batch_size+len(names),:]=feature_AMAC.cpu()
                
                assert name_list[i*batch_size:i*batch_size+len(names)]==list(names)
                img_list.extend(names)
        
        with open("./feature/feat_{}.pkl".format(model_name), "wb") as file_to_save:
            pickle.dump(
                {
                'name':img_list,
                'ramac':feat_dict['ramac'].half().numpy(),
                'Grmac':feat_dict['Grmac'].half().numpy(),
                'AMAC':feat_dict['AMAC'].half().numpy()
                    }, 
                file_to_save,
                -1
                )