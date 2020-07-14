import sys
import os
import csv
from tqdm import tqdm
import numpy as np
import pickle
import warnings

import torch
import torchvision
import torchvision.models as models

from training_dataset import retrieval_dataset
from pooling import *
from features_resnext import transform_480
# from net import seresnet152, densenet201
from net_resnext import resnext101, densenet201
from map import map_calculate


def load_feature(feat_name):
    with open(feat_name, "rb") as file_to_read:
        feature=pickle.load(file_to_read)
    name=feature['name']
    return name,feature

if __name__ == "__main__":

    os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'

    test_image_path='./test'
    result_path='./result/result.csv'

    resnext101_feature_path='./feature/Tianchi/feat_resnext101.pkl'
    resnext101_path='./pretrained/resnext101.pth'


    name_list,resnext101_feature=load_feature(resnext101_feature_path)
    feature={'resnext101':resnext101_feature}

    feat_type={
        'resnext101':['Grmac','AMAC']
        }
    dim_feature={
        'resnext101':2048
    }
    batch_size=20

    similarity=torch.zeros(len(os.listdir(test_image_path)),len(name_list))
    print(similarity.size())

    for model_name in ['resnext101']:
        feature_model=feature[model_name]
        query_class={}
        for item in feat_type[model_name]:
            if model_name == 'resnext101':
                model=resnext101(resnext101_path,item)
            else:
                pass

            feat_reserved=feature_model[item]

            dataset = retrieval_dataset(test_image_path,transform=transform_480)
            testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            model=model.cuda()
            model.eval()
            query=torch.empty(len(os.listdir(test_image_path)),dim_feature[model_name])
            query_dict={}
            name_test=[]
            with torch.no_grad():
                for i, (inputs, names) in tqdm(enumerate(testloader)):
                    result=model(inputs.cuda()).cpu()
                    query[i*batch_size:i*batch_size+len(names)] = result
                    name_test.extend(names)

                    temp_idx=list(range(i*batch_size,i*batch_size+len(names)))
                    for j in range(len(names)):
                        query_dict[names[j]]=temp_idx[j]

            feat_reserved=torch.Tensor(feat_reserved)
            feat_reserved_=feat_reserved.transpose(1,0)
            query=torch.Tensor(query)

            query_class[item]=query

            sim=torch.matmul(query,feat_reserved_)
            
            similarity+=sim

            _, predicted = similarity.topk(7)
            predicted=predicted.tolist()
            dict_result=dict(zip(name_test,predicted))
        
        ## QE
        sim2=torch.zeros(len(os.listdir(test_image_path)),len(name_list))

        for item in feat_type[model_name]:

            feat_reserved=feature_model[item]
            feat_reserved=torch.Tensor(feat_reserved)
            feat_reserved_=feat_reserved.transpose(1,0)

            query2=query_class[item]
            ### TopK ###
            for item_ in dict_result:
                temp_list=dict_result[item_]
                for k in range(2):
                    idx=temp_list[k]
                    query2[query_dict[item_]]+=feat_reserved[idx]*(2-k)/3
            
            sim2+=torch.matmul(query2,feat_reserved_)
        _, predicted = sim2.topk(7)
        predicted=predicted.tolist()
        dict_result=dict(zip(name_test,predicted))

        ## saving csv
        img_results=[]
        name_test.sort()
        for name in name_test:
            temp=[name.split('.')[0]]
            for idx in dict_result[name]:
                temp.append(name_list[idx].split('.')[0])
            img_results.append(temp)
        print('saving')

        with open('./result/{}_{}_result.csv'.format(model_name,item),'w') as out:
            csv_write = csv.writer(out)
            csv_write.writerows(img_results)

