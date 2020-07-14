import os
import pickle
from PIL import Image
import torch
import torch.utils.data as data
import numpy as np

def conversion(img, angle=None):
    #img: PIL format
    img=np.array(img)
    h,w,c=img.shape
    if h>w:
        img2=np.ones((h,h,c),dtype='uint8')*255
        start=(h-w)//2
        img2[:,start:start+w,:]=img
    elif w>h:
        img2=np.ones((w,w,c),dtype='uint8')*255
        start=(w-h)//2
        img2[start:start+h,:,:]=img
    else:
        img2=img
    img2=Image.fromarray(img2)
    if angle!=None:
        img2=img2.transpose(angle)
    return img2

class retrieval_dataset(data.Dataset):
    def __init__(self,root_path,transform = None,crop_aug=False, rotation='0'):
        self.root=root_path
        self.image_list=os.listdir(self.root)
        self.image_list.sort()

        # self.image_list=self.image_list[:1000]

        self.transform=transform
        self.rotation=rotation
        self.rot_dict={
            '0':None,
            '90':Image.ROTATE_90,
            '180':Image.ROTATE_180,
            '270':Image.ROTATE_270,
            'H':Image.FLIP_LEFT_RIGHT,
            'V':Image.FLIP_TOP_BOTTOM
        }
    
    def __getitem__(self,idx):
        image=Image.open(os.path.join(self.root,self.image_list[idx])).convert('RGB')
        image=conversion(image, self.rot_dict[self.rotation])
        output_imgs=self.transform(image)
        return output_imgs, self.image_list[idx]

    def __len__(self):
        return len(self.image_list)
