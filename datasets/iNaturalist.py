import json
import numpy as np
import random
from os.path import join
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL.Image import open as open_
from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True


random.seed(1) # For reproducibility when using decimation




def imread(path_to_im):
    im = open_(path_to_im).convert(mode='RGB')
    
    return im
    
    

class iNaturalist(Dataset):
    def __init__(self,root,annotations,images,transform,decimate_factor=None):
        
        self.root = root
        self.images = images
        
        ann_path = join(root,annotations)
        with open(ann_path) as file: 
            data = json.load(file)
        self.ims = [x['file_name'] for x in data['images']]
        self.im_ids = [x['id'] for x in data['images']]  
        self.classes = [x['category_id'] for x in data['annotations']]

        self.transform = transform

        if decimate_factor:
            requested_length = int(len(self.ims)*decimate_factor)
            idx = random.sample(range(0,len(self.ims)),requested_length)
            self.ims = [self.ims[i] for i in idx]
            self.im_ids = [self.im_ids[i] for i in idx]
            self.classes = [self.classes[i] for i in idx]

        
    def __len__(self):
        return len(self.ims)
        
        
    def __getitem__(self,idx):
        path_im = join(self.root,self.ims[idx])
        im = imread(path_im)
        target = self.classes[idx]
        im_id = self.im_ids[idx]
        if self.transform:
            im = self.transform(im)

        return im,target,im_id



    


def get_dataloaders(root,train_annotations,test_annotations,images,batch_size,sampling_train,sampling_test=None,decimate_factor=None):


    transform = transforms.Compose([transforms.Resize(size=(sampling_train,sampling_train)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    data = iNaturalist(root,train_annotations,images,transform,decimate_factor)
    train_loader = DataLoader(data,batch_size=batch_size,shuffle=True,num_workers=8,pin_memory=True)
    # Test loader also needs transform for normalization and conversion to tensor

    transform = transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])


    if sampling_test:
        transform = transforms.Compose([transforms.Resize(size=(sampling_test,sampling_test)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

    data = iNaturalist(root,test_annotations,images,transform)
    test_loader = DataLoader(data,batch_size=1,shuffle=False)

    return train_loader, test_loader





