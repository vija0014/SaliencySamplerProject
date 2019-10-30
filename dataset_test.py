import numpy as np
import sys
from datasets.iNaturalist import iNaturalist, get_dataloaders
import pickle
from PIL import Image

path = '/work3/s144137/NTU/project/data/ImageNet'
annotation = 'train2019.json'
images = 'train_val2019'

dataloader,_ = get_dataloaders(path,annotation,annotation,images,1)
batch = next(iter(dataloader))
im, label, im_id = batch
'''
print(np.shape(im.numpy().squeeze()))
print(type(im.numpy().squeeze()[0,0,0]))
print(im.min(),im.max())
im = im.numpy()*255
im = im.astype('uint8').squeeze()
print(im.shape)
pil_im = Image.fromarray(np.moveaxis(im,0,2))
pil_im.save('test.jpeg')
'''

for batch in dataloader:
    im,_,_ = batch
    s = im.shape
    if s[1] != 3:
        print(1)


