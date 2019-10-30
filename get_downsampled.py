import numpy as np
from datetime import datetime
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.autograd import Variable
from datasets.iNaturalist import get_dataloaders
from models import resnet, saliency_network, saliency_sampler
from load_util import load_st
torch.backends.cudnn.benchmark=True
from PIL import Image

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

unnorm = UnNormalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))

outputs = []
def hook(module,input_,output):
    outputs.append(output)

# Data paths
path = '/work3/s144137/NTU/project/data/ImageNet'
train_annotation = 'train2019.json'
test_annotation = 'val2019.json'
images = 'train_val2019'
st_path = '/zhome/45/0/97860/Documents/NTU/SaliencySamplerProject/trained_models_detached/state_dict26102019110003.pkl'



train_loader, test_loader = get_dataloaders(path,train_annotation,test_annotation,images,1,800,decimate_factor=1)
_,test_loader224 = get_dataloaders(path,train_annotation,test_annotation,images,1,800,sampling_test=224,decimate_factor=1)

# Instantiate models
task_network = resnet.resnet101()
saliency_network = saliency_network.saliency_network_resnet18()
model = saliency_sampler.Saliency_Sampler(task_network,saliency_network,224,224).cuda()
model = load_st(model,st_path)



model.conv_last.register_forward_hook(hook)

im_ids = np.random.randint(0,100,10)
im_ids = [98]
print(im_ids)

for i, (batch, batch224) in enumerate(zip(test_loader,test_loader224)):
    print(i)
    if i in im_ids:
        im, _,_ = batch
        im224,_,_ = batch224
        im_og = Variable(im,requires_grad = False).cuda()
        out = model(im_og,1)

        inter = outputs[0]
        xs = nn.functional.interpolate(inter,size=(model.grid_size,model.grid_size),mode='bilinear',align_corners=False)
        xs = xs.view(-1,model.grid_size*model.grid_size)
        xs = nn.Softmax(dim=1)(xs)
        xs = xs.view(-1,1,model.grid_size,model.grid_size)   
        xs_hm = nn.ReplicationPad2d(model.padding_size)(xs)

        grid = model.create_grid(xs_hm)

        x_sampled = nn.functional.grid_sample(im_og, grid)





        x_sampled = unnorm(x_sampled).detach().cpu().squeeze().permute(1,2,0).numpy()
        x_sampled = x_sampled/x_sampled.max()
        x_sampled = x_sampled*255.0



        print(x_sampled.min(),x_sampled.max())
        im = Image.fromarray(x_sampled.astype('uint8'))
        im.save('intermediate.png')
        im_og = unnorm(im_og)
        im_og = im_og.detach().cpu().squeeze().permute(1,2,0).numpy()
        
        im_og = im_og/im_og.max()
        im_og = im_og*255.0
        im_og = Image.fromarray(im_og.astype('uint8'))
        im_og.save('Original.png')

        im224 = unnorm(im224)
        im224 = im224.detach().cpu().squeeze().permute(1,2,0).numpy()
        im224 = im224/im224.max()
        im224 = im224*255.0
        im224 = Image.fromarray(im224.astype('uint8'))
        im224.save('Resized.png')
    if i > max(im_ids):
        break


