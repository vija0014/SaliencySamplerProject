import numpy as np
import pickle
import os
import string
import random
import time
import sys
from datetime import datetime
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.autograd import Variable
from datasets.iNaturalist import get_dataloaders
from models import resnet, saliency_network, saliency_sampler
from load_util import load_st
torch.backends.cudnn.benchmark=True


# Create a log list. Append all items you want logged
# to a txt file in the order that you want them displayed.
# Append start date and time
save_list = []
now = datetime.now()
now = now.replace(hour=now.hour+6) #Change to Singapore time
print("now =", now)
# dd/mm/YY H:M:S
t = now.strftime("%d/%m/%Y %H:%M:%S")
save_list.append(t)
save_list.append(sys.argv[0])


# Data paths
path = '/work3/s144137/NTU/project/data/ImageNet'
test_annotation = 'val2019.json'
train_annotation = 'train2019.json'
images = 'train_val2019'
st_path_sampling = '/zhome/45/0/97860/Documents/NTU/SaliencySamplerProject/trained_models_detached/state_dict26102019110003.pkl'
st_path_no_sampling = '/zhome/45/0/97860/Documents/NTU/SaliencySamplerProject/trained_models_detached/state_dict26102019110009.pkl'




_, test_loader_800 = get_dataloaders(path,train_annotation,test_annotation,images,1,800)
_, test_loader_224 = get_dataloaders(path,train_annotation,test_annotation,images,1,224,sampling_test=224)

# Instantiate models
task_network = resnet.resnet101()
saliency_network = saliency_network.saliency_network_resnet18()
model_sampling = saliency_sampler.Saliency_Sampler(task_network,saliency_network,224,224).cuda()
model_sampling = load_st(model_sampling,st_path_sampling)
	
model_no_sampling = resnet.resnet101().cuda()
model_no_sampling = load_st(model_no_sampling,st_path_no_sampling)

model_sampling.eval()
test_suc1 = 0
test_suc5 = 0
t = 0
for i, batch in enumerate(test_loader_800):
    im, target,_ = batch
    im = Variable(im,requires_grad=False).cuda()
    target = Variable(target,requires_grad=False).long().cuda()
    t1 = time.time()
    out,_,_ = model_sampling(im,1)
    t2 = time.time()
    t += (t2-t1)
    if out.argmax() == target:
        test_suc1 += 1
    if target.tolist()[0] in out.topk(5)[1].tolist()[0]:
        test_suc5 += 1
print('Test accuracy 1 % with sampling ',test_suc1/(i+1))
print('Test accuracy 5 % with sampling ',test_suc5/(i+1))
print('Average forward pass time ', t/(i+1))

model_no_sampling.eval()
test_suc1 = 0
test_suc5 = 0
t = 0
for i, batch in enumerate(test_loader_224):
    im, target,_ = batch
    im = Variable(im,requires_grad=False).cuda()
    target = Variable(target,requires_grad=False).cuda() 
    t1 = time.time()
    out = model_no_sampling(im)
    t2 = time.time()
    t += (t2-t1)
    if out.argmax() == target:
        test_suc1 += 1
    if target.tolist()[0] in out.topk(5)[1].tolist()[0]:
        test_suc5 += 1

print('Test accuracy 1 % without sampling ', test_suc1/(i+1))
print('Test accuracy 5 % without sampling ', test_suc5/(i+1))
print('Average forward pass time ', t/(i+1))



