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
save_list.append('Using resnet34 saliency network with skip last')
save_list.append(t)
save_list.append(sys.argv[0])


# Data paths
path = '/work3/s144137/NTU/project/data/ImageNet'
train_annotation = 'train2019.json'
test_annotation = 'val2019.json'
images = 'train_val2019'


#Values for commmand line
epochs = int(sys.argv[1])
N_train_with_blur = int(sys.argv[2])
batch_size = int(sys.argv[3])
decimate_factor = float(sys.argv[4])
if len(sys.argv) == 6:
    st_path = sys.argv[5]
else:
    st_path = None

save_list.append(N_train_with_blur)
save_list.append(batch_size)
save_list.append(decimate_factor)

train_loader, test_loader = get_dataloaders(path,train_annotation,test_annotation,images,batch_size,800,decimate_factor=decimate_factor)


# Instantiate models
task_network = resnet.resnet101()
saliency_network = saliency_network.saliency_network_resnet34()
model = saliency_sampler.Saliency_Sampler(task_network,saliency_network,224,224).cuda()
model = load_st(model,st_path)
opt = Adam(model.parameters())
loss = nn.CrossEntropyLoss().cuda()

model.train()
for epoch in range(epochs):
    t1 = time.time()
    epoch_loss = 0
    for i, batch in enumerate(train_loader):

        im, label,_ = batch
        im = Variable(im).cuda()
        label = Variable(label).long().cuda()
        if epoch > N_train_with_blur:
            p = 1
        else:
            p = 0
        try:
            out,_,_ = model(im,p)
            L = loss(out,label)
            L.backward()

            opt.step()
            opt.zero_grad()

            epoch_loss += L.item()
        except Exception as e:
            print(repr(e))
            print(i)
            continue
        if i == 1:
            pass
    t2 = time.time()
    print('Elapsed time is ',t2-t1)
    print('Epoch ',epoch+1,' loss is ',epoch_loss/(i+1))
    save_list.append('Epoch '+str(epoch+1)+' loss is '+str(epoch_loss/(i+1)))
    save_list.append('Elapsed time is '+str(t2-t1))


print('Testing model')
save_list.append('Testing model')
test_loss = 0.0
test_err = 0.0
model.eval()
t1 = time.time()
for i, batch in enumerate(test_loader):
    im,label,_ = batch
    im = Variable(im,requires_grad = False).cuda()
    label = Variable(label,requires_grad = False).long().cuda()
    out,_,_ = model(im,1)
    L = loss(out,label)
    test_loss += L.item()
    if out.argmax() != label:
        test_err += 1.0
    if i == 1:
        pass
t2 = time.time()


print('Testing elapsed time is ', t2-t1)
print('Test loss is ',test_loss/(i+1))
print('Test accuracy is ',1-(test_err/(i+1)))
save_list.append('Testing elapsed time is '+str(t2-t1))
save_list.append('Test loss is '+str(test_loss/(i+1)))
save_list.append('Test accuracy is '+str(1.0-(test_err/(i+1))))


results_dir = '/zhome/45/0/97860/Documents/NTU/SaliencySamplerProject/trained_models/'
digits = string.digits
#file_id = ''.join(random.choice(digits) for i in range(5))
file_id = now.strftime("%d%m%Y%H%M%S")
with open(os.path.join(results_dir,'state_dict'+file_id+'.pkl'),'wb') as fp:
    pickle.dump(model.state_dict(),fp)
with open(os.path.join(results_dir,'results'+file_id+'.txt'),'w') as fp:
    for item in save_list:
        fp.write("%s\n" % item)


