import sys
sys.path.append('~/miniconda3/pkgs')

import argparse
import torch
import torch.nn as nn
import torchvision
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp
import matplotlib.pyplot as plt
import random

#from models.i3d_resnet import * 
from dataset.uav_human import UAVHumanDataSet


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
NUM_STEPS_STOP = 5555 #CSV1
INPUT_SIZE_SCALE = 1 #2 for RGB, 1 for Fisheye/Night
NUM_FRAMES = 16

DATA_DIRECTORY = 'PATH'
DATA_LIST_PATH = './dataset/UAVHuman/test.txt'

net = torch.load('snapshots/uavhuman_far/UAVHuman_10000.pth')
net = torch.nn.DataParallel(net).cuda() 


trainloader = data.DataLoader(
		UAVHumanDataSet(DATA_DIRECTORY, DATA_LIST_PATH, max_iters=NUM_STEPS_STOP,
					mean=IMG_MEAN, set='train', num_frames=NUM_FRAMES, input_size_scale=INPUT_SIZE_SCALE),
		batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
	
trainloader_iter = enumerate(trainloader)

accuracy = 0
top5_accuracy = 0

for i in range(0, NUM_STEPS_STOP):
	_, batch = trainloader_iter.__next__()
	images, labels = batch
	images = images.permute(0,2,1,3,4) 
	images = Variable(images).cuda()
	out = net(images)
	#out = out[:,:,0] #Only for X3D

	out = out.detach()
	outsort = torch.argsort(out,descending=True)
	out = torch.argmax(out)
	out = out.cpu().numpy()
	labels = labels.numpy()
	print("Iteration: ", i, ": ", outsort[0,:5],labels)

	if(out==labels):
		accuracy = accuracy+1

	out = outsort[0,:5].cpu().numpy()
	if labels in out:
		top5_accuracy = top5_accuracy + 1

	print("Accuracy is: ", (accuracy*100)/(i+1))


print("Accuracy is: ", (accuracy*100)/NUM_STEPS_STOP)	
print("Top 5 Accuracy is: ", (top5_accuracy*100)/NUM_STEPS_STOP)	
