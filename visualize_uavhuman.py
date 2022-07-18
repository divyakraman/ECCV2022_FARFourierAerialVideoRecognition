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

from dataset.uav_human import UAVHumanDataSet

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
NUM_STEPS_STOP = 20 #5555 

DATA_DIRECTORY = 'PATH'
DATA_LIST_PATH = './dataset/UAVHuman/test.txt'
SNAPSHOT_DIR = 'Visualizations/UAVHumanNight'

net = torch.load('snapshots/uavhuman_far/UAVHuman_10000.pth', map_location= 'cpu')
net = net.cuda()

trainloader = data.DataLoader(
		UAVHumanDataSet(DATA_DIRECTORY, DATA_LIST_PATH, max_iters=NUM_STEPS_STOP,
					mean=IMG_MEAN, set='train', num_frames=8),
		batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
	
trainloader_iter = enumerate(trainloader)

accuracy = 0

channels = [0,100,200,300,400,500]

if not os.path.exists(SNAPSHOT_DIR):
		os.makedirs(SNAPSHOT_DIR)

for i in range(0, NUM_STEPS_STOP):
	_, batch = trainloader_iter.__next__()
	images, labels = batch
	images = images.permute(0,2,1,3,4)
	images = Variable(images).cuda()
	out = net(images)
	out = out.detach()
	outsort = torch.argsort(out,descending=True)
	out = torch.argmax(out)
	out = out.cpu().numpy()
	labels = labels.numpy()
	print("Iteration: ", i, ": ", outsort[0,:5],labels)
	torch.cuda.empty_cache()
	if(out==labels):
		accuracy = accuracy+1
	if(out==labels):
	#if(out!=labels):#Failure Cases
		x = net.module.model_backbone.conv1(images)
		x = net.module.model_backbone.bn1(x)
		x = net.module.model_backbone.relu(x)
		x = net.module.model_backbone.maxpool(x) #64,32,120,68 for input 32, 270, 480

		x = net.module.model_backbone.layer1(x) #256,32,120,68 for input 32, 270, 480
		
		x = net.module.model_backbone.layer2(x) #512,16,60,34 for input 32, 270, 480
		x1 = net.module.objback_disentangle(x)
		x2 = net.module.spatialCaus(x) - x
		x1 = x1.detach().cpu().numpy()
		x2 = x2.detach().cpu().numpy()
		x_before = x.detach().cpu().numpy()
		
		images = images[0,:,:,:,:]
		images = images.permute(1,2,3,0)
		images = images.cpu().numpy()
		frameno = -1
		for framenum in range(0,images.shape[0],2):
			img_save = images[framenum,:,:,:]
			frameno = frameno+1
			#plt.imsave('Visualizations/UAVHumanFail/Image'+str(i)+'_Class'+str(out)+'_frame'+str(frameno)+'.png', img_save)
			plt.imsave('Visualizations/UAVHumanFail/Image'+str(i)+'_PredClass'+str(out)+'_GTClass'+str(labels)+'_frame'+str(frameno)+'.png', img_save)

		x1 = np.sum(x1,1)
		x2 = np.sum(x2,1)
		x_before = np.sum(x_before,1)
		
		
		for framenum in range(0,4):
			c=0
			img_save = x1[0,framenum,:,:]
			plt.imsave('Visualizations/UAVHumanFail/Image'+str(i)+'_Class'+str(out)+'_frame'+str(framenum)+'_objbackdisentangle.png', img_save)
			img_save = x2[0,framenum,:,:]
			plt.imsave('Visualizations/UAVHumanFail/Image'+str(i)+'_Class'+str(out)+'_frame'+str(framenum)+'_spatialCaus.png', img_save)
			img_save = x_before[0,framenum,:,:]
			plt.imsave('Visualizations/UAVHumanFail/Image'+str(i)+'_Class'+str(out)+'_frame'+str(framenum)+'_before.png', img_save)
		

		del x,x_before,x1,x2
	
	print("Accuracy is: ", (accuracy*100)/(i+1))
	torch.cuda.empty_cache()
	


print("Accuracy is: ", (accuracy*100)/NUM_STEPS_STOP)	
