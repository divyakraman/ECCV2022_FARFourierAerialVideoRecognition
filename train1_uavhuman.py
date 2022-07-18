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

from models.discaus1 import *
from dataset.uav_human import UAVHumanDataSet

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

#Batch size 2 iter size 16 works
BATCH_SIZE = 2
ITER_SIZE = 16
NUM_WORKERS = 0

DATA_DIRECTORY = 'PATH'
DATA_LIST_PATH = './dataset/UAVHuman/train.txt'
#INPUT_SIZE = '540,960' #1080,1920,3
INPUT_SIZE_SCALE = 2
LEARNING_RATE = 0.01
MOMENTUM = 0.9
NUM_CLASSES = 400 #155 actually
NUM_STEPS = 5000
NUM_STEPS_STOP = 5000 
POWER = 0.9
RANDOM_SEED = 1234
#RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_PRED_EVERY = 10
SNAPSHOT_DIR = './snapshots/uavhuman_far/'
WEIGHT_DECAY = 0.0005

NUM_FRAMES = 8

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(LEARNING_RATE, i_iter, NUM_STEPS_STOP, POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def main():
	"""Create the model and start the training."""

	net = DisCaus(101, NUM_CLASSES, 0.5, without_t_stride=False)
	net.train()    
	net = torch.nn.DataParallel(net).cuda()
	#net = net.cuda() 

	if not os.path.exists(SNAPSHOT_DIR):
		os.makedirs(SNAPSHOT_DIR)

	trainloader = data.DataLoader(
		UAVHumanDataSet(DATA_DIRECTORY, DATA_LIST_PATH, max_iters=NUM_STEPS_STOP * ITER_SIZE * BATCH_SIZE,
					mean=IMG_MEAN, set='train', num_frames = NUM_FRAMES, input_size_scale=INPUT_SIZE_SCALE),
		batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
	
	trainloader_iter = enumerate(trainloader)

	# implement model.optim_parameters(args) to handle different models' lr setting

	optimizer = optim.SGD(net.parameters(),
						  lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum = MOMENTUM)
	optimizer.zero_grad()

	criterion = nn.CrossEntropyLoss()
	
	for i_iter in range(1198, NUM_STEPS_STOP):

		loss_value = 0
		
		optimizer.zero_grad()
		
		if(i_iter%10==0):
			adjust_learning_rate(optimizer, i_iter)
		
		for sub_i in range(ITER_SIZE):


			#Supervised set training
			_, batch = trainloader_iter.__next__()
			images, labels = batch #Images: Batch size x frames x 3 X H x W
			images = images.permute(0,2,1,3,4) 
			
			images = Variable(images).cuda()
			labels = Variable(labels.long()).cuda()
			out = net(images)
			out_pred = torch.argmax(out, 1)
	 
			loss = criterion(out, labels)
			loss = loss.sum()/BATCH_SIZE
			loss = loss/ITER_SIZE
			# proper normalization
			loss.backward()
			loss_value = loss_value + loss.data.cpu().numpy() * ITER_SIZE

			#grad = [x.grad for x in net.parameters()]
			
			print('exp = {}'.format(SNAPSHOT_DIR))
			print('Iteration: ', i_iter, ' Step: ', sub_i, ' Loss: ', loss_value/(sub_i+1))
			print("Prediction: ", out_pred, " GT: ", labels)
	   
		optimizer.step()
		
		torch.cuda.empty_cache()
		
		
		if i_iter >= NUM_STEPS_STOP - 1:
			print('save model ...')
			torch.save(net, osp.join(SNAPSHOT_DIR, 'UAVHuman_' + str(NUM_STEPS_STOP) + '.pth'))
			break

		if i_iter % SAVE_PRED_EVERY == 0 and i_iter != 0:
			print('taking snapshot ...')
			torch.save(net, osp.join(SNAPSHOT_DIR, 'UAVHuman_' + str(NUM_STEPS_STOP) + '.pth'))
			
if __name__ == '__main__':
	main()




