import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import cv2

class UAVHumanDataSet(data.Dataset):
	def __init__(self, root, list_path, max_iters=None, mean=(128, 128, 128), set='train', num_frames=64, input_size_scale=2):
		self.root = root
		self.list_path = list_path
		self.mean = mean
		self.img_ids = [i_id.strip() for i_id in open(list_path)]
		if not max_iters==None:
			self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
		self.files = []
		self.set = set
		self.num_frames = num_frames
		self.input_size_scale = int(input_size_scale)
		for name in self.img_ids:
			img_file = self.root + name
			self.files.append({
				"img": img_file,
				"name": name
			})


	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		datafiles = self.files[index]
		capture = cv2.VideoCapture(datafiles["img"])
		ret, frame = capture.read() #1080,1920,3; number of frames variable for each image
		frame_no = 0
		frames = []
		frame_size = np.asarray(frame).shape
		while ret:
			if(frame_no%4==0):
				#frame = cv2.resize(frame,(540,960))
				frame = cv2.resize(frame,(int(frame_size[0]/self.input_size_scale), int(frame_size[1]/self.input_size_scale)))
				frame = np.asarray(frame)
				frames.append(torch.from_numpy(frame).permute(2,0,1)/255.0)
			frame_no += 1
			ret, frame = capture.read()
		frame = frames[0]
		frame_shape = frame.shape
		
		step = int(len(frames)/self.num_frames)
		if(step<=0):
			running_count=0
		else:
			running_count = np.random.randint(step)
		#running_count = 0
		x_new = torch.zeros(self.num_frames, 3, frame_shape[1], frame_shape[2])
		for i in range(0, self.num_frames):
			x_new[i,:,:,:] = frames[running_count]
			running_count = running_count + step 
		frames_torch = x_new
		del x_new
		
		gt_label = datafiles["name"]
		gt_label = gt_label[-18:-15]
		gt_label = int(gt_label[0]) * 100 + int(gt_label[1]) * 10 + int(gt_label[2]) * 1
		gt_label = gt_label

		return frames_torch, gt_label

		
