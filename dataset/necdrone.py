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
import pandas as pd 
import glob


class NECDataSet(data.Dataset):
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
			name = name.split('\t')
			class_name = name[1]
			img_file = self.root + name[0]
			self.files.append({
				"img": img_file,
				"name": name[0],
				"class_name": class_name
			})


	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		datafiles = self.files[index]
		gt_label = int(datafiles["class_name"])
		path = datafiles["img"] + '*.jpg'
		video_images_list = glob.glob(path)
		#video_images_list = os.listdir(datafiles["img"])
		video_images_list.sort()
		num_frames_video = len(video_images_list)

		step = int(num_frames_video/self.num_frames)
		if(step<=0):
			running_count=0
		else:
			running_count = np.random.randint(step)
		#running_count = 0
		
		frame = cv2.imread(video_images_list[0])
		frame_shape = frame.shape
		x = torch.zeros(self.num_frames, 3, int(1920/self.input_size_scale), int(1080/self.input_size_scale))

		for i in range(0, self.num_frames):
			frame = cv2.imread(video_images_list[running_count])
			frame = cv2.resize(frame,(int(1080/self.input_size_scale), int(1920/self.input_size_scale)))
			frame = np.asarray(frame)
			frame = torch.from_numpy(frame).permute(2,0,1)/255.0
			running_count = running_count + step 
			x[i,:,:,:] = frame
		
		return x, gt_label

		
