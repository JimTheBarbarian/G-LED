import numpy as np
import pdb
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class bfs_dataset(Dataset):
	def __init__(self,
				 data_location=['data/data0.npy','data/data1.npy'],
				 trajec_max_len=50,
				 start_n=0,
				 #num_trajs=1,
				 val_split=.9,
				 test_split=.95,
				 flag = 'train'):
		#assert n_span > trajec_max_len
		solution0 = np.load(data_location[0],allow_pickle = True)

		self.start_n = start_n
		self.trajec_max_len = trajec_max_len

		solution1 = np.load(data_location[1],allow_pickle = True)
		solution  = np.concatenate([solution0,
									solution1],axis = 0)
		self.solution = torch.from_numpy(solution)
		self.flag = flag
		val_index = int(len(self.solution) * val_split)
		test_index = int(len(self.solution) * test_split)
		if flag == 'train':
			self.solution = self.solution[:2,:,:]
		elif flag == 'val':
			self.solution = self.solution[val_index:test_index]
		elif flag == 'test':
			self.solution = self.solution[2:4,:,:]
		self.num_trajs = self.solution.shape[0]
		self.n_span = self.solution.shape[1]
		self.trajec_max_len = trajec_max_len
	def __len__(self):
		return self.num_trajs
		
	def __getitem__(self, index):
		return self.solution[index]
		#if self.flag == 'test':
		#	return self.solution[index,:self.start_n]
		#else:
		#	return self.solution[index]


if __name__ == '__main__':
	dset = bfs_dataset()
	dloader = DataLoader(dataset=dset, batch_size = 20,shuffle = True)
	for iteration, batch in enumerate(dloader):
		print(iteration)
		print(batch.shape)
		print('Do something!')
		pdb.set_trace()