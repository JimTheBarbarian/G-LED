import sys
import argparse
import pdb
from datetime import datetime
import os
from torch.utils.data import DataLoader, DistributedSampler
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import json

"""
Internal pacakage
"""
from main_seq_ks import Args as SEQ_ARGS
from mimagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer

sys.path.insert(0, './util')
from utils import save_args, read_args_txt,setup_for_distributed
sys.path.insert(0, './data')
from data_ks_preprocess import bfs_dataset 
sys.path.insert(0, './train_test_spatial')
from  train_diff import train_diff

class Args:
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		"""
		for finding the dynamics dir
		"""
		self.parser.add_argument("--bfs_dynamic_folder", 
								 default='output/ks_2025_04_26_00_56_49',
								 help='all the information of ks training')
		"""
		for diffusion model
		"""
		self.parser.add_argument("--Nt",
								 default = 320,
								 help = 'Time steps we use as a single seq')
		self.parser.add_argument("--unet_dim", 
								 default=32,
								 help='The unet dimension')
		self.parser.add_argument("--num_sample_steps", 
								 default=20,
								 help='The noise forward/reverse step')
		
		"""
		for training 
		"""
		self.parser.add_argument("--batch_size", default = 1)
		self.parser.add_argument("--epoch_num", default = 1)
		self.parser.add_argument("--device", type=str, default = "cuda:1")
		self.parser.add_argument("--shuffle",default=True)
		
		"""
		DDP arguments
		"""
		self.parser.add_argument("--local_rank", type=int, default=-1, help='Local rank for distributed training')
		self.parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help='Total number of GPUs')
		self.parser.add_argument("--master_addr", type=str, default='localhost', help='Master address for DDP')
		self.parser.add_argument("--master_port", type=str, default='12355', help='Master port for DDP')
		self.parser.add_argument("--dist_backend", type=str, default='nccl', help='Distributed backend')

	def update_args(self):
		args = self.parser.parse_args()
		# output dataset
		args.experiment_path = os.path.join(args.bfs_dynamic_folder,'diffusion_folder')
		if not os.path.isdir(args.experiment_path):
			os.makedirs(args.experiment_path)
		args.model_save_path = os.path.join(args.experiment_path,'model_save')
		if not os.path.isdir(args.model_save_path):
			os.makedirs(args.model_save_path)
		args.logging_path = os.path.join( args.experiment_path,'logging') 
		if not os.path.isdir(args.logging_path):
			os.makedirs(args.logging_path)

		args.seq_args_txt = os.path.join(args.bfs_dynamic_folder,
										 'logging','args.txt' )
        # Determine if distributed training is enabled
		if "WORLD_SIZE" in os.environ:
			args.world_size = int(os.environ["WORLD_SIZE"])
			args.distributed = args.world_size > 1
		elif args.local_rank != -1: # Check if launched with torch.distributed.launch
			args.world_size = dist.get_world_size() # Get world size if already initialized
			args.distributed = args.world_size > 1
		else:
			args.world_size = 1
			args.distributed = False

        # Create directories only on the main process
		if not args.distributed or args.local_rank == 0: # Check if main process
			if not os.path.isdir(args.experiment_path):
				os.makedirs(args.experiment_path)
			if not os.path.isdir(args.model_save_path):	
				os.makedirs(args.model_save_path)
			if not os.path.isdir(args.logging_path):
				os.makedirs(args.logging_path)	
		return args



if __name__ == '__main__':

	"""
	Diff args
	"""
	diff_args = Args()
	diff_args = diff_args.update_args()

	"""
	Initialize distributed training
	"""

	if diff_args.distributed:
		if "LOCAL_RANK" in os.environ:
			diff_args.local_rank = int(os.environ["LOCAL_RANK"])
		if diff_args.local_rank != -1:
			print(f"Initializing process group for rank {diff_args.local_rank}...")
			torch.cuda.set_device(diff_args.local_rank)
			diff_args.device = f'cuda:{diff_args.local_rank}'
			# Setup environment variables if not set by launcher
			#os.environ['MASTER_ADDR'] = diff_args.master_addr
			#os.environ['MASTER_PORT'] = diff_args.master_port
			dist.init_process_group(backend=diff_args.dist_backend, init_method='env://', world_size=diff_args.world_size, rank=diff_args.local_rank)
			print(f"Process group initialized for rank {diff_args.local_rank}.")
		else:
			# Handle non-local rank based DDP init if necessary, or raise error
			raise RuntimeError("Distributed training requested but local_rank is not set.")
		setup_for_distributed(diff_args.local_rank == 0) # Pass is_main_process flag
	else:
		# Setup for non-distributed case
		setup_for_distributed(True)
		if diff_args.device is None: # Set default device if not distributed and not specified
			diff_args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	
	# Save args only on the main process
		


	save_args(diff_args)
	"""
	Sequence args
	"""
	seq_args = read_args_txt(SEQ_ARGS(),diff_args.seq_args_txt)
	
	"""
	Fetch dataset
	"""
	data_set = bfs_dataset(
						   trajec_max_len = diff_args.Nt,#seq_args.trajec_max_len,
						   start_n        = seq_args.start_n,
						   )
	sampler = DistributedSampler(data_set, shuffle = diff_args.shuffle) if diff_args.distributed else None
	data_loader = DataLoader(dataset=data_set, 
						  	sampler=sampler,
							shuffle=False if diff_args.distributed else diff_args.shuffle,
							batch_size=diff_args.batch_size,
							pin_memory=True,
							num_workers=4)
	
	"""
	Create diffusion model
	"""
	device = torch.device(diff_args.device)
	unet1 = Unet3D(dim=diff_args.unet_dim,
				   cond_images_channels=1, 
				   memory_efficient=True, 
				   dim_mults=(1, 2, 4, 8)).to(device)  #mid: mid channel
	image_sizes = (320)
	image_width = (64)
	imagen = ElucidatedImagen(
		unets = (unet1),
		image_sizes = image_sizes,
		image_width = image_width,   
		channels = 1,   # Han Gao add the input to this args explicity     
		random_crop_sizes = None,
		num_sample_steps = diff_args.num_sample_steps, # original is 10
		#cond_drop_prob = 0.1,
		sigma_min = 0.002,
		sigma_max = (80),      # max noise level, double the max noise level for upsampler  （80，160）
		sigma_data = 0.5,      # standard deviation of data distribution
		rho = 7,               # controls the sampling schedule
		P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
		P_std = 1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
		S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in apper
		S_tmin = 0.05,
		S_tmax = 50,
		S_noise = 1.003,
		#condition_on_text = False,
		auto_normalize_img = False  # Han Gao make it false
		).to(device)
	#if diff_args.distributed:
		#imagen = DDP(imagen, device_ids=[diff_args.local_rank], output_device=diff_args.local_rank)
	trainer = ImagenTrainer(imagen, device =device, use_ddp=diff_args.distributed)
	train_diff(diff_args=diff_args,
               seq_args=seq_args,
               trainer=trainer,
               data_loader=data_loader)
	
	if diff_args.distributed:
		dist.destroy_process_group()
		print("Distributed process group destroyed.")