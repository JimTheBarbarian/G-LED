import torch
from tqdm import tqdm
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import pickle
import pdb
from tqdm import tqdm
from accelerate import Accelerator


def get_data_location(args):
	if args.dataset == 'ins_channel':
		data_location = os.path.join(args.data_location, 'data_set_ins')
	elif args.dataset == 'backward_facing':
		data_location = os.path.join(args.data_location, 'data_set_pitz')
	elif args.dataset == 'duan':
		data_location = os.path.join(args.data_location, 'data_set_duan')
	else:
		raise ValueError('Not implemented')
	return data_location


def save_loss(args, loss_list, Nt,accelerator):
	plt.figure()
	plt.plot(loss_list,'-o')
	plt.yscale('log')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.title(str(min(loss_list))+'Nt'+str(Nt))
	print(os.path.join(args.logging_path, 'loss_curve.png'))
	plt.savefig(os.path.join(args.logging_path, 'loss_curve.png'))
	plt.close()
	if accelerator.is_main_process:
		np.savetxt(os.path.join(args.logging_path, 'loss_curve.txt'), 
				np.asarray(loss_list))

def save_args(args,accelerator):
	if accelerator.is_main_process:
		with open(os.path.join(args.logging_path, 'args.txt'), 'w') as f:
			json.dump(args.__dict__, f, indent=2)

def save_args_sample(args,name,accelerator):
	if accelerator.is_main_process:

		with open(os.path.join(args.experiment_path, name), 'w') as f:
			json.dump(args.__dict__, f, indent=2)

def read_args_txt(args, argtxt):
	#args.parser.parse_args(namespace=args.update_args_no_folder_create()) 
	f = open (argtxt, "r")   
	args = args.parser.parse_args(namespace=argparse.Namespace(**json.loads(f.read())))
	return args
	return t

def save_model(model, args, Nt, accelerator, bestModel = False):
	accelerator.wait_for_everyone()
	save_dir = args.current_model_save_path
	if bestModel:
		save_dir = os.path.join(save_dir, f"best_model_Nt_{Nt}")
	else:
		save_dir = os.path.join(save_dir, f"final_model_Nt_{Nt}")
    
    # accelerator.save_state saves model, optimizer, scheduler states etc.
	accelerator.save_state(save_dir) 
	if accelerator.is_main_process:
        # Optionally save args again or other metadata if needed
		pass 
	
def load_model(model,args_train,args_sample):
	if args_sample.usebestmodel:
		model.load_state_dict(torch.load(args_train.current_model_save_path+'best_model_sofar'))
	else:
		model.load_state_dict(torch.load(args_train.current_model_save_path+'model_epoch_'+str(args_sample.model_epoch)))
	return model














class normalizer_1dks(object):
	"""
	arguments:
	target_dataset (torch.utils.data.Dataset) : this is dataset we
												want to normalize
	"""
	def __init__(self, target_dataset,args):
		# mark the orginal device of the target_dataset
		self.mean = target_dataset.mean().to(args.device)
		self.std  = target_dataset.std().to(args.device)
	def normalize(self, batch):
		return (batch - self.mean) / self.std
	def normalize_inv(self, batch):
		return batch * self.std +self.mean




















if __name__ == '__main__':
	num_videos = 10
	fig, axs = plt.subplots(2,int(num_videos/2))
	number_of_sample = int(num_videos/2)
	fig.subplots_adjust(hspace=-0.9,wspace=0.1)
	videos_to_plot = [np.zeros([1,3,1,64,256]) for _ in range(num_videos)]
	j = 0
	for k in range(0, num_videos):
		this_video = videos_to_plot[k-1]
		axs[k//number_of_sample, k%number_of_sample].imshow(np.sqrt(this_video[0,0,j,:,:]**2 + this_video[0,1,j,:,:]**2))
		axs[k//number_of_sample, k%number_of_sample].set_xticks([]) 
		axs[k//number_of_sample, k%number_of_sample].set_yticks([])
	plt.savefig('test_space.png',bbox_inches='tight')
