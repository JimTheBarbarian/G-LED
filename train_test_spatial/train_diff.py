from tqdm import tqdm
import torch
import pdb
import sys
import os
import numpy as np
sys.path.insert(0, './util')
from utils import save_loss

def train_diff(diff_args,
			   seq_args,
			   trainer,
			   data_loader):
	loss_list = []
	avg_epoch_loss = []
	print('We are training for {} epochs'.format(diff_args.epoch_num))
	for epoch in range(diff_args.epoch_num):
		down_sampler = torch.nn.Upsample(size=[1,seq_args.coarse_dim], 
								     	 mode=seq_args.coarse_mode)
		up_sampler   = torch.nn.Upsample(size=[1, 64], 
								     	 mode=seq_args.coarse_mode)
		model, loss, avg_loss = train_epoch(diff_args,seq_args, trainer, data_loader,down_sampler,up_sampler)
		loss_list += loss
		if epoch >= 1:
			if avg_loss < min(avg_epoch_loss):
				model.save(path=os.path.join(diff_args.model_save_path, 
											'best_model_sofar'))
				np.savetxt(os.path.join(diff_args.model_save_path, 
									'best_model_sofar_epoch'),np.ones(2)*epoch)
		if epoch == diff_args.epoch_num - 1:
			model.save(path=os.path.join(diff_args.model_save_path, 
										'last_model'))
			np.savetxt(os.path.join(diff_args.model_save_path, 
								'last_model_epoch'),np.ones(2)*epoch)
			save_loss(diff_args, loss_list, epoch)
		print("finish training epoch {}".format(epoch))
	return loss_list

def train_epoch(diff_args,seq_args, trainer, data_loader,down_sampler,up_sampler):
	loss_epoch = []
	print('Iteration is ', len(data_loader))
	for iteration, batch in tqdm(enumerate(data_loader)):

		batch = batch[:,65:,:] # only predicting after the warmup
		batch = batch.to(diff_args.device).float()
		batch_spectral = torch.fft.rfft(batch,dim = -1)[:,:,:8]
		batch_coarse2fine = torch.fft.irfft(batch_spectral,axis=-1,n=64)

		bsize = batch.shape[0]
		ntime = batch.shape[1] 
		batch_coarse2fine = batch_coarse2fine.reshape([bsize,1,1,ntime,64])
		#batch_coarse      = down_sampler(batch.reshape([bsize*ntime,1,1,64]))
		#batch_coarse2fine = up_sampler(batch_coarse).reshape([bsize,1,1,ntime,64])
		cond_images = batch_coarse2fine.permute(0,2,1,3,4)
		images = batch.unsqueeze(1).unsqueeze(3) # shape B x 1 x ntime x 1 x 64
		#need # B x F x T x H x W
		images = images.permute(0,1,3,2,4) # B x 1 x 1 x ntime x 64
		#batch= batch.permute([0,2,1,3,4])
		#batch_coarse2fine = batch_coarse2fine.permute([0,2,1,3,4])
		#print(batch.device)
		loss=trainer(images,cond_images=cond_images,unet_number=1,ignore_time=False)
		trainer.update(unet_number=1)
		print("loss is ", loss)
	
		loss_epoch.append(loss)
	avg_epoch_loss = sum(loss_epoch)/len(loss_epoch)
	print(len(loss_epoch))
	return trainer, loss_epoch, avg_epoch_loss
