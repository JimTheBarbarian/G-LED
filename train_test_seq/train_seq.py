import pdb
import torch
from tqdm import tqdm
from test_seq import test_epoch
import os
import time
import sys
sys.path.insert(0, './util')
import torch.distributed as dist
from utils import save_model,is_main_process
def train_seq_shift(args, 
					model, 
					data_loader, 
					sampler_train,
					data_loader_copy,
					data_loader_valid,
					loss_func, 
					optimizer,
					scheduler
					):
	# N C L
	down_sampler = torch.nn.Upsample(size=(1,args.coarse_dim), 
								     mode=args.coarse_mode).to(args.device)
	Nt = args.start_Nt
	valid_Nt = args.n_span_valid
	warm_start_len = args.start_n_valid
	for epoch in tqdm(range(args.epoch_num), disable=not is_main_process()):
		tic = time.time()
		if args.distributed and sampler_train is not None:
			sampler_train.set_epoch(epoch)
		 # This variable helps make sure we have the correct Nt value on all gpus
		 # While still only doing validation on the main gpu
		march_nt_decision = [False]
		if epoch >0:

			if is_main_process():
				print('Start epoch '+ str(epoch)+' at Nt ', Nt)
				if epoch % 1 == 0:
				
					max_mre,min_mre, mean_mre, sigma3 = test_epoch(args=args,
														   model=model, 
														   data_loader=data_loader_valid,
														   loss_func=loss_func,
														   Nt=valid_Nt,
														   warm_start_len = warm_start_len,
														   down_sampler=down_sampler,
														   ite_thold = 5,
														   device = args.device,
														   distributed=args.distributed
														   )
			
				

					print('#### max  re train####=',max_mre)
					print('#### mean re train####=',mean_mre)
					print('#### min  re train####=',min_mre)
					print('#### 3 sigma train ####=',sigma3)
					if (max_mre < args.march_tol) or (mean_mre < args.march_tol*0.1):
						march_nt_decision[0] = True
						save_model(model, args, Nt, bestModel = True)
				
			Nt += args.d_Nt
		#if args.distributed and epoch > 0:
		#	dist.broadcast_object_list(march_nt_decision,src=0)


		#if march_nt_decision[0]:
		#	Nt += args.d_Nt
		#	scheduler.step()
			
		
		#if args.distributed:
		#	dist.barrier()

		#if not march_nt_decision[0]:					
			model = train_epoch(args=args,
							model=model, 
							data_loader=data_loader,
							loss_func=loss_func,
							optimizer=optimizer,
							down_sampler=down_sampler,
							device=args.device,
							distributed=args.distributed)
			scheduler.step()

			print('Epoch elapsed ', time.time()-tic)
	if is_main_process():
		save_model(model, args, Nt, bestModel = False)
		return

def train_epoch(args, 
				model, 
				data_loader, 
				loss_func, 
				optimizer,
				down_sampler,
				device,
				distributed
				):
	model.train()
	for iteration, batch in tqdm(enumerate(data_loader), disable=not is_main_process()):
		batch = batch[:,:,:64].to(device).float()
		batch_min = torch.min(batch)
		batch_max = torch.max(batch)
		normalized_batch = (batch - batch_min) / (batch_max - batch_min + 1e-8) # Normalize to [0, 1]

		#batch = batch.float()
		b_size = batch.shape[0]
		num_time = batch.shape[1]
		#num_velocity = 2 # Not doing BFS equations
		##batch = batch.reshape([b_size,num_time, 1,64])
		#batch_coarse = down_sampler(batch).reshape([b_size, 
		#											num_time, 
	
		#											args.coarse_dim])
		#batch_coarse_flatten = batch_coarse.reshape([b_size, 
		#											 num_time,
		#											 args.coarse_dim])
		assert num_time == args.n_ctx + 1
		for j in (range(num_time - args.n_ctx)):
			model.train()
			optimizer.zero_grad()
			xn = normalized_batch[:,j:j+args.n_ctx,:]
			xnp1,_,_,_=model(inputs_embeds = xn, past=None)
			xn_label = normalized_batch[:,j+1:j+1+args.n_ctx,:]
			loss = loss_func(xnp1, xn_label)
			loss.backward()
			optimizer.step()
	return model