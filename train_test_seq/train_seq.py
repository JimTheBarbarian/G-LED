import pdb
import torch
from tqdm import tqdm
from test_seq import test_epoch
import os
import time
import sys
sys.path.insert(0, './util')
from utils import save_model
def train_seq_shift(args, 
					model, 
					data_loader, 
					data_loader_copy,
					data_loader_valid,
					loss_func, 
					optimizer,
					scheduler,
					accelerator):
	# N C L
	down_sampler = torch.nn.Upsample(size=(1,args.coarse_dim), 
								     mode=args.coarse_mode).to(accelerator.device)
	Nt = args.start_Nt
	for epoch in tqdm(range(args.epoch_num)):
		tic = time.time()
		accelerator.print('Start epoch '+ str(epoch)+' at Nt ', Nt)
		if epoch >0:
			max_mre,min_mre, mean_mre, sigma3 = test_epoch(args=args,
														   model=model, 
														   data_loader=data_loader_valid,
														   loss_func=loss_func,
														   Nt=Nt,
														   down_sampler=down_sampler,
														   ite_thold = 2,
														   accelerator = accelerator)
			accelerator.print('#### max  re valid####=',max_mre)
			accelerator.print('#### mean re valid####=',mean_mre)
			accelerator.print('#### min  re valid####=',min_mre)
			accelerator.print('#### 3 sigma valid####=',sigma3)
			accelerator.print('Last LR is '+str(scheduler.get_last_lr()))
			max_mre,min_mre, mean_mre, sigma3 = test_epoch(args = args,
							                               model = model, 
                                                           data_loader = data_loader_copy,
                                                           loss_func = loss_func,
                                                           Nt = Nt,
                                                           down_sampler = down_sampler,
                                                           ite_thold = 5,
														   accelerator = accelerator)
			accelerator.print('#### max  re train####=',max_mre)
			accelerator.print('#### mean re train####=',mean_mre)
			accelerator.print('#### min  re train####=',min_mre)
			accelerator.print('#### 3 sigma train ####=',sigma3)
			if (max_mre < args.march_tol) or (mean_mre < args.march_tol*0.1):
				save_model(model, args, Nt, bestModel = True)
				Nt += args.d_Nt
				scheduler.step()
				continue
		
		model = train_epoch(args=args,
							model=model, 
							data_loader=data_loader,
							loss_func=loss_func,
							optimizer=optimizer,
							down_sampler=down_sampler)
				
		accelerator.print('Epoch elapsed ', time.time()-tic)
	save_model(model, args, Nt, bestModel = False)
def train_epoch(args, 
				model, 
				data_loader, 
				loss_func, 
				optimizer,
				down_sampler,
				accelerator):
	accelerator.print('Nit = ',len(data_loader))
	for iteration, batch in tqdm(enumerate(data_loader),disable=not accelerator.is_local_main_process):	
		#batch = batch.to(args.device).float()
		batch = batch.float()
		b_size = batch.shape[0]
		num_time = batch.shape[1]
		#num_velocity = 2 # Not doing BFS equations
		batch = batch.reshape([b_size,num_time, 1,64])
		batch_coarse = down_sampler(batch).reshape([b_size, 
													num_time, 
	
													args.coarse_dim])
		batch_coarse_flatten = batch_coarse.reshape([b_size, 
													 num_time,
													 args.coarse_dim])
		assert num_time == args.n_ctx + 1
		for j in (range(num_time - args.n_ctx)):
			model.train()
			optimizer.zero_grad()
			xn = batch_coarse_flatten[:,j:j+args.n_ctx,:]
			xnp1,_,_,_=model(inputs_embeds = xn, past=None)
			xn_label = batch_coarse_flatten[:,j+1:j+1+args.n_ctx,:]
			loss = loss_func(xnp1, xn_label)
			accelerator.backward(loss)
			optimizer.step()
	return model