import pdb
import torch
from tqdm import tqdm
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.distributed as dist
import sys
from torch.utils.data import DataLoader as Dataloader


# Get the absolute path to the project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add paths to system path
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'data'))
sys.path.append(os.path.join(ROOT_DIR, 'util'))
sys.path.append(os.path.join(ROOT_DIR, 'transformer'))


import argparse
from data.data_ks_preprocess import bfs_dataset 
from util.utils import is_main_process,read_args_txt

from transformer.sequentialModel import SequentialModel as transformer
from main_seq_ks import Args as Args_train_base 

# Define a simple MSE loss function for evaluation
def mse_loss(pred, target):
    return torch.mean((pred - target)**2)

"""
Start training test
"""
def test_epoch(args,
			   model, 
			   data_loader,
			   loss_func,
			   Nt,
			   warm_start_len,
			   down_sampler,
			   ite_thold = None,
			   device = None,
			   distributed = False
			   ):
	assert device is not None, "device must be provided"
	assert warm_start_len > 0, "warm_start_len must be positive"
    
    # +++ Use model.module if distributed, otherwise use model directly
	eval_model = model.module if distributed else model
	eval_model.eval() # Set the underlying model to eval mode
	with torch.no_grad():
		#IDHistory = [0] + [i for i in range(2, args.n_ctx)]
		IDHistory = [i for i in range(1, args.n_ctx)]
		REs = []
		if is_main_process():print("Total ite", len(data_loader))
		for iteration, batch in tqdm(enumerate(data_loader)):
			if ite_thold is None:
				pass
			else:
				if iteration>ite_thold:
					break
			batch = batch.to(device).float()
			print(batch.shape)
			#batch = batch.float()
			b_size = batch.shape[0]
			num_time_total = batch.shape[1]
			#num_velocity = 2 not doing BFS
			batch = batch.reshape([b_size,num_time_total, 1,64])
			batch_coarse = down_sampler(batch).reshape([b_size, 
														num_time_total, 
														args.coarse_dim])
			batch_coarse_flatten = batch_coarse.reshape([b_size, 
														 num_time_total,
														 args.coarse_dim])
			

			warm_start_coarse = batch_coarse_flatten[:,:warm_start_len,:]
			past = None
			xn = warm_start_coarse[:,0:1,:]

			for k in range(warm_start_len-1):
				if past is not None and past[0][0].shape[2] >= args.n_ctx:
					past_trimmed = [[past[l][0][:,:,IDHistory,:], past[l][1][:,:,IDHistory,:]] for l in range(args.n_layer)]
				else: 
					past_trimmed = past
				_, past, _, _, = eval_model(inputs_embeds = xn, past=past_trimmed)
				xn = warm_start_coarse[:,k+1:k+2,:]
				
			mem = []
			for j in (range(Nt)):
				if j == 0:
					xnp1,past,_,_=eval_model(inputs_embeds = xn, past=past)
				elif past[0][0].shape[2] < args.n_ctx and j > 0:
					xnp1,past,_,_=eval_model(inputs_embeds = xn, past=past)
				else:
					past = [[past[l][0][:,:,IDHistory,:], past[l][1][:,:,IDHistory,:]] for l in range(args.n_layer)]
					xnp1,past,_,_=eval_model(inputs_embeds = xn, past=past)
				xn = xnp1
				mem.append(xn)
			mem=torch.cat(mem,dim=1)

			# Calculate loss for the local batch
            # +++ Vectorize loss calculation if possible
			target = batch_coarse_flatten[:,warm_start_len:warm_start_len+Nt,:]	
			er = loss_func(mem, target)
            # Avoid division by zero if target norm is zero
			target_norm_sq = loss_func(target*0, target)
			r_er = er / target_norm_sq if target_norm_sq > 1e-9 else torch.tensor(0.0, device=device)
            
            # +++ Store the single relative error for the batch (or average if needed)
			REs.append(r_er.item()) # Append scalar relative error for this batch

        # +++ Gather results if distributed
		all_REs = []
		if distributed:
            # Ensure all processes reach this point
			dist.barrier() 
            # Gather lists of scalars using all_gather_object
			world_size = dist.get_world_size()
			gathered_REs = [None] * world_size
			dist.all_gather_object(gathered_REs, REs)
			if is_main_process():
				all_REs = [item for sublist in gathered_REs for item in sublist]
			else:
				all_REs = REs
		        # +++ Calculate final metrics only on the main process			
		if is_main_process():
			if not all_REs: # Handle case with no results
				return 0.0, 0.0, 0.0, 0.0 
	
			REs_np = np.array(all_REs)
			max_mre = np.max(REs_np)
			min_mre = np.min(REs_np)
			mean_mre = np.mean(REs_np)
			sigma3 = np.std(REs_np) * 3
			return max_mre, min_mre, mean_mre, sigma3







def test_plot_eval(args,
				   args_sample,
				   model, 
				   data_loader,
				   loss_func,
				   Nt,
				   down_sampler):
	try:
		os.makedirs(args.experiment_path+'/contour')
	except:
		pass
	contour_dir = args.experiment_path+'/contour'
	with torch.no_grad():
		#IDHistory = [0] + [i for i in range(2, args.n_ctx)]
		IDHistory = [i for i in range(1, args.n_ctx)]
		REs = []
		print("Total ite", len(data_loader))
		for iteration, batch in tqdm(enumerate(data_loader)):
			batch = batch.to(args.device).float()		
			b_size = batch.shape[0]
			num_time = batch.shape[1]
			#num_velocity = 2
			batch = batch.reshape([b_size*num_time, 64])
			batch_coarse = down_sampler(batch).reshape([b_size, 
														num_time, 
														args.coarse_dim])
			batch_coarse_flatten = batch_coarse.reshape([b_size, 
														 num_time,
														 args.coarse_dim])
			
			past = None
			xn = batch_coarse_flatten[:,0:1,:]
			previous_len = 1 
			mem = []
			for j in (range(Nt)):
				if j == 0:
					xnp1,past,_,_=model(inputs_embeds = xn, past=past)
				elif past[0][0].shape[2] < args.n_ctx and j > 0:
					xnp1,past,_,_=model(inputs_embeds = xn, past=past)
				else:
					past = [[past[l][0][:,:,IDHistory,:], past[l][1][:,:,IDHistory,:]] for l in range(args.n_layer)]
					xnp1,past,_,_=model(inputs_embeds = xn, past=past)
				xn = xnp1
				mem.append(xn)
			mem=torch.cat(mem,dim=1)

			local_batch_size = mem.shape[0]
			for i in tqdm(range(local_batch_size)):
				target = batch_coarse_flatten[i:i+1,previous_len:previous_len+Nt,:]
				prediction = mem[i:i+1]
				er = loss_func(prediction,
							   target)
				
				target_norm_sq = loss_func(target*0, target)
				r_er = er/target_norm_sq if target_norm_sq > 1e-9 else torch.tensor(0.0, device=args.device)

				REs.append(r_er.item())
				pred_np = prediction.squeeze(0).cpu().numpy()
				truth_np = target.squeeze(0).cpu().numpy()
				error_np = np.abs(pred_np - truth_np)
				spatial_dim = pred_np.shape[1]
				x_axis  = np.arange(spatial_dim)
				time_axis = np.arange(Nt) + previous_len
				# spatial recover


				
				seq_name = 'batch'+str(iteration)+'sample'+str(i)
				sampler_dir = os.path.join(contour_dir,seq_name)
				try:
					os.makedirs(sampler_dir)
				except OSError:
					pass

				fig_heatmap, axes_heatmap = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True)
				fig_heatmap.suptitle(f'Sample {i}: Predicted vs. Truth Heatmaps')

				                # Determine shared color limits based on truth range for better comparison
				vmin = truth_np.min()
				vmax = truth_np.max()

                # Plot Prediction Heatmap
				im0 = axes_heatmap[0].imshow(pred_np.T, aspect='auto', origin='lower', cmap='jet', vmin=vmin, vmax=vmax,
                                             extent=[time_axis.min(), time_axis.max(), 0, spatial_dim])
				axes_heatmap[0].set_title('Prediction')
				axes_heatmap[0].set_ylabel('Spatial Dimension')


				# Plot Truth Heatmap

				im1 = axes_heatmap[1].imshow(truth_np.T, aspect='auto', origin='lower', cmap='jet', vmin=vmin, vmax=vmax,
											 extent=[time_axis.min(), time_axis.max(), 0, spatial_dim])
				axes_heatmap[1].set_title('Truth')
				axes_heatmap[1].set_ylabel('Spatial Dimension')

				im2 = axes_heatmap[2].imshow(error_np.T, aspect='auto', origin='lower', cmap='jet', vmin=vmin, vmax=vmax,
											 extent=[time_axis.min(), time_axis.max(), 0, spatial_dim])
				axes_heatmap[2].set_title('Error')
				axes_heatmap[2].set_xlabel('Time Step')
				axes_heatmap[2].set_ylabel('Spatial Dimension')
				fig_heatmap.colorbar(im2, ax=axes_heatmap, orientation='horizontal')
				
				plt.tight_layout(rect=[0,0.03, 1, 0.95])


				heatmap_filename = os.path.join(sample_dir, f'heatmap_comparison.png')
				fig_heatmap.savefig(heatmap_filename, bbox_inches='tight', dpi=150)
				plt.close(fig_heatmap)

                # --- 2. Relative Error vs. Time Plot ---
                # Calculate relative error at each time step
				error_norm_time = np.linalg.norm(error_np, axis=1) # L2 norm of error across space for each time step
				truth_norm_time = np.linalg.norm(truth_np, axis=1) # L2 norm of truth across space for each time step

                # Avoid division by zero
				relative_error_time = np.zeros_like(error_norm_time)
				valid_indices = truth_norm_time > 1e-9 # Or some small epsilon
				relative_error_time[valid_indices] = error_norm_time[valid_indices] / truth_norm_time[valid_indices]

				fig_rel_err, ax_rel_err = plt.subplots(figsize=(10, 4))
				ax_rel_err.plot(time_axis, relative_error_time, marker='o', linestyle='-')
				ax_rel_err.set_title(f'Sample {i}: Relative Error over Time')
				ax_rel_err.set_xlabel('Time Step')
				ax_rel_err.set_ylabel('Relative Error (L2 Norm)')
				ax_rel_err.grid(True)
				ax_rel_err.set_yscale('log') # Often useful for errors


				rel_err_filename = os.path.join(sample_dir, f'relative_error_vs_time.png')
				fig_rel_err.savefig(rel_err_filename, bbox_inches='tight', dpi=150)
				plt.close(fig_rel_err)


	if REs:
		REs_np = np.array(REs)
		print(f"Plotting Eval Metrics: Mean RE={np.mean(REs_np):.4f}, Max RE={np.max(REs_np):.4f}")

		'''
				try:
					os.makedirs(contour_dir+'/'+seq_name)
				except:
					pass
				for d in tqdm(range(num_velocity+1)):
					try:
						os.makedirs(contour_dir+'/'+seq_name+'/'+str(d))
					except:
						pass
					sub_seq_name = contour_dir+'/'+seq_name+'/'+str(d)
					for t in tqdm(range(Nt)):
						
						
						if d == 2:
							data = np.sqrt(prediction[0,t,0,:,:].cpu().numpy()**2+prediction[0,t,1,:,:].cpu().numpy()**2)
						else:
							data = prediction[0,t,d,:,:].cpu().numpy()
						
						if d == 2:
							X_AR = np.sqrt(truth[0,t,0,:,:].cpu().numpy()**2+truth[0,t,1,:,:].cpu().numpy()**2)
						else:
							X_AR = truth[0,t,d,:,:].cpu().numpy()
						#asp = data.shape[1]/data.shape[0]
						fig, axes = plt.subplots(nrows=3, ncols=1)
						fig.subplots_adjust(hspace=0.5)
						norm = matplotlib.colors.Normalize(vmin=X_AR.min(), vmax=X_AR.max())
						im0 = axes[0].imshow(data[:,:],extent=[0,10,0,2],cmap = 'jet',interpolation='bicubic',norm=norm)
						axes[0].set_title('LED Macro')#,rotation=-90, position=(1, -1))#, ha='left', va='center')
						#axes[0].invert_yaxis()
						#axes[0].set_xlabel('x')
						axes[0].set_ylabel('y')
						axes[0].set_xticks([])
						
						im1 = axes[1].imshow(X_AR[:,:],extent=[0,10,0,2], cmap = 'jet',interpolation='bicubic',norm=norm)
						axes[1].set_title('Label')#,rotation=-90, position=(1, -1), ha='left', va='center')
						#axes[1].invert_yaxis()
						axes[1].set_ylabel('y')
						axes[1].set_xticks([])

						im2 = axes[2].imshow(np.abs(X_AR-data),extent=[0,10,0,2], cmap = 'jet',interpolation='bicubic',norm=norm)
						axes[2].set_title('Error')#,rotation=-90, position=(1, -1), ha='left', va='center')
						#axes[2].invert_yaxis()
						axes[2].set_xlabel('x')
						axes[2].set_ylabel('y')
						#axes[2].set_xticks([])
						
						fig.subplots_adjust(right=0.8)
						fig.colorbar(im0,orientation="horizontal",ax = axes)
						fig.savefig(sub_seq_name+'/time'+str(t)+'.png', bbox_inches='tight',dpi=500)
						plt.close(fig)
						'''
"""
start test
"""
def eval_seq_overall(args_train,
					 args_sample,
					 model, 
					 data_loader, 
					 loss_func):
	down_sampler = torch.nn.Upsample(size=args_train.coarse_dim, 
								     mode=args_train.coarse_mode)
	Nt = args_sample.test_Nt
	warm_start_len = args_sample.start_n
	tic = time.time()
	print('Start test')
	max_mre,min_mre, mean_mre, sigma3 = test_epoch(args=args_train,
												   model=model, 
												   data_loader=data_loader,
												   loss_func=loss_func,
												   Nt=Nt,
												   warm_start_len=warm_start_len,
												   down_sampler=down_sampler,
												   ite_thold = None,
												   device = args_sample.device)
	print('#### max mre test####=',max_mre)
	print('#### mean mre test####=',mean_mre)
	print('#### min mre test####=',min_mre)
	print('#### 3 sigma ####=',sigma3)
	print('Test elapsed ', time.time()-tic)
	'''
	test_plot_eval(args=args_train,
				   args_sample = args_sample,
				   model=model, 
				   data_loader=data_loader,
				   loss_func=loss_func,
				   Nt=Nt,
				   down_sampler=down_sampler)
	'''



class Args_eval:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Evaluate Sequential Model')
        # Model/Training Args Location
        self.parser.add_argument("--train_args_txt", type=str, default='./output/ks_2025_04_20_18_06_37/logging/args.txt', help='Path to args.txt from training')
        self.parser.add_argument("--model_path", type=str,default='./output/ks_2025_04_20_18_06_37/model_save/final_model_Nt_2000.pt', help='Path to the trained model .pt file')
        # Data Args
        self.parser.add_argument("--data_location", type=str, default=['./data/data1.npy'], help='Directory containing evaluation data (e.g., .npy files)')
        #self.parser.add_argument("--file_name", type=str, default='data_test.npy', help='Name of the evaluation data file')
        self.parser.add_argument("--trajec_max_len", type=int, default=151, help='Max sequence length in data')
        self.parser.add_argument("--start_n", type=int, default=64, help='Starting index for data loading')
        self.parser.add_argument("--n_span", type=int, default=151, help='Number of steps to load from start_n')
        # Evaluation Params
        self.parser.add_argument("--test_Nt", type=int, default=321, help='Number of steps to predict forward')
        self.parser.add_argument("--batch_size", type=int, default=1, help='Batch size for evaluation')
        self.parser.add_argument("--device", type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device for evaluation (e.g., cuda:0 or cpu)')
        self.parser.add_argument("--experiment_path", type=str, default='./eval_output', help='Directory to save evaluation outputs (like plots)')

    def update_args(self):
        args = self.parser.parse_args()
        if not os.path.exists(args.experiment_path):
            os.makedirs(args.experiment_path)
        return args

if __name__ == "__main__":
	args_eval = Args_eval().update_args()
	args_train = read_args_txt(Args_train_base(), args_eval.train_args_txt)


	device = torch.device(args_eval.device)



	# Load Data

	eval_dataset = bfs_dataset(data_location=args_eval.data_location,
							   flag='test',
							   start_n = args_eval.start_n)
	eval_loader = Dataloader(dataset = eval_dataset,
						  batch_size = args_eval.batch_size,
						  shuffle = False,)
	model = transformer(args_train).to(device)

	checkpoint = torch.load(args_eval.model_path, map_location=device)
	state_dict = checkpoint.get('model_state_dict', checkpoint)
	new_state_dict = {}
	for k, v in state_dict.items():
		if k.startswith('module.'):
			new_state_dict[k[7:]] = v
		else:
			new_state_dict[k] = v
	model.load_state_dict(new_state_dict)
	model.eval()
	print('Model loaded successfully')

	loss_func = nn.MSELoss()


	print('Start evaluation')

	eval_seq_overall(args_train, args_eval,model,eval_loader,loss_func)
	