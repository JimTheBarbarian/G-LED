'''
This is for training non auto-regressive deep learning models, originally designed for LSTF,
on Kuramoto-Sivashinsky equation data.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import random
import argparse
from tqdm import tqdm
sys.path.insert(0, './util')
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
sys.path.insert(0, './data')
# Corrected import based on other files in the workspace
from data_ks_preprocess import bfs_dataset
from utils import save_model, is_main_process, setup_for_distributed # Assuming setup_for_distributed exists
from torch.utils.data import DataLoader, Dataset
# Assuming these models exist in forecasting_models.py
from forecasting_models.FWin import FWin
from forecasting_models.informer import informer
from forecasting_models.iTransformer import iTransformer
from forecasting_models.Spectcaster import Spectcaster
from forecasting_models.DLinear import DLinear
from layers.embed import DataEmbedding, DataEmbedding_inverted
from layers.FWin_attentions import FullAttention as FullFwin, ProbAttention as ProbFWin, AttentionLayerWin as AttnLayerFWin, AttentionLayerCrossWin as AttnLayerCrossFWin

from layers.SelfAttention_Family import ProbAttention, FullAttention, AttentionLayerWin, AttentionLayerCrossWin
from layers.Transformer_EncDec import ConvLayer, EncoderLayer, Encoder, FourierMix, DecoderLayerWithFourier, Decoder


def train_test_seq(args, model_real, train_loader, sampler_train, valid_loader, test_loader,optimizer_real, scheduler_real, num_epochs):
    best_val_mre = float('inf')
    down_sampler = torch.nn.Upsample(size=(1,16),mode='bilinear')
    for epoch in range(num_epochs):
        if is_main_process():
            print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        tic = time.time()
        if args.distributed and sampler_train is not None:
            sampler_train.set_epoch(epoch)

        # Training Step
        train_loss = train_epoch(args, model_real, train_loader, optimizer_real,down_sampler=down_sampler, device=args.gpu) # Use args.gpu from setup
        if is_main_process():
            print(f"Epoch {epoch+1} Train Loss: {train_loss:.6f}")

        # Validation Step (only on main process for efficiency)
        #if is_main_process():
        avg_loss = valid_epoch(args, model_real,valid_loader,device=args.gpu, down_sampler=down_sampler) # Use model.module for validation
        print(f"Epoch {epoch+1} Validation Loss: {avg_loss:.6f}")
            # max_mre, min_mre, mean_mre, sigma3 = valid_epoch(args, model_real.module, model_imag.module,valid_loader, device=args.gpu, down_sampler=down_sampler) # Use model.module for validation
            #print(f'Validation - Max MRE: {max_mre:.4f}, Mean MRE: {mean_mre:.4f}, Min MRE: {min_mre:.4f}, 3 Sigma: {sigma3:.4f}')
        print(f'Time for Epoch {epoch+1}: {time.time()-tic:.2f}s')

            # Save model based on validation performance
        if avg_loss < best_val_mre:
            best_val_mre = avg_loss
            save_path_real = os.path.join(args.output_dir, f"{args.model_name}_{args.pred_len}_best.pt")
            #save_path_imag = os.path.join(args.output_dir, f"{args.model_name}_imag_best.pt")
            torch.save(model_real.module.state_dict(), save_path_real)
                #torch.save(model_imag.module.state_dict(), save_path_imag)
            print(f"New best models saved to {save_path_real} ")

        # Step the scheduler
        scheduler_real.step()


        # Ensure all processes sync before next epoch
        #if args.distributed:
        #    dist.barrier()

    # Final Test Step (only on main process)
    #error_curve = None
    if is_main_process():
        print("\n--- Final Testing ---")
        # Load best model for testing
        best_model_path = os.path.join(args.output_dir, f"{args.model_name}_{args.pred_len}_best.pt")

        if os.path.exists(best_model_path):
            if args.model_name == 'informer':
                model_instance_real = informer(args).to(args.gpu)
            elif args.model_name == 'iTransformer':
                model_instance_real = iTransformer(args).to(args.gpu)
            elif args.model_name == 'Spectcaster':
                model_instance_real = Spectcaster(args).to(args.gpu)
            elif args.model_name == 'DLinear':
                model_instance_real = DLinear(input_len=args.input_len, output_len=args.pred_len, individual=True, input_features=args.enc_in, output_features=args.c_out).to(args.gpu)
            else: 
                model_instance_real = FWin(seq_len=args.input_len, label_len=args.label_len, out_len=args.pred_len, enc_in=args.enc_in, dec_in=args.dec_in, c_out=args.c_out, window_size=args.window_size, attn='prob', num_windows=args.num_windows, d_model=args.d_model, d_ff=args.d_ff).to(args.gpu)
            model_instance_real.load_state_dict(torch.load(best_model_path, map_location=f'cuda:{args.gpu}'))
            model_instance_real.eval()


        print(f"Loaded best real model from {best_model_path}  for testing.")
            #error_curve = test_epoch(args, model_instance_real, model_instance_imag, test_loader, device=args.gpu, down_sampler=down_sampler) # Test the best single models
        test_loss = valid_epoch(args, model_instance_real, test_loader, device=args.gpu, down_sampler=down_sampler) # Test the best single models
        print(f"Test Error Curve (Mean Relative Error per Step): {test_loss}")
    else:
        print("No best real and/or imaginary model found to test.")



'''
        best_model_path_real = os.path.join(args.output_dir, f"{args.model_name}_real_best.pt")
        best_model_path_imag = os.path.join(args.output_dir, f"{args.model_name}_imag_best.pt")

        if os.path.exists(best_model_path_real) and os.path.exists(best_model_path_imag):
            if args.model_name == 'informer':
                model_instance_real = informer(args).to(args.gpu)
            elif args.model_name == 'iTransformer':
                model_instance_real = iTransformer(args).to(args.gpu)
            elif args.model_name == 'Spectcaster':
                model_instance_real = Spectcaster(args).to(args.gpu)
            elif args.model_name == 'DLinear':
                model_instance_real = DLinear(input_len=args.input_len, output_len=args.pred_len, individual=True, input_features=args.enc_in, output_features=args.c_out).to(args.gpu)
            else: 
                model_instance_real = FWin(seq_len=args.input_len, label_len=args.label_len, out_len=args.pred_len, enc_in=args.enc_in, dec_in=args.dec_in, c_out=args.c_out, window_size=args.window_size, attn='prob', num_windows=args.num_windows, d_model=args.d_model, d_ff=args.d_ff).to(args.gpu)
            model_instance_real.load_state_dict(torch.load(best_model_path_real, map_location=f'cuda:{args.gpu}'))
            model_instance_real.eval()

            if args.model_name == 'informer':
                model_instance_imag = informer(args).to(args.gpu)
            elif args.model_name == 'iTransformer':
                model_instance_imag = iTransformer(args).to(args.gpu)
            elif args.model_name == 'Spectcaster':
                model_instance_imag = Spectcaster(args).to(args.gpu)
            elif args.model_name == 'DLinear':
                model_instance_imag = DLinear(input_len=args.input_len, output_len=args.pred_len, individual=True, input_features=args.enc_in, output_features=args.c_out).to(args.gpu)
            else: 
                model_instance_imag = FWin(seq_len=args.input_len, label_len=args.label_len, out_len=args.pred_len, enc_in=args.enc_in, dec_in=args.dec_in, c_out=args.c_out, window_size=args.window_size, attn='prob', num_windows=args.num_windows, d_model=args.d_model, d_ff=args.d_ff).to(args.gpu)
            model_instance_imag.load_state_dict(torch.load(best_model_path_imag, map_location=f'cuda:{args.gpu}'))
            model_instance_imag.eval()
            '''


def train_epoch(args,model, train_loader, optimizer,device,down_sampler):
    model.train()
    total_loss = 0.0
    loader_wrapper = tqdm(train_loader) if is_main_process() else train_loader

    num_pred_segments = (args.sample_len - args.input_len) // args.pred_len 
    #total_pred_len = num_pred_segments * args.pred_len

    for batch_idx, (batch) in enumerate(loader_wrapper):
        batch = batch.to(device).float()
        if args.downsample:
            batch_min = torch.min(batch)
            batch_max = torch.max(batch)
            normalized_batch = (batch - batch_min) / (batch_max - batch_min + 1e-8) # Normalize to [0, 1]
            #data_spectral = torch.fft.rfft(batch-torch.mean(batch,dim = (1,2), keepdim=True),dim=-1)
            #data_spectral_real = data_spectral.real
            #data_spectral_imag = data_spectral.imag
            #data_coarse2fine = torch.fft.irfft(data_spectral, axis=-1,n=17)
            b_size = batch.shape[0]
            num_time = batch.shape[1]
		    #num_velocity = 2 # Not doing BFS equations
            #batch = batch.reshape([b_size,num_time, 1,64])
            #batch_coarse = down_sampler(batch).reshape([b_size, 
			#										num_time, 
	
			#										args.coarse_dim])
            num_predicts = (num_time - args.input_len - args.pred_len + 1) // args.stride

            for j in range(0,num_predicts):
                optimizer.zero_grad()
                start_idx = j * args.stride
                target_start_idx = start_idx + args.input_len
                current_input = normalized_batch[:, start_idx:start_idx + args.input_len, :]
                ground_truth = normalized_batch[:, target_start_idx:target_start_idx + args.pred_len, :]
                if args.decoder: 
                    label_start_idx = args.input_len - args.label_len
                    label = torch.cat([current_input[:, label_start_idx:args.input_len, :], torch.zeros((current_input.shape[0],args.pred_len, current_input.shape[2]),device = device)], dim=1)
                    output = model(current_input,label)
                    loss = F.mse_loss(output, ground_truth)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() / ( num_predicts)
    avg_loss = total_loss / len(train_loader)
    return avg_loss

        # below is for frequency training
'''
                optimizer_real.zero_grad()
                optimizer_imag.zero_grad()
                current_input_real = data_spectral_real[:, j:j + args.input_len, :]
                current_input_imag = data_spectral_imag[:, j:j + args.input_len, :]
                ground_truth_real = data_spectral_real[:, j + args.input_len:j + args.input_len + args.pred_len, :]
                ground_truth_imag = data_spectral_imag[:, j + args.input_len:j + args.input_len + args.pred_len, :]

                if args.decoder:
                    label_start_idx = args.input_len - args.label_len
                    label_real = torch.cat([current_input_real[:, label_start_idx:args.input_len, :], torch.zeros((current_input_real.shape[0],args.pred_len, current_input_real.shape[2]),device = device)], dim=1)
                    label_imag = torch.cat([current_input_imag[:, label_start_idx:args.input_len, :], torch.zeros((current_input_imag.shape[0],args.pred_len, current_input_imag.shape[2]),device = device)], dim=1)
                    output_real = model_real(current_input_real,label_real)
                    output_imag = model_imag(current_input_imag,label_imag)
                loss_real = F.mse_loss(output_real, ground_truth_real)
                loss_imag = F.mse_loss(output_imag, ground_truth_imag)
                total_loss += (loss_real.item() +loss_imag.item())/ (num_pred_segments * num_predicts)
                loss_real.backward()
                loss_imag.backward()
                optimizer_real.step()
                optimizer_imag.step()
                

        for i in range(num_pred_segments):
            
            all_predictions = []
                label_start_idx = args.input_len - args.label_len
                label = torch.cat([current_input[:, label_start_idx:args.input_len, :], torch.zeros((current_input.shape[0],args.pred_len, current_input.shape[2]),device = device)], dim=1)
                optimizer.zero_grad()
                next_segment = model(current_input, label)
                all_predictions.append(next_segment)

                if args.input_len == args.pred_len:
                    current_input = next_segment
                else:
                    combined = torch.cat([current_input[:, args.pred_len:, :], next_segment], dim=1)
                    current_input = combined 
            
            final_predictions = torch.cat(all_predictions, dim=1) # Concatenate all predictions
            

            comparison_len = min(final_predictions.shape[1], ground_truth.shape[1])
            loss = F.mse_loss(final_predictions[:, :comparison_len, :], ground_truth[:, :comparison_len, :])
            
        
            loss.backward() 
            optimizer.step()

            batch_loss = loss.item()
            num_segments_processed += 1
        
        if num_pred_segments > 0:
            total_loss += batch_loss / num_pred_segments
    
'''

    # Average loss over predictions
    # If you need the exact average loss across all GPUs:
    #if args.distributed:
    #    loss_tensor = torch.tensor(avg_loss).to(device)
    #    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    #    avg_loss = loss_tensor.item()



def valid_epoch(args, model, valid_loader,device,down_sampler):
    model.eval()
    total_loss = 0.0
    #batch_relative_errors= []
    loader_wrapper = tqdm(valid_loader) if is_main_process() else valid_loader

    num_pred_segments = (args.sample_len - args.input_len) // args.pred_len
    total_pred_len = num_pred_segments * args.pred_len

    with torch.no_grad():
        for batch_idx, (data) in enumerate(loader_wrapper):
            data = data.to(device).float()

            data_min = torch.min(data)
            data_max = torch.max(data)
            normalized_batch = (data - data_min) / (data_max - data_min + 1e-8) # Normalize to [0, 1]

            num_time = data.shape[1]
            num_predicts = (num_time - args.input_len - args.pred_len + 1) // args.stride

            for j in range(0,num_predicts):
                start_idx = j * args.stride
                target_start_idx = start_idx + args.input_len
                current_input = normalized_batch[:, start_idx:start_idx + args.input_len, :]
                ground_truth = normalized_batch[:, target_start_idx:target_start_idx + args.pred_len, :]

                if args.decoder: 
                    label_start_idx = args.input_len - args.label_len
                    label = torch.cat([current_input[:, label_start_idx:args.input_len, :], torch.zeros((current_input.shape[0],args.pred_len, current_input.shape[2]),device = device)], dim=1)
                    output = model(current_input,label)
                    loss = F.mse_loss(output, ground_truth)
                    total_loss += loss.item() / ( num_predicts)
                else:
                    final_predictions_reconstructed = model(current_input)
                    total_loss += F.mse_loss(final_predictions_reconstructed, ground_truth) / (num_pred_segments * num_predicts)

        avg_loss = total_loss / len(valid_loader)
        return avg_loss
            #comparison_len = min(final_predictions_reconstructed.shape[1], ground_truth_spatial.shape[1])
            #batch_mse = F.mse_loss(final_predictions_reconstructed[:, :comparison_len, :], ground_truth_spatial[:, :comparison_len, :])

            #target_norm_sq = F.mse_loss(ground_truth_spatial[:, :comparison_len, :], torch.zeros_like(ground_truth_spatial[:, :comparison_len, :]))

            #atch_r_er = batch_mse / target_norm_sq if target_norm_sq > 1e-9 else torch.tensor(0.0, device=device)
            #batch_relative_errors.append(batch_r_er.item())

            

    #errors_np = np.array(batch_relative_errors)

    #max_mre = np.max(errors_np)
    #min_mre = np.min(errors_np)
    #mean_mre = np.mean(errors_np)
    #sigma3 = np.std(errors_np) * 3




    '''
            if args.downsample:
                data_spectral = torch.fft.rfft(data-torch.mean(data,dim = (1,2), keepdim=True),dim=-1)
                data_spectral_real = data_spectral.real
                data_spectral_imag = data_spectral.imag
                #data_coarse2fine = torch.fft.irfft(data_spectral, axis=-1,n=17)
                b_size = data.shape[0]
                num_time = data.shape[1]
                #data = data.reshape([b_size,num_time, 1,64])
                #data_coarse = down_sampler(data).reshape([b_size, 
                #                                    num_time, 
    
                #                                    args.coarse_dim])
                #data = data_coarse
            current_input_real = data_spectral_real[:, :args.input_len, :]
            current_input_imag = data_spectral_imag[:, :args.input_len, :]
            all_predictions_real = []
            all_predictions_imag = []

            original_spatial_dim_size = data.shape[-1] # Get original spatial dimension size from input data
            ground_truth_spatial = data[:, args.input_len: args.input_len + total_pred_len, :]

            for i in range(num_pred_segments):
                label_start_idx = args.input_len - args.label_len
                
                label_real = torch.cat([current_input_real[:, label_start_idx:args.input_len, :], torch.zeros((current_input_real.shape[0],args.pred_len, current_input_real.shape[2]),device = device)], dim=1)
                label_imag = torch.cat([current_input_imag[:, label_start_idx:args.input_len, :], torch.zeros((current_input_imag.shape[0],args.pred_len, current_input_imag.shape[2]),device = device)], dim=1)
                
                next_segment_real = model_real(current_input_real, label_real)
                next_segment_imag = model_imag(current_input_imag, label_imag)
                
                all_predictions_real.append(next_segment_real)
                all_predictions_imag.append(next_segment_imag)

                if args.input_len == args.pred_len:
                    current_input_real = next_segment_real
                    current_input_imag = next_segment_imag
                else:
                    current_input_real = torch.cat([current_input_real[:, args.pred_len:, :], next_segment_real], dim=1)
                    current_input_imag = torch.cat([current_input_imag[:, args.pred_len:, :], next_segment_imag], dim=1)
            
            final_predictions_real = torch.cat(all_predictions_real, dim=1)
            final_predictions_imag = torch.cat(all_predictions_imag, dim=1)

            final_predictions_complex = torch.complex(final_predictions_real, final_predictions_imag)
            final_predictions_reconstructed = torch.fft.irfft(final_predictions_complex, n=original_spatial_dim_size, dim=-1)
            '''

    return max_mre, min_mre, mean_mre, sigma3

def test_epoch(args, model_real, model_imag, test_loader,device,down_sampler):
    model_real.eval()
    model_imag.eval()

    num_pred_segments = (args.sample_len - args.input_len) // args.pred_len
    total_pred_len = num_pred_segments * args.pred_len
    # Initialize tensors to store sum of errors and target norms per step
    sum_mse_per_step = torch.zeros(total_pred_len, device=device)
    sum_target_norm_sq_per_step = torch.zeros(total_pred_len, device=device)
    num_samples = 0
    loader_wrapper = tqdm(test_loader) if is_main_process() else test_loader
    with torch.no_grad():
        for batch_idx, (data) in enumerate(loader_wrapper):
            data = data.to(device).float()
            if args.downsample:
                #data_spectral = torch.fft.rfft(data)[:,:,8]
                #data_coarse2fine = torch.fft.irfft(data_spectral, axis=-1,n=17)
                b_size = data.shape[0]
                num_time = data.shape[1]
                #data = data.reshape([b_size,num_time, 1,64])
                #data_coarse = down_sampler(data).reshape([b_size, 
                #                                    num_time, 
    
                #                                    args.coarse_dim])
                #data = data_coarse
            
            ground_truth = data[:, args.input_len: args.input_len + total_pred_len, :]
            original_spatial_dim_size = data.shape[-1]

            data_spectral = torch.fft.rfft(data - torch.mean(data, dim=(1,2), keepdim=True), dim=-1)
            data_spectral_real = data_spectral.real
            data_spectral_imag = data_spectral.imag

            current_input_real = data_spectral_real[:, :args.input_len, :]
            current_input_imag = data_spectral_imag[:, :args.input_len, :]
            all_predictions_real = []
            all_predictions_imag = []

            for i in range(num_pred_segments):
                label_start_idx = args.input_len - args.label_len
                
                label_real = torch.cat([current_input_real[:, label_start_idx:args.input_len, :], torch.zeros((current_input_real.shape[0],args.pred_len, current_input_real.shape[2]),device = device)], dim=1)
                label_imag = torch.cat([current_input_imag[:, label_start_idx:args.input_len, :], torch.zeros((current_input_imag.shape[0],args.pred_len, current_input_imag.shape[2]),device = device)], dim=1)
                
                next_segment_real = model_real(current_input_real, label_real)
                next_segment_imag = model_imag(current_input_imag, label_imag)
                
                all_predictions_real.append(next_segment_real)
                all_predictions_imag.append(next_segment_imag)

                if args.input_len == args.pred_len:
                    current_input_real = next_segment_real
                    current_input_imag = next_segment_imag
                else:
                    current_input_real = torch.cat([current_input_real[:, args.pred_len:, :], next_segment_real], dim=1)
                    current_input_imag = torch.cat([current_input_imag[:, args.pred_len:, :], next_segment_imag], dim=1)
            
            final_predictions_real = torch.cat(all_predictions_real, dim=1)
            final_predictions_imag = torch.cat(all_predictions_imag, dim=1)

            final_predictions_complex = torch.complex(final_predictions_real, final_predictions_imag)
            final_predictions_reconstructed = torch.fft.irfft(final_predictions_complex, n=original_spatial_dim_size, dim=-1)
            
            final_predictions = final_predictions_reconstructed
            comparison_len = min(final_predictions.shape[1], ground_truth.shape[1])




            # Calculate MSE per prediction step: Mean over Batch and Features



            # Calculate MSE per prediction step: Mean over Batch and Features
            batch_mse_per_step = torch.mean((final_predictions[:,:comparison_len,:] - ground_truth[:,:comparison_len,:]) ** 2, dim=(0, 2)) # Shape: [PredLen]
            # Calculate Target Norm Squared per prediction step: Mean over Batch and Features
            batch_target_norm_sq_per_step = torch.mean(ground_truth[:,:comparison_len,:]**2, dim=(0, 2)) # Shape: [PredLen]

            batch_size = data.shape[0]
            sum_mse_per_step[:comparison_len] += batch_mse_per_step * batch_size
            sum_target_norm_sq_per_step[:comparison_len] += batch_target_norm_sq_per_step * batch_size
            num_samples += batch_size

    # Calculate mean relative error per step, avoiding division by zero
    mean_relative_errors_per_step = torch.zeros(total_pred_len, device=device)
    
    valid_indices = sum_target_norm_sq_per_step > 1e-9
    mean_relative_errors_per_step[valid_indices] = sum_mse_per_step[valid_indices] / sum_target_norm_sq_per_step[valid_indices]
    start = torch.zeros(args.input_len, device=device)
    mean_relative_errors_per_step = torch.cat((start, mean_relative_errors_per_step), dim=0) # Concatenate zeros for input length
    mean_relative_errors_per_step_np = mean_relative_errors_per_step.cpu().numpy()
    return mean_relative_errors_per_step_np



def main():
    parser = argparse.ArgumentParser(description='Train Forecasting Models on KS Data with DDP')

    # --- Data Arguments ---
    parser.add_argument('--data_location', type=str, default=['./data/data0.npy', './data/data1.npy'], help='Path(s) to data files')
    # trajec_max_len for bfs_dataset is the length of sequence to load per sample
    parser.add_argument('--sample_len', type=int, default=385, help='Total sequence length for each sample (input_len + pred_len)')
    parser.add_argument('--val_split', type=float, default=0.9, help='Validation split fraction')
    parser.add_argument('--test_split', type=float, default=0.95, help='Test split fraction (relative to total)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')

    # --- Model Arguments ---
    parser.add_argument('--model_name', type=str, help='Name of the forecasting model')
    parser.add_argument('--input_len', type=int, default=64, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=32, help='Prediction sequence length')
    parser.add_argument('--label_len', type=int, default=32, help='Label length for decoder input (overlap with input)')
    parser.add_argument('--enc_in', type=int, default=65, help='Encoder input size (spatial dimension)') # Assuming 64 based on other files
    parser.add_argument('--dec_in', type=int, default=65, help='Decoder input size (spatial dimension)')
    parser.add_argument('--c_out', type=int, default=65, help='Output size (spatial dimension)')
    parser.add_argument('--factor', type=int, default=5, help='')
    parser.add_argument('--window_size', type=int, default=16, help='')
    parser.add_argument('--num_windows', type=int, default=2, help='')
    parser.add_argument('--attn', type=str, default='prob', help='Attention type (prob/full)')
    parser.add_argument('--distil', action='store_false', help='')
    parser.add_argument('--patch_len', type = int, default=16, help='')
    parser.add_argument('--depth', type = int, default=3, help='')
    # Add common transformer args (might be ignored by simpler models)
    parser.add_argument('--d_model', type=int, default=256, help='Dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--decoder',action='store_false', help='Use decoder')
    parser.add_argument('--output_attention', action='store_true', help='Output attention weights')
    # DLinear specific
    parser.add_argument('--individual', action='store_true', help='Individual channels for DLinear')


    # --- Training Arguments ---
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--scheduler_step_size', type=int, default=10, help='StepLR step size')
    parser.add_argument('--scheduler_gamma', type=float, default=0.9, help='StepLR gamma')
    parser.add_argument('--output_dir', type=str, default='./forecasting_output_ms', help='Directory to save results')
    parser.add_argument('--downsample', action='store_false', help='Use downsampling')
    parser.add_argument('--coarse_dim', type=int, default=16, help='Coarse dimension for downsampling')
    parser.add_argument('--stride', type=int, default=10, help='Stride for downsampling')
    # local_rank is handled by torchrun/launch
    parser.add_argument('--seed', type=int, default=20398, help='Random seed')


    args = parser.parse_args()



    # --- Distributed Setup ---
    # setup_for_distributed should initialize process group, set rank, world_size, gpu
    # and potentially set args.distributed = True
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        if args.local_rank != -1:
            
            args.device = f'cuda:{args.local_rank}'
            torch.cuda.set_device(args.local_rank)
        else:
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # +++ Initialize distributed environment
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.distributed = args.world_size > 1
    else:
        args.world_size = 1
        args.distributed = False

    if args.distributed:
        print(f"Initializing process group for rank {args.local_rank}...")
        dist.init_process_group(backend='nccl', init_method='env://')
        # Ensure setup_for_distributed is called after init_process_group
        setup_for_distributed(args.local_rank == 0) # Pass is_main_process flag
        print(f"Process group initialized for rank {args.local_rank}.")
    else:
        # Still call setup for non-distributed case to set the flag
        setup_for_distributed(True) 

    args.gpu = args.local_rank if args.distributed else 0 # Set GPU for DDP or single process
    device = torch.device('cuda',args.gpu) # Use the device set above

    # --- Seed ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Create Output Dir ---
    if is_main_process() and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")


    # --- Load Data ---
    if is_main_process(): print("Loading datasets...")
    # Pass sample_len as trajec_max_len to bfs_dataset
    train_dataset = bfs_dataset(data_location=args.data_location, trajec_max_len=args.sample_len, val_split=args.val_split, test_split=args.test_split, flag='train')
    valid_dataset = bfs_dataset(data_location=args.data_location, trajec_max_len=args.sample_len, val_split=args.val_split, test_split=args.test_split, flag='val')
    test_dataset = bfs_dataset(data_location=args.data_location, trajec_max_len=args.sample_len, val_split=args.val_split, test_split=args.test_split, flag='test')

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if args.distributed else None
    valid_sampler = None # Typically no need to sample validation/test in DDP
    test_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True, shuffle=(train_sampler is None))
    # Use larger batch size for validation/testing if memory allows
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size , sampler=valid_sampler, num_workers=args.num_workers, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size , sampler=test_sampler, num_workers=args.num_workers, pin_memory=True, shuffle=False)
    if is_main_process(): print("Datasets loaded.")

    # --- Instantiate Model ---
    if is_main_process(): print("Datasets loaded.")

    # --- Instantiate Model ---
    base_output_dir = args.output_dir 
    for model_name in ['Spectcaster','FWin','DLinear']:
        args.model_name = model_name
        args.output_dir = os.path.join(base_output_dir, args.model_name) # Set model-specific output dir
        if is_main_process():
            os.makedirs(args.output_dir, exist_ok=True) 
        if is_main_process(): print(f"Creating models: {args.model_name}")

        # Instantiate real model
        if args.model_name == 'informer':
            args.num_epochs = 10
            args.label_len = 32
            model_real = informer(args).to(device)
        elif args.model_name == 'iTransformer':
            args.num_epochs = 10
            args.label_len = 32
            model_real = iTransformer(args).to(device)
        elif args.model_name == 'Spectcaster':
            args.label_len = 32
            model_real = Spectcaster(args).to(device)
            if args.distributed: model_real = nn.SyncBatchNorm.convert_sync_batchnorm(model_real)
        elif args.model_name == 'DLinear':
            model_real = DLinear(input_len = args.input_len, output_len = args.pred_len, individual = True, input_features = args.enc_in, output_features = args.c_out)
            model_real = model_real.to(device)
        else: # FWin as default
            args.label_len = 32 
            args.num_epochs = 10
            model_real = FWin(seq_len=args.input_len, label_len = args.label_len, out_len=args.pred_len, enc_in=args.enc_in,dec_in=args.dec_in,c_out=args.c_out,window_size=args.window_size,attn = 'prob',num_windows=args.num_windows,d_model = args.d_model, d_ff = args.d_ff).to(device)
        '''
        # Instantiate imaginary model
        if args.model_name == 'informer':
            model_imag = informer(args).to(device)
        elif args.model_name == 'iTransformer':
            model_imag = iTransformer(args).to(device)
        elif args.model_name == 'Spectcaster':
            model_imag = Spectcaster(args).to(device)
            if args.distributed: model_imag = nn.SyncBatchNorm.convert_sync_batchnorm(model_imag)
        elif args.model_name == 'DLinear':
            model_imag = DLinear(input_len = args.input_len, output_len = args.pred_len, individual = True, input_features = args.enc_in, output_features = args.c_out)
            model_imag = model_imag.to(device)
        else: # FWin as default
            model_imag = FWin(seq_len=args.input_len, label_len = args.label_len, out_len=args.pred_len, enc_in=args.enc_in,dec_in=args.dec_in,c_out=args.c_out,window_size=args.window_size,attn = 'prob',num_windows=args.num_windows,d_model = args.d_model, d_ff = args.d_ff).to(device)
        '''
        if args.distributed:
            model_real = DDP(model_real, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
            #model_imag = DDP(model_imag, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
        if is_main_process(): print(f"Models created on device: {device}")


        # --- Optimizer and Scheduler ---
        optimizer_real = optim.Adam(model_real.parameters(), lr=args.learning_rate)
        scheduler_real = optim.lr_scheduler.StepLR(optimizer_real, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
        #optimizer_imag = optim.Adam(model_imag.parameters(), lr=args.learning_rate)
        #scheduler_imag = optim.lr_scheduler.StepLR(optimizer_imag, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

        # --- Run Training ---
        if is_main_process():
            print(f"\nStarting training for {args.model_name}...")

        error_curve = train_test_seq(
            args=args,
            model_real=model_real, 
            #model_imag=model_imag,
            train_loader=train_loader,
            sampler_train=train_sampler,
            valid_loader=valid_loader,
            test_loader=test_loader,
            optimizer_real=optimizer_real,
            #optimizer_imag=optimizer_imag,
            scheduler_real=scheduler_real,
            #scheduler_imag=scheduler_imag,
            num_epochs=args.num_epochs
        )

    # --- Save Final Results (Optional) ---
        if is_main_process():
            print(f"\nTraining finished for {args.model_name}.")
        # Save the final model state dict (optional, usually best model is preferred)
        # model_save_path = os.path.join(args.output_dir, f"{args.model_name}_final_epoch.pt")
        # torch.save(final_model.module.state_dict(), model_save_path) # Save the underlying model
        # print(f"Final epoch model saved to {model_save_path}")

        # Save error curve if generated
        '''
            if error_curve is not None:
                plt.figure(figsize=(10, 6))
                t_values = np.arange(0, len(error_curve)) # Time step indices
                plt.semilogy(t_values, error_curve) # Use absolute time steps for x-axis
                plt.xlabel('t (Time Step Index)')
                plt.ylabel('e(t) (Mean Relative Error)')
                plt.title(f'{args.model_name} - Test Error Curve')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                # Add vertical line at the end of the input sequence
                plt.axvline(x=args.input_len, color='r', linestyle='--', label=f'Input End (t={args.input_len})')

                # Adjust x-limits to show context around input/prediction boundary
                plot_xlim_start = 0
                plot_xlim_end = args.sample_len
                plt.xlim(plot_xlim_start, plot_xlim_end)

                # Add legend
                plt.legend()

                plot_path = os.path.join(args.output_dir, f"{args.model_name}_test_error_plot.png")
                plt.savefig(plot_path)
                plt.close() # Close the plot to free memory
                print(f"Test error plot saved to {plot_path}")

        '''
    # --- Cleanup ---
    if args.distributed:
        dist.destroy_process_group()
    print("Script finished.")


if __name__ == "__main__":
    main()
