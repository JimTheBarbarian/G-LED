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
from layers.embed import DataEmbedding, DataEmbedding_inverted

from layers.SelfAttention_Family import ProbAttention, FullAttention, AttentionLayerWin, AttentionLayerCrossWin
from layers.Transformer_EncDec import ConvLayer, EncoderLayer, Encoder, FourierMix, DecoderLayerWithFourier, Decoder


def train_test_seq(args, model, train_loader, sampler_train, valid_loader, test_loader,optimizer, scheduler, num_epochs):
    best_val_mre = float('inf')
    for epoch in range(num_epochs):
        if is_main_process():
            print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        tic = time.time()
        if args.distributed and sampler_train is not None:
            sampler_train.set_epoch(epoch)

        # Training Step
        train_loss = train_epoch(args, model, train_loader, optimizer, device=args.gpu) # Use args.gpu from setup
        if is_main_process():
            print(f"Epoch {epoch+1} Train Loss: {train_loss:.6f}")

        # Validation Step (only on main process for efficiency)
        if is_main_process():
            max_mre, min_mre, mean_mre, sigma3 = valid_epoch(args, model.module, valid_loader, device=args.gpu) # Use model.module for validation
            print(f'Validation - Max MRE: {max_mre:.4f}, Mean MRE: {mean_mre:.4f}, Min MRE: {min_mre:.4f}, 3 Sigma: {sigma3:.4f}')
            print(f'Time for Epoch {epoch+1}: {time.time()-tic:.2f}s')

            # Save model based on validation performance
            if mean_mre < best_val_mre:
                best_val_mre = mean_mre
                save_path = os.path.join(args.output_dir, f"{args.model_name}_best.pt")
                torch.save(model.module.state_dict(), save_path)
                print(f"New best model saved to {save_path}")

        # Step the scheduler
        scheduler.step()

        # Ensure all processes sync before next epoch
        if args.distributed:
            dist.barrier()

    # Final Test Step (only on main process)
    error_curve = None
    if is_main_process():
        print("\n--- Final Testing ---")
        # Load best model for testing
        best_model_path = os.path.join(args.output_dir, f"{args.model_name}_best.pt")
        if os.path.exists(best_model_path):
             # Need to instantiate a non-DDP model to load state_dict easily
            model_instance = create_model(args, device=args.gpu) # Helper function to create model
            model_instance.load_state_dict(torch.load(best_model_path, map_location=args.gpu))
            model_instance.eval()
            print(f"Loaded best model from {best_model_path} for testing.")
            error_curve = test_epoch(args, model_instance, test_loader, device=args.gpu) # Test the best single model
            print(f"Test Error Curve (Mean Relative Error per Step): {error_curve}")
        else:
            print("No best model found to test.")


    return error_curve # Return the DDP model and error curve (only valid on main process)


def train_epoch(args,model, train_loader, optimizer,device):
    model.train()
    total_loss = 0.0
    # Wrap train_loader with tqdm only on main process
    loader_wrapper = tqdm(train_loader) if is_main_process() else train_loader
    for batch_idx, (data) in enumerate(loader_wrapper):
        data = data.to(device).float()
        target = data[:,args.input_len:,:]
        input_data = data[:,:args.input_len,:] # Renamed variable
        # Prepare label for decoder input (common in seq2seq forecasting)
        label = torch.cat([input_data[:,-args.label_len:,:], torch.zeros_like(target)], dim=1) # Zeros as placeholder for target steps

        optimizer.zero_grad()
        # Forward pass through the DDP model
        output = model(input_data, label)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        # Accumulate loss (gather across processes if needed, but avg loss per epoch is usually sufficient)
        total_loss += loss.item()

    # Average loss over batches
    avg_loss = total_loss / len(train_loader)
    # If you need the exact average loss across all GPUs:
    #if args.distributed:
    #    loss_tensor = torch.tensor(avg_loss).to(device)
    #    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    #    avg_loss = loss_tensor.item()

    return avg_loss


def valid_epoch(args, model, valid_loader,device):
    model.eval()
    batch_relative_errors= []
    loader_wrapper = tqdm(valid_loader) if is_main_process() else valid_loader
    with torch.no_grad():
        for batch_idx, (data) in enumerate(loader_wrapper):
            data = data.to(device).float()
            target = data[:,args.input_len:,:]
            input_data = data[:,:args.input_len,:]
            label = torch.cat([input_data[:,-args.label_len:,:],torch.zeros_like(target)], dim = 1)
            output = model(input_data,label) # Use the non-DDP model instance
            batch_mse = F.mse_loss(output, target)

            # Calculate target norm squared (mean squared value) for relative error
            target_norm_sq = F.mse_loss(target, torch.zeros_like(target))

            # Calculate relative error, handle potential division by zero
            batch_r_er= batch_mse / target_norm_sq if target_norm_sq > 1e-9 else torch.tensor(0.0,device = device)
            batch_relative_errors.append(batch_r_er.item())

    errors_np = np.array(batch_relative_errors)

    max_mre = np.max(errors_np)
    min_mre = np.min(errors_np)
    mean_mre = np.mean(errors_np)
    sigma3 = np.std(errors_np) * 3

    return max_mre, min_mre, mean_mre, sigma3

def test_epoch(args, model, test_loader,device):
    model.eval()
    # Initialize tensors to store sum of errors and target norms per step
    sum_mse_per_step = torch.zeros(args.pred_len, device=device)
    sum_target_norm_sq_per_step = torch.zeros(args.pred_len, device=device)
    num_samples = 0
    loader_wrapper = tqdm(test_loader) if is_main_process() else test_loader
    with torch.no_grad():
        for batch_idx, (data) in enumerate(loader_wrapper):
            data = data.to(device).float()
            target = data[:,args.input_len:,:] # Shape: [Batch, PredLen, Features]
            input_data = data[:,:args.input_len,:]
            label = torch.cat([input_data[:,-args.label_len:,:],torch.zeros_like(target)], dim = 1)
            output = model(input_data,label) # Use the non-DDP model instance

            # Calculate MSE per prediction step: Mean over Batch and Features
            batch_mse_per_step = torch.mean((output - target) ** 2, dim=(0, 2)) # Shape: [PredLen]
            # Calculate Target Norm Squared per prediction step: Mean over Batch and Features
            batch_target_norm_sq_per_step = torch.mean(target**2, dim=(0, 2)) # Shape: [PredLen]

            batch_size = data.shape[0]
            sum_mse_per_step += batch_mse_per_step * batch_size
            sum_target_norm_sq_per_step += batch_target_norm_sq_per_step * batch_size
            num_samples += batch_size

    # Calculate mean relative error per step, avoiding division by zero
    mean_relative_errors_per_step = torch.zeros(args.pred_len, device=device)
    
    valid_indices = sum_target_norm_sq_per_step > 1e-9
    mean_relative_errors_per_step[valid_indices] = sum_mse_per_step[valid_indices] / sum_target_norm_sq_per_step[valid_indices]
    start = torch.zeros(args.input_len, device=device).cpu().numpy()
    mean_relative_errors_per_step = torch.cat((start, mean_relative_errors_per_step), dim=0) # Concatenate zeros for input length
    mean_relative_errors_per_step_np = mean_relative_errors_per_step.cpu().numpy()
    return mean_relative_errors_per_step_np

# Helper function to create model instance (avoids code duplication)
def create_model(args, device):
    # Generic arguments dictionary (adjust based on actual model signatures)
    model_args = {
        'seq_len': args.input_len,
        'pred_len': args.pred_len,
        'label_len': args.label_len,
        'enc_in': args.enc_in,
        'dec_in': args.dec_in,
        'c_out': args.c_out,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'e_layers': args.e_layers,
        'd_layers': args.d_layers,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'activation': args.activation,
        'output_attention': False, # Typically not needed
        'device': device # Pass device if models expect it
    }
    # Filter args based on model type if necessary (e.g., DLinear doesn't need d_model)
    if args.model_name == 'FWin':
         # Assuming FWin has a specific signature
         # Ensure forecasting_models.FWin.Model exists and check its signature
         model = FWin(seq_len=args.input_len, label_len = args.label_len, out_len=args.pred_len, enc_in=args.enc_in,dec_in=args.dec_in,c_out=args.c_out).to(device) # Placeholder signature
    elif args.model_name == 'informer':
        # Informer likely uses many of the transformer args
        # Ensure forecasting_models.informer.Model exists and check its signature
        # You might need to filter model_args to only pass expected arguments
        model = informer(**model_args).to(device)
    elif args.model_name == 'iTransformer':
        # iTransformer likely uses many of the transformer args
        # Ensure forecasting_models.iTransformer.Model exists and check its signature
        # You might need to filter model_args to only pass expected arguments
        model = iTransformer.Model(**model_args).to(device)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")
    return model

def main():
    parser = argparse.ArgumentParser(description='Train Forecasting Models on KS Data with DDP')

    # --- Data Arguments ---
    parser.add_argument('--data_location', type=str, default=['./data/data0.npy', './data/data1.npy'], help='Path(s) to data files')
    # trajec_max_len for bfs_dataset is the length of sequence to load per sample
    parser.add_argument('--sample_len', type=int, default=385, help='Total sequence length for each sample (input_len + pred_len)')
    parser.add_argument('--val_split', type=float, default=0.6, help='Validation split fraction')
    parser.add_argument('--test_split', type=float, default=0.8, help='Test split fraction (relative to total)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')

    # --- Model Arguments ---
    parser.add_argument('--model_name', type=str, help='Name of the forecasting model')
    parser.add_argument('--input_len', type=int, default=64, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=321, help='Prediction sequence length')
    parser.add_argument('--label_len', type=int, default=32, help='Label length for decoder input (overlap with input)')
    parser.add_argument('--enc_in', type=int, default=64, help='Encoder input size (spatial dimension)') # Assuming 64 based on other files
    parser.add_argument('--dec_in', type=int, default=64, help='Decoder input size (spatial dimension)')
    parser.add_argument('--c_out', type=int, default=64, help='Output size (spatial dimension)')
    parser.add_argument('--factor', type=int, default=5, help='')
    # Add common transformer args (might be ignored by simpler models)
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    # DLinear specific
    parser.add_argument('--individual', action='store_true', help='Individual channels for DLinear')


    # --- Training Arguments ---
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--scheduler_step_size', type=int, default=10, help='StepLR step size')
    parser.add_argument('--scheduler_gamma', type=float, default=0.9, help='StepLR gamma')
    parser.add_argument('--output_dir', type=str, default='./forecasting_output', help='Directory to save results')
    # local_rank is handled by torchrun/launch
    parser.add_argument('--seed', type=int, default=42, help='Random seed')


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
    base_output_dir = args.output_dir # Base output directory for saving models
    for model_name in [ 'informer', 'iTransformer']:
        args.model_name = model_name
        args.output_dir = os.path.join(base_output_dir, args.model_name) # Set model-specific output dir
        if is_main_process():
            os.makedirs(args.output_dir, exist_ok=True) # Create model-specific output dir if it doesn't exist
        if is_main_process(): print(f"Creating model: {args.model_name}")
        model = create_model(args, device=device) # Create on the correct device

        if args.distributed:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True) # Adjust find_unused_parameters if needed
        if is_main_process(): print(f"Model created on device: {device}")


        # --- Optimizer and Scheduler ---
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

        # --- Run Training ---
        if is_main_process():
            print(f"\nStarting training for {args.model_name}...")

        error_curve = train_test_seq(
        args=args,
        model=model, # Pass the DDP model
        train_loader=train_loader,
        sampler_train=train_sampler,
        valid_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
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
            if error_curve is not None:
                plt.figure(figsize=(10, 6))
                t_values = np.arange(0, args.pred_len + args.input_len)
                plt.semilogy(t_values, error_curve) # Use absolute time steps for x-axis
                plt.xlabel('t (Time Step Index)')
                plt.ylabel('e(t) (Mean Relative Error)')
                plt.title(f'{args.model_name} - Test Error Curve')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                # Add vertical line at the end of the input sequence
                plt.axvline(x=args.input_len, color='r', linestyle='--', label=f'Input End (t={args.input_len})')

                # Adjust x-limits to show context around input/prediction boundary
                plot_xlim_start = 0
                plot_xlim_end = args.input_len + args.pred_len
                plt.xlim(plot_xlim_start, plot_xlim_end)

                # Add legend
                plt.legend()

                plot_path = os.path.join(args.output_dir, f"{args.model_name}_test_error_plot.png")
                plt.savefig(plot_path)
                plt.close() # Close the plot to free memory
                print(f"Test error plot saved to {plot_path}")
    # --- Cleanup ---
    if args.distributed:
        dist.destroy_process_group()
    print("Script finished.")


if __name__ == "__main__":
    main()
