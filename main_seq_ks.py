# General Package
import argparse
import sys
import pdb
from datetime import datetime
import os
import numpy as np
import json
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Internal package
sys.path.insert(0, './util')
from utils import save_args,is_main_process,setup_for_distributed
sys.path.insert(0, './data')
from data_ks_preprocess import bfs_dataset 
sys.path.insert(0, './transformer')
from sequentialModel import SequentialModel as transformer
sys.path.insert(0, './train_test_seq')
from train_seq import train_seq_shift
import time

class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        """
        for dataset
        """
        self.parser.add_argument("--dataset",
                                 default='ks',
                                 help='name it')
        self.parser.add_argument("--data_location", 
                                 default = ['./data/data0.npy','./data/data1.npy'],
                                 help='the relative or abosolute data.npy file')
        self.parser.add_argument("--trajec_max_len", 
                                 default=319,
                                 help = 'max seq_length (per seq) to train the model')
        self.parser.add_argument("--start_n", 
                                 default=0,
                                 help = 'the starting step of the data')
        self.parser.add_argument("--n_span",
                                 default=320,
                                 help='the total step of the data from the staring step')



        self.parser.add_argument("--trajec_max_len_valid", 
                                 default=384,
                                 help = 'max seq_length (per seq) to valid the model')
        self.parser.add_argument("--start_n_valid", 
                                 default=64,
                                 help = 'the starting step of the data')
        self.parser.add_argument("--n_span_valid",
                                 default=321,
                                 help='the total step of the data from the staring step')
               

        """
        for model
        """
        self.parser.add_argument("--n_layer", 
                                 default =8,#8
                                 help = 'number of attention layer')
        self.parser.add_argument("--output_hidden_states", 
                                 default= True,
                                 help='out put hidden matrix')
        self.parser.add_argument("--output_attentions",
                                 default = True,
                                 help = 'out put attention matrix')
        self.parser.add_argument("--n_ctx",
                                 default = 384,
                                 help='number steps transformer can look back at')
        self.parser.add_argument("--n_embd", 
                                 default =64,
                                 help='The hidden state dim transformer to predict') 
        self.parser.add_argument("--n_head", 
                                 default = 8,
                                 help='number of head per layer')
        self.parser.add_argument("--embd_pdrop",
                                 default = 0.0,
                                 help='T.B.D')
        self.parser.add_argument("--layer_norm_epsilon", 
                                 default=1e-5,
                                 help='############ Do not change')
        self.parser.add_argument("--attn_pdrop", 
                                 default = 0.0,
                                 help='T.B.D')
        self.parser.add_argument("--resid_pdrop", 
                                 default = 0.0,
                                 help='T.B.D')
        self.parser.add_argument("--activation_function", 
                                 default = "relu",
                                 help='Trust OpenAI and Nick')
        self.parser.add_argument("--initializer_range", 
                                 default = 0.02,
                                 help='Trust OpenAI and Nick')
        
        
        """
        for training
        """
        self.parser.add_argument("--start_Nt",
                                 default=1,
                                 help='The starting length of forward propgatate')
        self.parser.add_argument("--d_Nt",
                                 default=1,
                                 help='The change length of forward propgatate')
        self.parser.add_argument("--batch_size",
                                 default=16, #max 16->0.047
                                 help = 'how many seqs you want to train together per bp')
        self.parser.add_argument("--batch_size_valid",
                                 default=32, #max 16->0.047
                                 help = 'how many seqs you want to train together per valid')
        self.parser.add_argument("--shuffle",
                                 default=True,
                                 help = 'shuffle the batch')
        #self.parser.add_argument("--device",
        #                         default='cuda:1')
        self.parser.add_argument("--epoch_num", 
                                 default = 200,
                                 help='epoch_num')
        self.parser.add_argument("--learning_rate", 
                                 default = 3e-5,
                                 help='learning rate')
        self.parser.add_argument("--gamma",
                                 default=0.99083194489,
                                 help='learning rate decay')
        
        self.parser.add_argument("--coarse_dim",
                                 default=16,
                                 help='the coarse shape (hidden) of transformer')
        self.parser.add_argument('--coarse_mode',
                                 default='bilinear',
                                 help='the way of downsampling the snpashot')
        self.parser.add_argument("--march_tol", 
                                 default=0.01,
                                 help='march threshold for Nt + 1')
        self.parser.add_argument("--local_rank",type=int,default=-1)
    def update_args(self):
        args = self.parser.parse_args()
        # +++ Set device based on local rank
        if args.local_rank != -1:
            args.device = f'cuda:{args.local_rank}'
            torch.cuda.set_device(args.device)
        else: # Handle non-distributed case or CPU
             args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args.time = '{0:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        # output dataset
        args.dir_output = 'output/'
        args.fname = args.dataset + '_' +args.time
        args.experiment_path = args.dir_output + args.fname
        args.model_save_path = args.experiment_path + '/' + 'model_save/'
        args.logging_path = args.experiment_path + '/' + 'logging/'
        args.current_model_save_path = args.model_save_path
        args.logging_epoch_path = args.logging_path + 'epoch_history.csv'
        #if not os.path.isdir(args.logging_path):
        #    os.makedirs(args.logging_path)
        #if not os.path.isdir(args.model_save_path):
        #    os.makedirs(args.model_save_path)
        return args








if __name__ == '__main__':
    

    args = Args()
    args = args.update_args()


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

    if is_main_process():
        if not os.path.isdir(args.logging_path):
            os.makedirs(args.logging_path)
        if not os.path.isdir(args.model_save_path):
            os.makedirs(args.model_save_path)
        save_args(args)

    """
    pre-check
    """
    #assert args.coarse_dim == args.n_embd
    #assert args.trajec_max_len_valid == args.n_ctx + 1
    
    """
    fetch data
    """
    print('Start data_set')
    tic = time.time()
    

    data_set_train = bfs_dataset(data_location  = args.data_location,
                                 trajec_max_len = args.trajec_max_len,
                                 start_n        = args.start_n)
                                 #n_span         = args.n_span)
    data_set_test_on_train = bfs_dataset(data_location  = args.data_location,
                                         trajec_max_len = args.trajec_max_len_valid,
                                         start_n        = args.start_n)
                                         #n_span         = args.n_span)
    data_set_valid = bfs_dataset(data_location  = args.data_location,
                                 trajec_max_len = args.trajec_max_len_valid,
                                 start_n        = args.start_n_valid,
                                 flag = 'valid')
    sampler_train = DistributedSampler(data_set_train, shuffle=args.shuffle) if args.distributed else None
    sampler_test_on_train = DistributedSampler(data_set_test_on_train, shuffle=False) if args.distributed else None
    sampler_valid = DistributedSampler(data_set_valid, shuffle=False) if args.distributed else None

    data_loader_train = DataLoader(dataset    = data_set_train,
                                   sampler = sampler_train,
                                   shuffle    = False if args.distributed else args.shuffle,
                                   pin_memory = True,
                                   batch_size = args.batch_size,
                                   num_workers = 4)
                                   
    data_loader_test_on_train = DataLoader(dataset    = data_set_test_on_train,
                                           sampler = sampler_test_on_train,
                                           pin_memory = True,
                                           shuffle    = False,
                                           batch_size = args.batch_size_valid,
                                           num_workers = 4)
    data_loader_valid = DataLoader(dataset    = data_set_valid,
                                   sampler = sampler_valid,
                                   pin_memory = True,
                                   shuffle    = False,
                                   batch_size = args.batch_size_valid,
                                   num_workers = 4)
    if is_main_process(): print(f'Done data-set use time ', {time.time() - tic})
    """
    create model
    """
    #model = transformer(args).to(args.device).float()

    model = transformer(args).to(args.device).float()
    if is_main_process(): print('Number of parameters: {}'.format(model._num_parameters()))
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        #model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    """
    create loss function
    """
    loss_func = torch.nn.MSELoss()
    
    """
    create optimizer
    """
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=args.learning_rate)
    """
    create scheduler
    """
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=1,
                                                gamma=args.gamma)    

    """
    train
    """
    train_seq_shift(args=args, 
                    model=model, 
                    data_loader=data_loader_train, 

                    sampler_train = sampler_train,
                    data_loader_copy = data_loader_test_on_train,
                    data_loader_valid = data_loader_valid,
                    loss_func=loss_func, 
                    optimizer=optimizer,
                    scheduler=scheduler)