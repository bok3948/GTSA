import argparse
import os
import time
import datetime
import json
import numpy as np
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from model_pretrain import gtsa
from engine_pretrain import train_one_epoch

from util.datasets import pretrain_dataset
import util.misc as misc
from util.lr_sched import cosine_scheduler, get_params_groups
from util.gtsa_transforms import GTSATransforms

# %%
def get_args_parser():
    parser = argparse.ArgumentParser(description='GTSA', add_help=False)
    
    #hardware
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--device', default='cuda',
                        help='default(cuda)')
    parser.add_argument('--local_rank', default=0, type=int)
    
    #dataloader
    #/workspace/mmsegmentation/data/ade/ADEChallengeData2016/images/training
    #/workspace/mmdetection/data/coco/train2017
    parser.add_argument('--data', default='/workspace/mmdetection/data/coco/train2017')
    parser.add_argument('--input_size', default=(224, 224), type=int)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size per gpu')
    parser.add_argument('--num_workers', default=10, type=int)
    
    #multi_crop 
    parser.add_argument("--size-crops", type=int, nargs="+", default=[224, 96])
    parser.add_argument("--num-crops", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--min_scale_crops", type=float, nargs="+", default=[0.3, 0.1])
    parser.add_argument("--max_scale_crops", type=float, nargs="+", default=[1., 0.3])
    
    #model
    #backbone
    parser.add_argument('--model', default='gtsa_small', type=str)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--momentum_teacher', default=0.996, type=float)
    
    #projector & predictor
    parser.add_argument('--projector_dim', default=384, type=int)
    parser.add_argument('--predictor_dim', default=768, type=int)
    parser.add_argument('--out_dim', default=768, type=int)
    
    parser.add_argument('--roi_out_size', type=int, nargs="+", default=[10, 4])

    #loss
    parser.add_argument('--pc_loss_coef', type=float, default=0.5)
    parser.add_argument('--num_matches', type=tuple, default=(98, 18))
    parser.add_argument('--rot_pre_coef', type=float, default=0.5)
    
    #optimizer
    parser.add_argument('--blr', default=0.0005, type=float)
    parser.add_argument('--use_fp16', type=bool, default=True)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--weight_decay', type=float, default=0.04)
    parser.add_argument('--clip_grad', type=float, default=3.0)
    parser.add_argument('--weight_decay_end', type=float, default=0.4)
    parser.add_argument('--freeze_last_layer', default=1, type=int)

    #run
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    #print
    parser.add_argument('--print_freq', default=500, type=int)
    
    #save
    parser.add_argument('--output_dir', default='./checkpoint_pretrain', type=str)
    parser.add_argument('--log_dir', default='./log_pretrain', type=str)
    
    return parser

def main(args):
    misc.init_distributed_mode(args)
    
    print(f'job dir: {os.path.dirname(os.path.realpath(__file__))}')
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    args.dir_name = os.path.dirname(os.path.realpath(__file__))
    
    device = torch.device(args.device)
    
    seed = 0 + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cudnn.benchmark = True
          
    #Augmentation
    train_transform = GTSATransforms(size_crops=args.size_crops,
                 nmb_crops=args.num_crops,
                 min_scale_crops=args.min_scale_crops,
                 max_scale_crops=args.max_scale_crops)
    
    
    train_dataset = pretrain_dataset(args.data, transform=train_transform)
    print(f"Data loaded: there are {len(train_dataset)} images.")
    
    #Sampler
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    
    #Dataloader
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, 
        batch_size=args.batch_size, num_workers=args.num_workers,
        drop_last=True
    )
    
    eff_batch_size = args.batch_size * misc.get_world_size()
    args.num_gpus = misc.get_world_size()
    args.eff_batch_size = eff_batch_size
    print(f'effective batch size: {eff_batch_size}')
    
    #Model
    model = gtsa.__dict__[args.model](img_size=args.input_size, patch_size=args.patch_size,
                                      projector_dim=args.projector_dim, predictor_dim=args.predictor_dim,
                                      out_dim=args.out_dim, args=args)
    model = model.cuda()
    
    print(model)
    
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # we need DDP wrapper to have synchro batch norms working...
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    print(f"Model are built: {args.model} network.")
 
    #Optimizer
    args.lr = args.blr * eff_batch_size / 256
    
    print(f'base learning rate: {args.blr}')
    print(f'actual learning rate: {args.lr}')
    
    params_groups = get_params_groups(model)
    optimizer = torch.optim.AdamW(params_groups)
    
    #Loss
    loss_scaler = None
    if args.use_fp16:
        loss_scaler = torch.cuda.amp.GradScaler()
    
    #scheduler
    lr_schedule = cosine_scheduler(
        args.lr ,  
        args.min_lr,
        args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(train_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(train_loader))
    #Loss
    #loss defined in model
    criterion = None
    loss_scaler = None
    
    if args.use_fp16:
        loss_scaler = torch.cuda.amp.GradScaler()

    #Resume
    if len(args.resume) > 3:
         misc.load_model(args, model_without_ddp, optimizer, loss_scaler)
    
    #Run
    print(f'Start training for {args.epochs} epochs')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,  train_loader, optimizer,
            criterion, device, epoch, 
            lr_schedule, wd_schedule, momentum_schedule, loss_scaler,
            None, args
        )
        
        #Save
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 ==args.epochs):
            misc.save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                          'epoch': epoch,}
        
        if args.log_dir and misc.is_main_process():
            with open('pretrain_log.txt', mode='a', encoding='utf-8') as f:
                f.write(json.dumps(log_stats) + '\n')
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')
    
# %%
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
    