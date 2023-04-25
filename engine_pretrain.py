import math
import sys
from typing import Iterable
import copy

import torch
from torch.nn import functional as F

import util.misc as misc
import util.lr_sched as lr_sched
from util.misc import clip_gradients, cancel_gradients_last_layer


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer, criterion: torch.nn.Module,
                    device: torch.device, epoch: int, 
                    lr_schedule=None, wd_schedule=None, momentum_schedule=None,fp16_scaler=None,
                    log_writer=None, args=None):
    
    model.train(True)
    
    metric_logger = misc.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('crop_loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('pc_loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('rot_pre_loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq
    
    for it, (inputs) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        views = [view.to(device, non_blocking=True) for view in inputs[0]]
        bboxs = {"all": inputs[1]["all"].to(device, non_blocking=True), 
                 "gc": inputs[1]["gc"].to(device, non_blocking=True),
                 "angles": [angle.to(device, non_blocking=True) for angle in inputs[1]["angles"]]}
        
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]
        m = momentum_schedule[it]
        
        with torch.cuda.amp.autocast(True):

            crop_loss, pc_loss, rot_pre_loss = model(views, bboxs, m)
            
            if args.warmup_epochs < epoch:
                pc_loss_coef = args.pc_loss_coef
            else:
                pc_loss_coef = 0.0
            loss = crop_loss + pc_loss_coef * pc_loss + args.rot_pre_coef * rot_pre_loss
        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = clip_gradients(model, args.clip_grad)
            cancel_gradients_last_layer(epoch, model,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = clip_gradients(model, args.clip_grad)
            cancel_gradients_last_layer(epoch, model,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        
        torch.cuda.synchronize()
        
        metric_logger.update(loss=loss_value)
        metric_logger.update(crop_loss= crop_loss.item())
        metric_logger.update(pc_loss= args.pc_loss_coef * pc_loss.item())
        metric_logger.update(rot_pre_loss= args.rot_pre_coef * rot_pre_loss.item())
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        
    #gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        