import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from spatialtip_model import SpatialTIPModel
import argparse
import os
import torch.distributed as dist
from dataset import SpatialTIPData
from scBERT.utils import CosineAnnealingWarmupRestarts
from utils import cleanup, AvgMeter, get_lr, get_reduced, Cal_Spatial_Net, mlm_mask, tokenize_batch
import torch.nn.functional as F
import tqdm
import numpy as np
import pandas as pd
import math
import random
from functools import reduce
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loader')
parser.add_argument('--save_dir', type=str, default='spatialtip_model', help='model saving directory')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--train_mask_prob', type=float, default=0.6, help='train mask prob')
parser.add_argument('--replace_prob', type=float, default=1.0, help='replace prob')
parser.add_argument('--world_size', type=int, default=4, help='number of processes for distributed training')
parser.add_argument('--finetune_sample', type=str, default='MISC1', help='finetune sample')


def data_loader_fn(args, test_samples):

    hest_dataset = SpatialTIPData(adata_path=f'hest_data/st_healthy_6k_v1/{test_samples}.h5ad',
                                patches_path=f'hest_data/patches/{test_samples}.h5', 
                                vocabulary_path='hest_data/vocabulary_healthy_3k.json')

    train_size = int(0.8 * len(hest_dataset))
    test_size = len(hest_dataset) - train_size
    
    print(f"Train size: {train_size}, Test size: {test_size}")

    train_dataset, test_dataset = torch.utils.data.random_split(hest_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(args.seed))
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=test_sampler, pin_memory=True, drop_last=True)

    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, args, lr_scheduler=None, neighbors=None, scaler=None):
    loss_meter = AvgMeter()
    tqdm_object = tqdm.tqdm(train_loader, total=len(train_loader), disable=dist.get_rank() != 0)

    for batch in tqdm_object:
        batch = {k: v.cuda() for k, v in batch.items() if k == "gene_exp" or k == 'image' or k == 'barcode' or k == 'gene_id' or k == 'tissue'}
        batch['gene_id'] = tokenize_batch(batch['gene_id'], append_cls=True, cls_id=0)
        batch['gene_exp'], batch['labels'] = mlm_mask(batch['gene_exp'], mask_token_id=-1, mask_prob_zero=0.05, mask_prob_nonzero=args.train_mask_prob)

        with torch.amp.autocast('cuda'):
            loss = model(batch, neighbors=neighbors, method='contrastive')

        optimizer.zero_grad()

        scaler.scale(loss).backward()

        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= args.world_size

        scaler.step(optimizer)
        scaler.update()

        count = batch['gene_exp'].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    
    lr_scheduler.step()

    return loss_meter


def test_epoch(model, test_loader, args, neighbors=None, scaler=None):
    loss_meter = AvgMeter()
    tqdm_object = tqdm.tqdm(test_loader, total=len(test_loader), disable=dist.get_rank() != 0)
    for batch in tqdm_object:
        batch = {k: v.cuda() for k, v in batch.items() if k == "gene_exp" or k == 'image' or k == 'barcode' or k == 'gene_id' or k == 'tissue'}
        batch['gene_id'] = tokenize_batch(batch['gene_id'], append_cls=True, cls_id=0)
        batch['gene_exp'], batch['labels'] = mlm_mask(batch['gene_exp'], mask_token_id=-1, mask_prob_zero=0.05, mask_prob_nonzero=args.train_mask_prob)

        with torch.amp.autocast('cuda'):
            loss = model(batch, neighbors=neighbors, method='contrastive')

        count = batch['gene_exp'].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(val_loss=loss_meter.avg)

    return loss_meter


def main():
    print('Start finetuning!')
    args = parser.parse_args()

    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    is_master = rank == 0
    local_rank = int(os.environ['LOCAL_RANK'])
    current_device = local_rank
    torch.cuda.set_device(current_device)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_sample = args.finetune_sample
    train_loader, test_loader = data_loader_fn(args, test_sample)

    neighbors = None # comment this line to use spatial neighbors

    print('Finetune sample:', test_sample)

    model = SpatialTIPModel(vision_model='vit_large_patch16_224').cuda(current_device)

    model_path = os.path.join(args.save_dir, 'spatialtip_hest_healthy_3k.pt')
    print('Model Path:', model_path)
    stat_dict = torch.load(model_path, map_location='cpu')

    model.load_state_dict(stat_dict, strict=False)

    model = DDP(model, device_ids=[current_device], find_unused_parameters=True)
    
    print('Load model!')

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=30,
        cycle_mult=1,
        max_lr=args.learning_rate,
        min_lr=5e-5,
        warmup_steps=2,
        gamma=0.98
    )

    best_loss = float('inf')
    best_epoch = 0
    stop_training = torch.tensor(0).cuda()

    # Finetune    
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch + 1}")

        train_loader.sampler.set_epoch(epoch)
        
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, args, lr_scheduler=scheduler, neighbors=neighbors, scaler=scaler)

        dist.barrier()

        train_loss = get_reduced(train_loss.avg, current_device, 0, args.world_size)

        model.eval()
        with torch.no_grad():
            test_loss = test_epoch(model, test_loader, args, neighbors=neighbors, scaler=scaler)
        
        dist.barrier()
        val_loss = get_reduced(test_loss.avg, current_device, 0, args.world_size)

        if is_master:
            print(f'Epoch {epoch + 1} | Train Loss {train_loss} | Test Loss {val_loss}')

        if val_loss < best_loss and is_master:
            best_loss = val_loss
            best_epoch = epoch + 1
            
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.module.state_dict(), os.path.join(args.save_dir, 'spatialtip_hest_healthy_' + test_sample + '_finetune.pt'))
            print("Saved Best Model! Loss: {}".format(best_loss))

        dist.barrier()
        if epoch - (best_epoch - 1) == 10 and is_master:
            print("Early stopping triggered")
            stop_training.fill_(1)

        dist.broadcast(stop_training, src=0)   

        if stop_training.item() == 1:
            break
        
        dist.barrier()
    
    if is_master:
        print(f"Best Epoch: {best_epoch} | Best Loss: {best_loss}")
    cleanup()
    
if __name__ == '__main__':
     main()

