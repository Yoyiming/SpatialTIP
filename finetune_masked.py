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
import scipy.sparse as sp
import random
from functools import reduce
from torch.nn.parallel import DistributedDataParallel as DDP
import scanpy as sc

parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loader')
parser.add_argument('--save_dir', type=str, default='spatialtip_model', help='model saving directory')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--train_mask_prob', type=float, default=0.6, help='train mask prob')
parser.add_argument('--test_mask_prob', type=float, default=0.2, help='test mask prob')
parser.add_argument('--replace_prob', type=float, default=1.0, help='replace prob')
parser.add_argument('--world_size', type=int, default=4, help='number of processes for distributed training')
parser.add_argument('--finetune_sample', type=str, default='MISC1', help='finetune sample')


def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob


# get the mask matrix which cannot be masked
def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask


def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)      # num of mask of a single sequence in average
    num_tokens = mask.sum(dim=-1, keepdim=True)     # num of pure tokens of each sequence except special tokens
    mask_excess = torch.cat((torch.zeros(0), torch.arange(mask.size(-1)).repeat(mask.size(0)))).reshape(mask.size(0),mask.size(-1)).to(device)
    mask_excess = (mask_excess >= (num_tokens * prob).ceil())        # only 15% of pure tokens can be masked
    mask_excess = mask_excess[:, :max_masked]       # get difference between 15% of pure tokens and 15% of all tokens
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)     # rand (0-1) as prob, special token use -1e9
    _, sampled_indices = rand.topk(max_masked, dim=-1)      # get index of topk prob to mask
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)        # delete difference of mask not pure
    new_mask = torch.zeros((batch, seq_len + 1), device=device)     # get (batch, seq_len) shape zero matrix
    new_mask.scatter_(-1, sampled_indices, 1)       # set masks in zero matrix as 1
    return new_mask[:, 1:].bool()       # the final mask, True is mask


def data_mask(data, mask_prob=0.15, relpace_prob=0.9, mask_token_id=-1, mask_ignore_token_ids=0):
    if mask_ignore_token_ids is None:
        mask_ignore_token_ids = []
    elif isinstance(mask_ignore_token_ids, (int, float)):
        mask_ignore_token_ids = [mask_ignore_token_ids]

    # do not mask tokens in the tokens designated to be excluded
    no_mask = mask_with_tokens(data, mask_ignore_token_ids)  # ignore_token as True, will not be masked later
    mask = get_mask_subset_with_prob(~no_mask, mask_prob)  # get the True/False mask matrix
    # mask input with mask token id
    masked_input = data.clone().detach()
    masked_input = masked_input.masked_fill(mask, mask_token_id)  # replace masked positions with mask_token_id

    # labels are the same as the original input
    labels = data

    return masked_input, labels


def data_loader_fn(args, test_samples, test_mask_prob):

    hest_dataset = SpatialTIPData(adata_path=f'hest_data/st_healthy_6k_masked_{str(args.seed)}/{test_samples}_{str(test_mask_prob)}.h5ad',
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


def generate_mask(mask_prob, adata_path, adata_save_path, mask_save_path):
    adata = sc.read_h5ad(adata_path)
    matrix = adata.X.toarray().copy()
    mask = np.ones(matrix.shape)
    for i in range(matrix.shape[0]):
        non_zero_indices = np.where(matrix[i] != 0)[0]

        mask_count = int(mask_prob * len(non_zero_indices))

        mask_indices = np.random.choice(non_zero_indices, mask_count, replace=False)

        mask[i][mask_indices] = 0
    
    masked_matrix = matrix * mask
    
    adata_masked = adata.copy()
    adata_masked.X = sp.csr_matrix(masked_matrix)
    # 只在主进程保存
    if dist.get_rank() == 0:
        adata_masked.write(adata_save_path)
        sp.save_npz(mask_save_path, sp.csr_matrix(mask))
    

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
    test_mask_prob = args.test_mask_prob

    print('Finetune sample:', test_sample)
    print('Test mask prob:', test_mask_prob)

    adata_path = f'hest_data/st_healthy_6k_v1/{test_sample}.h5ad'
    adata_save_path = f'hest_data/st_healthy_6k_masked_{str(args.seed)}/{test_sample}_{str(test_mask_prob)}.h5ad'
    mask_save_path = f'hest_data/masks_healthy_6k_{str(args.seed)}/{test_sample}_{str(test_mask_prob)}_mask.npz'

    os.makedirs(f'hest_data/st_healthy_6k_masked_{str(args.seed)}/', exist_ok=True)

    generate_mask(test_mask_prob, adata_path, adata_save_path, mask_save_path)

    train_loader, test_loader = data_loader_fn(args, test_sample, test_mask_prob)

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
        cycle_mult=2,
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
        train_loss = train_epoch(model, train_loader, optimizer, args, lr_scheduler=scheduler, scaler=scaler)

        dist.barrier()

        train_loss = get_reduced(train_loss.avg, current_device, 0, args.world_size)

        model.eval()
        with torch.no_grad():
            test_loss = test_epoch(model, test_loader, args, scaler=scaler)
        
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
