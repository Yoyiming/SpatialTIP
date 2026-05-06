import torch
import torch.amp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scBERT.utils import CosineAnnealingWarmupRestarts
import math
from functools import reduce
import argparse
import os
import torch.distributed as dist
from dataset import SpatialTIPData
from utils import cleanup, AvgMeter, get_lr, get_reduced, mlm_mask, tokenize_batch
import tqdm
from spatialtip_model import SpatialTIPModel


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=80, help='number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--world_size', type=int, default=4, help='number of distributed processes (GPUs)')
parser.add_argument('--num_workers', type=int, default=32, help='number of workers for data loader')
parser.add_argument('--save_dir', type=str, default='spatialtip_model', help='model saving directory')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument("--mask_prob", type=float, default=0.15, help='Probability of masking.')
parser.add_argument('--replace_prob', type=float, default=0.9, help='replace prob')
args = parser.parse_args()


# get the random prob matrix and True means smaller than prob threshold
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


def data_mask(data, mask_prob=args.mask_prob, replace_prob=0.9, mask_token_id=-1, mask_ignore_token_ids=0):
    if mask_ignore_token_ids is None:
        mask_ignore_token_ids = []
    elif isinstance(mask_ignore_token_ids, (int, float)):
        mask_ignore_token_ids = [mask_ignore_token_ids]

    # do not mask tokens in the tokens designated to be excluded
    no_mask = mask_with_tokens(data, mask_ignore_token_ids)  # ignore_token as True, will not be masked later
    mask = get_mask_subset_with_prob(~no_mask, mask_prob)  # get the True/False mask matrix

    # mask input with mask token id
    masked_input = data.clone().detach()
    # replace_prob = prob_mask_like(data, replace_prob) # get the mask matrix of token being masked
    masked_input = masked_input.masked_fill(mask, mask_token_id)  # replace masked positions with mask_token_id

    # labels are the same as the original input
    labels = data

    return masked_input, labels


def data_loader_fn(args):
    # train_samples = ['INT7', 'INT9', 'INT10', 'ZEN38', 'ZEN39', 'ZEN47', 'ZEN49', 'TENX31', 'TENX53', 'TENX14'] # 12 = -INT11-TENX13  # 10 = -TENX13-INT11-INT7-ZEN36 # 10* = -TENX13-INT11-INT8-ZEN36
    # train_samples = ['MISC11', 'MISC9', 'MISC8', 'MISC6', 'MISC5', 'MISC2', 'MEND35', 'NCBI828', 'NCBI829', 'NCBI830', 'NCBI599'] # 可以尝试去掉MISC12,10,7, 4, 3, 1 = 6  MISC4, 3, 1 = 9

    train_samples = ['MISC11', 'MISC7', 'MISC4', 
                     'MEND91', 'MEND92', 'MEND93', 'MEND95', 'MEND96',
                     'NCBI828', 'NCBI829', 'NCBI830', 
                     'NCBI709', 'NCBI711', 'NCBI712', 'NCBI713', 'NCBI714',
                     'MEND85', 'MEND86', 'MEND87', 'MEND89', 'MEND90'] 
    
    # train_samples = ['NCBI681', 'NCBI682', 'NCBI683', 'NCBI684',
    #                  'INT7', 'INT8', 'INT9', 'INT10',
    #                  'ZEN36', 'ZEN38', 'ZEN39', 'ZEN47', 'ZEN49',
    #                  'TENX53', 'TENX14', 
    #                 #  'MEND61', 'MEND62', 'MEND153', 'MEND154',
    #                  'TENX62', 'TENX72',
    #                  'NCBI643', 'NCBI642',
    #                  'NCBI569', 'NCBI570', 'NCBI571', 'NCBI572']

    dataset = []
    for sample in train_samples:
        sub_dataset = SpatialTIPData(adata_path=f'hest_data/st_healthy_3k/{sample}.h5ad',
                                patches_path=f'hest_data/patches/{sample}.h5', 
                                vocabulary_path='hest_data/vocabulary_healthy_3k.json')
        dataset.append(sub_dataset)
    
    dataset = torch.utils.data.ConcatDataset(dataset)
        
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    print(f"Train size: {train_size}, Test size: {test_size}")
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(args.seed))

    # Set up distributed data parallel
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=test_sampler, pin_memory=True, drop_last=True)

    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, args, lr_scheduler=None, scaler=None):
    loss_meter = AvgMeter()
    tqdm_object = tqdm.tqdm(train_loader, total=len(train_loader), disable=dist.get_rank() != 0)

    for batch in tqdm_object:
        batch = {k: v.cuda() for k, v in batch.items() if k == "gene_exp" or k == 'image' or k == 'barcode' or k == 'gene_id' or k == 'tissue'}
        batch['gene_id'] = tokenize_batch(batch['gene_id'], append_cls=True, cls_id=0)
        batch['gene_exp'], batch['labels'] = mlm_mask(batch['gene_exp'], mask_token_id=-1, mask_prob_zero=0.05, mask_prob_nonzero=args.mask_prob)

        with torch.amp.autocast('cuda'):
            loss = model(batch, method='contrastive')

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


def test_epoch(model, test_loader, args, scaler=None):
    loss_meter = AvgMeter()
    tqdm_object = tqdm.tqdm(test_loader, total=len(test_loader), disable=dist.get_rank() != 0)
    for batch in tqdm_object:
        batch = {k: v.cuda() for k, v in batch.items() if k == "gene_exp" or k == 'image' or k == 'barcode' or k == 'gene_id' or k == 'tissue'}
        batch['gene_id'] = tokenize_batch(batch['gene_id'], append_cls=True, cls_id=0)
        batch['gene_exp'], batch['labels'] = mlm_mask(batch['gene_exp'], mask_token_id=-1, mask_prob_zero=0.05, mask_prob_nonzero=args.mask_prob)

        with torch.amp.autocast('cuda'):
            loss = model(batch, method='contrastive')

        count = batch['gene_exp'].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(val_loss=loss_meter.avg)

    return loss_meter


def main():
    print('Starting!')

    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    is_master = rank == 0
    local_rank = int(os.environ['LOCAL_RANK'])
    current_device = local_rank
    torch.cuda.set_device(current_device)
    
    train_loader, test_loader = data_loader_fn(args)
    print('Built data loader!')

    model = SpatialTIPModel(vision_model='vit_large_patch16_224').cuda(current_device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[current_device], find_unused_parameters=True)
    print('Built model!')
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=40,
        cycle_mult=1,
        max_lr=args.learning_rate,
        min_lr=5e-5,
        warmup_steps=5,
        gamma=0.98
    )

    best_loss = float('inf')
    best_epoch = 0
    stop_training = torch.tensor(0).cuda()

    dist.barrier()
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
            torch.save(model.module.state_dict(), os.path.join(args.save_dir, 'spatialtip_hest_healthy_3k_corrected.pt'))
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
