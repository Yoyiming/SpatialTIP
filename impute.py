import torch
from torch.utils.data import DataLoader
from spatialtip_model import SpatialTIPModel
import argparse
import os
from dataset import SpatialTIPData
from utils import tokenize_batch
import tqdm
import numpy as np
import random
import scanpy as sc


parser = argparse.ArgumentParser(description='Gene imputation for unexpressed genes')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loader')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--test_sample', type=str, default='MISC1', help='test sample name')
parser.add_argument('--device', type=str, default='cuda:0', help='device to use')

# Data paths
parser.add_argument('--data_dir', type=str, default='hest_data', help='data directory')
parser.add_argument('--data_subdir', type=str, default='st_healthy_6k_v1', help='data subdirectory')
parser.add_argument('--patches_dir', type=str, default='hest_data/patches', help='patches directory')
parser.add_argument('--vocabulary_path', type=str, default='hest_data/vocabulary_healthy_3k.json', help='vocabulary file path')

# Model paths
parser.add_argument('--model_dir', type=str, default='spatialtip_model', help='model directory')
parser.add_argument('--model_name', type=str, default='spatialtip_hest_healthy_3k.pt', help='model filename')

# Output paths
parser.add_argument('--output_dir', type=str, default='Result/st_healthy_6k', help='output directory')


def data_loader_fn(args):
    adata_path = os.path.join(args.data_dir, args.data_subdir, f'{args.test_sample}.h5ad')
    patches_path = os.path.join(args.patches_dir, f'{args.test_sample}.h5')

    dataset = SpatialTIPData(
        adata_path=adata_path,
        patches_path=patches_path,
        vocabulary_path=args.vocabulary_path
    )

    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    return test_loader


def main():
    args = parser.parse_args()

    print('Start imputing!')
    print(f'Test sample: {args.test_sample}')

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load data
    test_loader = data_loader_fn(args)

    # Load model
    model = SpatialTIPModel(vision_model='vit_large_patch16_224').to(device)

    model_path = os.path.join(args.model_dir, args.model_name)
    print(f'Model path: {model_path}')

    state_dict = torch.load(model_path, map_location='cpu')
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    print('Model loaded!')

    # Imputation
    imputed_gene_expression = []
    model.eval()

    tqdm_object = tqdm.tqdm(test_loader, total=len(test_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items()
                 if k in ["gene_exp", "image", "barcode", "gene_id", "tissue"]}
        batch['gene_id'] = tokenize_batch(batch['gene_id'], append_cls=True, cls_id=0)
        batch['labels'] = batch['gene_exp'].clone().detach()
        batch['gene_exp'][batch['gene_exp'] != 0] = -1  # mask non-zero values, predict zero values

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                decoded_gene_expression = model(batch, mode='test')

        imputed_gene_expression.append(decoded_gene_expression.cpu().detach().numpy())

    imputed_gene_expression = np.concatenate(imputed_gene_expression, axis=0)

    # Save results
    adata_path = os.path.join(args.data_dir, args.data_subdir, f'{args.test_sample}.h5ad')
    adata = sc.read_h5ad(adata_path)
    adata.X = imputed_gene_expression

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'{args.test_sample}_imputed.h5ad')
    adata.write(output_path)
    print(f'Results saved to: {output_path}')


if __name__ == '__main__':
    main()
