import scipy as sp
import torch
from torch.utils.data import DataLoader
from spatialtip_model import SpatialTIPModel
import argparse
import os
from dataset import SpatialTIPData
from utils import tokenize_batch
import tqdm
import numpy as np
import pandas as pd
import random
import scanpy as sc


parser = argparse.ArgumentParser(description='Gene imputation on masked data')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loader')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--test_sample', type=str, default='MISC2', help='test sample name')
parser.add_argument('--test_mask_prob', type=float, default=1.0, help='mask probability used in data')
parser.add_argument('--device', type=str, default='cuda:0', help='device to use')

# Data paths
parser.add_argument('--data_dir', type=str, default='hest_data', help='data directory')
parser.add_argument('--masked_data_subdir', type=str, default=None, help='masked data subdirectory (default: st_healthy_6k_masked_{seed})')
parser.add_argument('--original_data_subdir', type=str, default='st_healthy_6k_v1', help='original data subdirectory for ground truth')
parser.add_argument('--patches_dir', type=str, default='hest_data/patches', help='patches directory')
parser.add_argument('--vocabulary_path', type=str, default='hest_data/vocabulary_healthy_3k.json', help='vocabulary file path')
parser.add_argument('--masks_dir', type=str, default=None, help='masks directory (default: hest_data/masks_healthy_6k_{seed})')

# Model paths
parser.add_argument('--model_dir', type=str, default='spatialtip_model', help='model directory')
parser.add_argument('--model_name', type=str, default=None, help='model filename (default: spatialtip_hest_healthy_{sample}_finetune.pt)')

# Output paths
parser.add_argument('--output_dir', type=str, default=None, help='output directory (default: Result/predicted_healthy_6k_masked_{seed})')


def data_loader_fn(args):
    masked_subdir = args.masked_data_subdir or f'st_healthy_6k_masked_{args.seed}'
    adata_path = os.path.join(args.data_dir, masked_subdir, f'{args.test_sample}_{args.test_mask_prob}.h5ad')
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


def evaluate_metrics(true_exp_matrix, imputed_exp_matrix, mask, save_path):
    mask = mask == 0

    RMSE = np.sqrt(np.mean((true_exp_matrix[mask] - imputed_exp_matrix[mask]) ** 2))
    MAE = np.mean(np.abs(true_exp_matrix[mask] - imputed_exp_matrix[mask]))

    # Spot-wise Pearson Correlation Coefficient
    spot_corr = np.array([
        np.corrcoef(imputed_exp_matrix[i, mask[i]], true_exp_matrix[i, mask[i]])[0, 1]
        for i in range(imputed_exp_matrix.shape[0])
    ])
    spot_corr = spot_corr[~np.isnan(spot_corr)]

    # Spot-wise Cosine Similarity
    spot_cosine = np.array([
        np.dot(imputed_exp_matrix[i, mask[i]], true_exp_matrix[i, mask[i]]) /
        (np.linalg.norm(imputed_exp_matrix[i, mask[i]]) * np.linalg.norm(true_exp_matrix[i, mask[i]]))
        for i in range(imputed_exp_matrix.shape[0])
    ])
    spot_cosine = spot_cosine[~np.isnan(spot_cosine)]

    # Gene-wise Pearson Correlation Coefficient
    gene_corr = np.array([
        np.corrcoef(imputed_exp_matrix[:, i][mask[:, i]], true_exp_matrix[:, i][mask[:, i]])[0, 1]
        for i in range(imputed_exp_matrix.shape[1])
    ])
    gene_corr = gene_corr[~np.isnan(gene_corr)]

    # Gene-wise Cosine Similarity
    gene_cosine = np.array([
        np.dot(imputed_exp_matrix[:, i][mask[:, i]], true_exp_matrix[:, i][mask[:, i]]) /
        (np.linalg.norm(imputed_exp_matrix[:, i][mask[:, i]]) * np.linalg.norm(true_exp_matrix[:, i][mask[:, i]]))
        for i in range(imputed_exp_matrix.shape[1])
    ])
    gene_cosine = gene_cosine[~np.isnan(gene_cosine)]

    evaluation_results = pd.DataFrame({
        'RMSE': [RMSE],
        'MAE': [MAE],
        'Spot-wise PCC': [np.mean(spot_corr)],
        'Spot-wise Cosine Similarity': [np.mean(spot_cosine)],
        'Gene-wise PCC': [np.mean(gene_corr)],
        'Gene-wise Cosine Similarity': [np.mean(gene_cosine)]
    })
    evaluation_results.to_csv(save_path, index=False)


def main():
    args = parser.parse_args()

    print('Start imputing on masked data!')
    print(f'Test sample: {args.test_sample}')
    print(f'Test mask prob: {args.test_mask_prob}')

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set default paths
    output_dir = args.output_dir or f'Result/predicted_healthy_6k_masked_{args.seed}'
    masks_dir = args.masks_dir or f'hest_data/masks_healthy_6k_{args.seed}'
    model_name = args.model_name or f'spatialtip_hest_healthy_{args.test_sample}_finetune.pt'

    # Load mask
    mask_path = os.path.join(masks_dir, f'{args.test_sample}_{args.test_mask_prob}_mask.npz')
    mask = sp.sparse.load_npz(mask_path).toarray()

    # Load data
    test_loader = data_loader_fn(args)

    # Load model
    model = SpatialTIPModel(vision_model='vit_large_patch16_224').to(device)

    model_path = os.path.join(args.model_dir, model_name)
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
        batch['gene_exp'][batch['gene_exp'] == 0] = -1  # mask zero values

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                decoded_gene_expression = model(batch, mode='test')

        imputed_gene_expression.append(decoded_gene_expression.cpu().detach().numpy())

    imputed_gene_expression = np.concatenate(imputed_gene_expression, axis=0)

    # Load original data for ground truth
    original_adata_path = os.path.join(args.data_dir, args.original_data_subdir, f'{args.test_sample}.h5ad')
    true_gene_expression = sc.read_h5ad(original_adata_path).X.toarray()

    # Save results
    adata = sc.read_h5ad(original_adata_path)
    adata.X = imputed_gene_expression

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{args.test_sample}_{args.test_mask_prob}.h5ad')
    adata.write(output_path)
    print(f'Results saved to: {output_path}')


if __name__ == '__main__':
    main()
