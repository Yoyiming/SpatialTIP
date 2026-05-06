import scanpy as sc
import os
import argparse
import numpy as np
import pandas as pd
import scipy as sp
from utils import Cal_Spatial_Net


parser = argparse.ArgumentParser(description='Evaluate imputation metrics')
parser.add_argument('--sample', type=str, default='MISC3', help='sample name')
parser.add_argument('--mask_prob', type=float, default=0.2, help='mask probability')
parser.add_argument('--seed', type=int, default=2024, help='random seed')

# Data paths
parser.add_argument('--data_dir', type=str, default='hest_data', help='data directory')
parser.add_argument('--masked_data_subdir', type=str, default=None, help='masked data subdirectory (default: st_healthy_6k_masked_{seed})')
parser.add_argument('--original_data_subdir', type=str, default='st_healthy_6k_v1', help='original data subdirectory')
parser.add_argument('--imputed_data_dir', type=str, default=None, help='imputed data directory (default: Result/predicted_healthy_6k_masked_{seed})')
parser.add_argument('--masks_dir', type=str, default=None, help='masks directory (default: hest_data/masks_healthy_6k_{seed})')

# Output paths
parser.add_argument('--output_dir', type=str, default=None, help='output directory (default: Metrices/healthy_6k_{seed})')

# Spatial smoothing
parser.add_argument('--no_spatial_smoothing', action='store_true', help='disable spatial neighbor smoothing')
parser.add_argument('--rad_cutoff', type=float, default=150, help='radius cutoff for spatial network')


def evaluate_metrics(true_exp_matrix, imputed_exp_matrix, mask, save_path):
    """Calculate evaluation metrics for imputation."""
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

    return evaluation_results


def apply_spatial_smoothing(imputed_matrix, adata, rad_cutoff=150):
    """Apply spatial neighbor smoothing to imputed matrix."""
    spatial_net = Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff, model='Radius')
    neighbors = {}
    for _, row in spatial_net.iterrows():
        spot1, spot2 = row['Spot1'], row['Spot2']
        neighbors.setdefault(spot1, set()).add(spot2)
        neighbors.setdefault(spot2, set()).add(spot1)

    smoothed_matrix = imputed_matrix.copy()
    for spot, neighbor_spots in neighbors.items():
        neighbor_spots = [int(spot) for spot in neighbor_spots]
        smoothed_matrix[int(spot)] = smoothed_matrix[neighbor_spots].mean(axis=0)

    return smoothed_matrix


def main():
    args = parser.parse_args()

    print('Start evaluation!')
    print(f'Sample: {args.sample}')
    print(f'Mask probability: {args.mask_prob}')

    # Set default paths
    masked_subdir = args.masked_data_subdir or f'st_healthy_6k_masked_{args.seed}'
    imputed_dir = args.imputed_data_dir or f'Result/predicted_healthy_6k_masked_{args.seed}'
    masks_dir = args.masks_dir or f'hest_data/masks_healthy_6k_{args.seed}'
    output_dir = args.output_dir or f'Metrices/healthy_6k_{args.seed}'

    # Load data
    masked_adata_path = os.path.join(args.data_dir, masked_subdir, f'{args.sample}_{args.mask_prob}.h5ad')
    original_adata_path = os.path.join(args.data_dir, args.original_data_subdir, f'{args.sample}.h5ad')
    imputed_adata_path = os.path.join(imputed_dir, f'{args.sample}_{args.mask_prob}.h5ad')
    mask_path = os.path.join(masks_dir, f'{args.sample}_{args.mask_prob}_mask.npz')

    print(f'Masked data: {masked_adata_path}')
    print(f'Original data: {original_adata_path}')
    print(f'Imputed data: {imputed_adata_path}')
    print(f'Mask file: {mask_path}')

    adata_masked = sc.read_h5ad(masked_adata_path)
    adata_raw = sc.read_h5ad(original_adata_path)
    adata_imputed = sc.read_h5ad(imputed_adata_path)

    # Load mask
    mask = sp.sparse.load_npz(mask_path).toarray()

    # Prepare matrices
    masked_matrix = adata_masked.X.toarray() if hasattr(adata_masked.X, 'toarray') else adata_masked.X
    raw_matrix = adata_raw.X.toarray() if hasattr(adata_raw.X, 'toarray') else adata_raw.X
    imputed_matrix = adata_imputed.X.copy() if hasattr(adata_imputed.X, 'copy') else np.array(adata_imputed.X)

    # Combine imputed values with original non-masked values
    zero_mask = masked_matrix == 0
    imputed_matrix = np.where(zero_mask, imputed_matrix, masked_matrix)

    # Apply spatial smoothing (enabled by default)
    if not args.no_spatial_smoothing:
        print('Applying spatial smoothing...')
        imputed_matrix = apply_spatial_smoothing(imputed_matrix, adata_masked, args.rad_cutoff)

    # Evaluate metrics
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{args.sample}_{args.mask_prob}_evaluation_metrics.csv')

    results = evaluate_metrics(raw_matrix, imputed_matrix, mask, output_path)
    print(f'\nEvaluation results saved to: {output_path}')
    print(results.to_string(index=False))


if __name__ == '__main__':
    main()
