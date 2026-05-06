# SpatialTIP: a multimodal pretrained framework for imputing missing values in spatial transcritomics

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
  - [Pretraining](#1-pretraining)
  - [Fine-tuning](#2-fine-tuning)
  - [Gene Imputation](#3-gene-imputation)
  - [Spatial Clustering](#4-spatial-clustering)
- [Project Structure](#project-structure)
- [Citation](#citation)

## Overview

SpatialTIP leverages a ViT-L vision encoder (UNI pretrained) cross-attended to a BERT-style gene expression decoder. 

### Key Features

- 🔬 **Gene Imputation**: Predict missing/unexpressed gene values
- 🗺️ **Spatial Clustering**: Identify spatial domains from imputed expression
- 🖼️ **Vision-Gene Integration**: Combines histology images with transcriptomics

## Installation

### Requirements

- Python 3.9
- CUDA 11.8
- R with mclust package (for clustering)

### Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate spatialtip

# Set R environment variables (Linux)
export R_HOME=/root/miniconda3/envs/spatialtip/lib/R
export R_USER=/root/miniconda3/envs/spatialtip/lib/python3.9/site-packages/rpy2
```

### Download Pretrained Weights

Download the UNI ViT-L pretrained weights and place them at:
```
uni/checkpoints/pytorch_model.bin
```

## Data Preparation

### Data Format

SpatialTIP expects the following data structure:

```
hest_data/
├── st/                          # Raw spatial transcriptomics data (.h5ad)
│   ├── MISC1.h5ad
│   ├── MISC2.h5ad
│   └── ...
├── patches/                     # Image patches (.h5)
│   ├── MISC1.h5
│   ├── MISC2.h5
│   └── ...
├── vocabulary_healthy_3k.json  # Gene vocabulary
└── tissue_dict.json             # Tissue type mapping
```

### Data Processing Pipeline

**Step 1: Download HEST Dataset** (Optional)

```bash
python hest_download.py
```

**Step 2: Preprocess Data**

This step:
1. Selects highly variable genes (HVGs) across samples
2. Creates gene vocabulary (`vocabulary_*.json`)
3. Creates tissue type mapping (`tissue_dict*.json`)
4. Filters and normalizes expression data

## Model Architecture

### Components

| Component | File | Description |
|-----------|------|-------------|
| Core BERT | `spatialtip.py` | BERT-style encoder-decoder with cross-attention |
| Model Wrapper | `spatialtip_model.py` | Combines vision encoder with gene decoder |
| Initialization | `spatialtip_init.py` | Model initialization utilities |
| Dataset | `dataset.py` | Data loading utilities |


### Configuration Parameters

**Important:** Model configuration in `spatialtip_init.py` depends on the training dataset. Key parameters that vary:

| Parameter | Description | Varies By |
|-----------|-------------|-----------|
| `vocab_size` | Gene vocabulary size | Number of HVGs in dataset |
| `tissue_type` | Number of tissue types | Tissue categories in dataset |
| `max_seq_len` | Maximum sequence length | Training stage (see below) |

**Sequence Length Configuration:**
- **Pretraining**: `max_seq_len = 3001` (shorter for efficiency across multiple samples)
- **Fine-tuning**: `max_seq_len = 6001` (longer for sample-specific genes)

When using a different dataset, update these values in `spatialtip_init.py` accordingly.

## Usage

### 1. Pretraining

Train on multiple samples for general representation learning:

```bash
torchrun --nproc_per_node=4 train.py \
    --batch_size 32 \
    --epochs 80 \
    --learning_rate 1e-4 \
    --mask_prob 0.15 \
    --world_size 4 \
    --save_dir spatialtip_model
```

**Key Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 32 | Batch size per GPU |
| `--epochs` | 80 | Number of training epochs |
| `--learning_rate` | 1e-4 | Learning rate |
| `--mask_prob` | 0.15 | Probability of masking non-zero genes |
| `--world_size` | 4 | Number of GPUs |
| `--save_dir` | spatialtip_model | Model save directory |

### 2. Fine-tuning

There are two fine-tuning modes depending on your goal:

#### Standard Fine-tuning (`finetune.py`)

Fine-tune on **original unmasked data** for general adaptation to a specific sample:

```bash
torchrun --nproc_per_node=4 finetune.py \
    --finetune_sample MISC1 \
    --epochs 60 \
    --train_mask_prob 0.6 \
    --batch_size 32
```

#### Masked Fine-tuning (`finetune_masked.py`)

Fine-tune on **artificially masked data** for imputation evaluation. This mode:
1. Randomly masks a portion of expressed genes (specified by `--test_mask_prob`)
2. Saves the mask for later evaluation
3. Trains the model to predict masked values

```bash
torchrun --nproc_per_node=4 finetune_masked.py \
    --finetune_sample MISC1 \
    --test_mask_prob 0.2 \
    --epochs 30
```

| Mode | Script | Data | Use Case |
|------|--------|------|----------|
| Standard | `finetune.py` | Original data | General adaptation |
| Masked | `finetune_masked.py` | Artificially masked data | Imputation evaluation |

### 3. Gene Imputation

#### General Imputation (`impute.py`)

Predict expression for **unexpressed genes** (genes with zero counts). This masks all non-zero values and asks the model to predict what the zero values should be. Works with any model weights (pretrained or fine-tuned):

```bash
python impute.py \
    --test_sample MISC1 \
    --model_dir spatialtip_model \
    --model_name spatialtip_hest_healthy_3k.pt
```

**Output:** `{output_dir}/{sample}_imputed.h5ad`

| Argument | Default | Description |
|----------|---------|-------------|
| `--test_sample` | MISC1 | Sample to impute |
| `--model_dir` | spatialtip_model | Model directory |
| `--model_name` | spatialtip_hest_healthy_3k.pt | Model filename |
| `--data_dir` | hest_data | Data directory |
| `--data_subdir` | st_healthy_6k_v1 | Data subdirectory |
| `--patches_dir` | hest_data/patches | Patches directory |
| `--vocabulary_path` | hest_data/vocabulary_healthy_3k.json | Vocabulary file |
| `--output_dir` | Result/st_healthy_6k | Output directory |
| `--device` | cuda:0 | Device to use |
| `--batch_size` | 64 | Batch size |

#### Masked Imputation (`impute_masked.py`)

Evaluate imputation accuracy on **artificially masked data**. This uses data where some expressed genes were masked out, and compares predictions against ground truth. Requires fine-tuned model on the masked data:

```bash
python impute_masked.py \
    --test_sample MISC1 \
    --test_mask_prob 0.2 \
    --model_dir spatialtip_model
```

**Output:** `{output_dir}/{sample}_{mask_prob}.h5ad`

| Argument | Default | Description |
|----------|---------|-------------|
| `--test_sample` | MISC2 | Sample to evaluate |
| `--test_mask_prob` | 1.0 | Masking probability used in data |
| `--model_dir` | spatialtip_model | Model directory |
| `--model_name` | spatialtip_hest_healthy_{sample}_finetune.pt | Model filename (auto-set if not specified) |
| `--data_dir` | hest_data | Data directory |
| `--masked_data_subdir` | st_healthy_6k_masked_{seed} | Masked data subdirectory |
| `--original_data_subdir` | st_healthy_6k_v1 | Original data subdirectory |
| `--masks_dir` | hest_data/masks_healthy_6k_{seed} | Masks directory |
| `--output_dir` | Result/predicted_healthy_6k_masked_{seed} | Output directory |
| `--device` | cuda:0 | Device to use |
| `--batch_size` | 32 | Batch size |

| Script | Input Data | Use Case |
|--------|------------|----------|
| `impute.py` | Original data | General imputation of unexpressed genes |
| `impute_masked.py` | Pre-masked data | Evaluate imputation accuracy with ground truth |

### 4. Spatial Clustering

Perform spatial domain identification using the imputed gene expression. See `spatial_domain_clustering.ipynb` for a step-by-step tutorial.

**Key Steps:**
1. Load original and imputed data
2. Combine imputed values with original non-zero values
3. Apply spatial neighbor smoothing
4. Dimensionality reduction with PCA
5. Clustering with mclust (GMM-based)
6. Optional label refinement

**Input Files:**
- Original adata: `hest_data/st/{sample}.h5ad`
- Imputed adata: `Results/{sample}/spatialtip_imputed.h5ad`

**Output:**
- Clustering visualization: `figures/clustering/{sample}/clustering_result.png`
- Clustered adata: `Results/{sample}/spatialtip_clustered.h5ad`

### Evaluation Metrics

Calculate evaluation metrics for imputation results:

```bash
python evaluate.py \
    --sample MISC1 \
    --mask_prob 0.2
```

**Output:** `{output_dir}/{sample}_{mask_prob}_evaluation_metrics.csv`

**Metrics:**
- RMSE / MAE
- Spot-wise Pearson Correlation Coefficient (PCC)
- Spot-wise Cosine Similarity
- Gene-wise PCC
- Gene-wise Cosine Similarity

| Argument | Default | Description |
|----------|---------|-------------|
| `--sample` | MISC3 | Sample name |
| `--mask_prob` | 0.2 | Mask probability |
| `--seed` | 2024 | Random seed |
| `--data_dir` | hest_data | Data directory |
| `--masked_data_subdir` | st_healthy_6k_masked_{seed} | Masked data subdirectory |
| `--original_data_subdir` | st_healthy_6k_v1 | Original data subdirectory |
| `--imputed_data_dir` | Result/predicted_healthy_6k_masked_{seed} | Imputed data directory |
| `--masks_dir` | hest_data/masks_healthy_6k_{seed} | Masks directory |
| `--output_dir` | Metrices/healthy_6k_{seed} | Output directory |
| `--no_spatial_smoothing` | False | Disable spatial neighbor smoothing |
| `--rad_cutoff` | 150 | Radius cutoff for spatial network |

## Project Structure

```
SpatialTIP/
├── spatialtip.py              # Core BERT architecture
├── spatialtip_model.py        # Main model class
├── spatialtip_init.py         # Model initialization
├── dataset.py                 # Dataset classes
├── utils.py                   # Utility functions
│
├── train.py                   # Pretraining script
├── finetune.py                # Fine-tuning script
├── finetune_masked.py          # Fine-tuning on masked data
│
├── impute.py                  # General imputation
├── impute_masked.py           # Imputation on masked data
├── evaluate.py                # Evaluation metrics
├── spatial_domain_clustering.ipynb  # Spatial clustering tutorial
│
├── preprocess_hest_healthy_union.py  # Healthy data preprocessing
├── preprocess_hest_cancer_union.py   # Cancer data preprocessing
├── hest_download.py           # HEST dataset downloader
│
├── environment.yml            # Conda environment
└── README.md                  # This file
```

## Sample ID Mapping

DLPFC samples map to standard dataset IDs:

| Sample | Dataset ID | Sample | Dataset ID |
|--------|-----------|--------|-----------|
| MISC1 | 151676 | MISC7 | 151670 |
| MISC2 | 151675 | MISC8 | 151669 |
| MISC3 | 151674 | MISC9 | 151510 |
| MISC4 | 151673 | MISC10 | 151509 |
| MISC5 | 151672 | MISC11 | 151508 |
| MISC6 | 151671 | MISC12 | 151507 |

## Dependencies

| Package | Version |
|---------|---------|
| PyTorch | 2.4.0 |
| CUDA | 11.8 |
| flash-attn | 2.6.3 |
| timm | 1.0.9 |
| transformers | 4.44.2 |
| scanpy | 1.10.2 |
| anndata | 0.10.9 |
| rpy2 | 3.5.11 |

## License

This project is licensed under the Apache 2.0 License.
