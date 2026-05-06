import os
import scanpy as sc
import json
import torch.utils
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = pow(2, 40).__str__()
import h5py
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


class SpatialTIPData(torch.utils.data.Dataset):
    """Dataset class for SpatialTIP model.

    Args:
        adata_path: Path to the h5ad file containing gene expression data.
        patches_path: Path to the h5 file containing image patches.
        vocabulary_path: Path to the JSON file containing gene vocabulary mapping.
    """

    def __init__(self, adata_path, patches_path, vocabulary_path):
        self.adata = sc.read_h5ad(adata_path)
        self.patches_h5 = h5py.File(patches_path, 'r')
        self.vocab = json.load(open(vocabulary_path, 'r'))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.index = self.adata.obs.index
        self.exp_mtx = self.adata.X.toarray()
        self.gene_id = np.array([self.vocab[gene] for gene in self.adata.var_names])
        self.tissue_type = self.adata.obs['tissue'].values

        self.id_spot_tran = dict(zip(range(self.exp_mtx.shape[0]), np.array(self.index)))
        self.id_spot_tran = {v: k for k, v in self.id_spot_tran.items()}

    def __getitem__(self, idx):
        item = {}
        barcode = self.index[idx]
        index = self.id_spot_tran[barcode]

        # Read patches from h5 file by barcode
        patches = self.patches_h5['img'][self.patches_h5['barcode'] == barcode][:]
        image = Image.fromarray(patches)
        image = self.transform(image)

        item['barcode'] = torch.tensor(index).float()
        item['image'] = image.clone().detach().float()
        item['gene_exp'] = torch.tensor(self.exp_mtx[idx, :]).float()
        item['gene_id'] = torch.tensor(self.gene_id).long()
        item['tissue'] = torch.tensor(self.tissue_type[idx]).long()

        return item

    def __len__(self):
        return self.exp_mtx.shape[0]
