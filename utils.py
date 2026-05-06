import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from scFoundation.model.load import *
import sklearn.neighbors
import ot


def tokenize_batch(
        gene_ids: torch.Tensor,
        append_cls: bool = True,
        cls_id: int = 0        
):
    if append_cls:
        cls_id = torch.tensor([cls_id], device=gene_ids.device).repeat(gene_ids.shape[0], 1)
        gene_ids = torch.cat([cls_id, gene_ids], dim=1)

    return gene_ids


def mlm_mask(data, mask_token_id=-1, mask_prob_zero=0.05, mask_prob_nonzero=0.15):
    """
    对序列进行MLM任务的mask
    Args:
        data: 输入张量 shape (batch_size, seq_len)
        mask_token_id: mask标记的token id 默认为-1
        mask_prob_zero: 对值为0的token进行mask的概率 默认0.05
        mask_prob_nonzero: 对值大于0的token进行mask的概率 默认0.15
    Returns:
        masked_input: 被mask后的输入
        labels: 原始输入作为标签
    """
    batch_size, seq_len = data.shape
    device = data.device
    
    rand_zero = torch.rand(batch_size, seq_len, device=device)
    rand_nonzero = torch.rand(batch_size, seq_len, device=device)
    
    zero_mask = (data == 0) & (rand_zero < mask_prob_zero)
    nonzero_mask = (data > 0) & (rand_nonzero < mask_prob_nonzero)
    
    final_mask = zero_mask | nonzero_mask
    
    masked_input = data.clone().detach()
    masked_input[final_mask] = mask_token_id
    
    return masked_input, data


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    # tensor_all = GatherLayer.apply(tensors)
    # tensors = tensors.contiguous()
    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def cleanup():
    dist.destroy_process_group()


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    

def get_reduced(tensor, current_device, dest_device, world_size):
    """
    将不同GPU上的变量或tensor集中在主GPU上, 并得到均值
    """
    tensor = tensor.clone().detach() if torch.is_tensor(tensor) else torch.tensor(tensor)
    tensor = tensor.to(current_device)
    torch.distributed.reduce(tensor, dst=dest_device)
    tensor_mean = tensor.item() / world_size
    return tensor_mean


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='stGPT', random_seed=2024, save_name='mclust'):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs[save_name] = mclust_res
    adata.obs[save_name] = adata.obs[save_name].astype('int')
    adata.obs[save_name] = adata.obs[save_name].astype('category')
    return adata


def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    # old_type = adata.obsm[key]
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    #adata.obs['label_refined'] = np.array(new_type)
    
    return new_type


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='KNN', position_path=None):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks of the spots.
    """

    assert(model in ['Radius', 'KNN'])
    print('------Calculating spatial graph...')

    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    # coor = pd.read_csv(position_path, index_col=0)
    # coor.drop(columns=['id', 'pxl_row_in_fullres', 'pxl_col_in_fullres'], inplace=True)
    # coor.rename(columns={'x': 'imagerow', 'y': 'imagecol'}, inplace=True)
    
    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Spot1', 'Spot2', 'Distance']

    Spatial_Net = KNN_df.copy()
    # Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    # id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    # Spatial_Net['Spot1'] = Spatial_Net['Spot1'].map(id_cell_trans)
    # Spatial_Net['Spot2'] = Spatial_Net['Spot2'].map(id_cell_trans)

    return Spatial_Net
