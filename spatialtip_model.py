import random
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from utils import concat_all_gather, all_gather_with_grad
from spatialtip_init import (
    SpatialTIPBase,
    disabled_train,
)
import numpy as np
from utils import is_dist_avail_and_initialized


class SpatialTIPModel(SpatialTIPBase):
    def __init__(self, vision_model, img_size=224, patch_size=16, hidden_size=512):
        super().__init__()

        self.vision_encoder = self.init_vision_encoder(vision_model, img_size, patch_size)
        for param in self.vision_encoder.patch_embed.parameters():
            param.requires_grad = False
        
        for i in range(18):
            for param in self.vision_encoder.blocks[i].parameters():
                param.requires_grad = False

        for i in range(18, 24):
            for param in self.vision_encoder.blocks[i].parameters():
                param.requires_grad = True

        for param in self.vision_encoder.norm.parameters():
            param.requires_grad = True
        
        for param in self.vision_encoder.fc_norm.parameters():
            param.requires_grad = False
        for param in self.vision_encoder.head_drop.parameters():
            param.requires_grad = False
        for param in self.vision_encoder.head.parameters():
            param.requires_grad = False    
       
        self.spatialtip = self.init_spatialtip()

        self.vision_proj = nn.Linear(self.vision_encoder.num_features, hidden_size)
        self.vision_norm = nn.LayerNorm(hidden_size)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, sample, neighbors=None, mode='train', method='reconstruction', pooling='cls', output_cls=False, output_attentions=False):
        img = sample['image']  
        gene_exp = sample['gene_exp']  # [batch_size, seq_length]
        labels = sample['labels']  # [batch_size, seq_length]
        barcode = sample['barcode']  # [batch_size, 1]
        gene_id = sample['gene_id']  # [batch_size, seq_length + 1]
        batch_id = sample['tissue']  # [batch_size, 1]

        image_embeds = self.vision_encoder.forward_features(img)
        image_embeds = self.vision_proj(image_embeds)
        image_embeds = self.vision_norm(image_embeds)  # [batch_size, seq_length, hidden_size]

        batch_size, seq_length = gene_id.shape
        _, image_seq_length, _ = image_embeds.shape

        attention_mask = (gene_exp == -1).long()  # mask positions are 1, others are 0
        attention_mask = torch.cat([torch.zeros(batch_size, 1, device=gene_exp.device, dtype=torch.long), attention_mask], dim=1)  # [batch_size, seq_length]
        attention_mask = torch.zeros_like(attention_mask, dtype=torch.long, device=attention_mask.device) # comment this line to use masked attention

        ########################### ISC Loss ################################
        if mode == 'train' and method == 'contrastive':
            image_cls_embeds = F.normalize(image_embeds[:, 0, :], dim=-1) # [batch_size, hidden_size]
            image_embeds_all = concat_all_gather(image_cls_embeds) # [world_size * batch_size, hidden_size]
            
            gene_output = self.spatialtip.bert(
                input_ids = gene_id,
                gene_exp=gene_exp,
                attention_mask=attention_mask,
                return_dict=True,
            )

            gene_embeddings = gene_output.last_hidden_state  # [batch_size, seq_length, hidden_size]

            if pooling == 'max':
                spot_embeddings = F.normalize(gene_embeddings.max(dim=1)[0], dim=-1)
            elif pooling == 'mean':
                spot_embeddings = F.normalize(gene_embeddings.mean(dim=1), dim=-1)
            elif pooling == 'cls':
                spot_embeddings = F.normalize(gene_embeddings[:, 0, :], dim=-1)  # [batch_size, hidden_size]

            spot_embeddings_all = concat_all_gather(spot_embeddings) # [world_size * batch_size, hidden_size]

            barcode_all = concat_all_gather(barcode).squeeze(-1)  # [world_size * batch_size, 1]
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * spot_embeddings_all @ image_embeds_all.t()  # [world_size * batch_size, world_size * batch_size]
            total_samples = spot_embeddings_all.size(0)

            rank = dist.get_rank()

            if neighbors is None:
                logits_per_image = logit_scale * image_cls_embeds @ spot_embeddings_all.t() # [batch_size, world_size * batch_size]
                logits_per_spot = logit_scale * spot_embeddings @ image_embeds_all.t() # [batch_size, world_size * batch_size]

                targets = torch.arange(rank * batch_size, (rank + 1) * batch_size, device=gene_exp.device, dtype=torch.long)

                loss_isc = (F.cross_entropy(logits_per_spot, targets) + F.cross_entropy(logits_per_image, targets)) / 2

            else: 
                positive_pairs = []
                negative_pairs = []
                for i in range(batch_size):
                    global_i = rank * batch_size + i
                    bc = barcode_all[global_i].item()
                    positive_indices = [j for j in range(total_samples) if float(barcode_all[j].item()) in neighbors[bc]]
                    negative_indices = [j for j in range(total_samples) if float(barcode_all[j].item()) not in neighbors[bc] and j != global_i]

                    for pos_idx in positive_indices:
                        positive_pairs.append((global_i, pos_idx))

                    for neg_idx in negative_indices:
                        negative_pairs.append((global_i, neg_idx))
                
                positive_pairs = torch.tensor(positive_pairs, device=gene_exp.device)
                negative_pairs = torch.tensor(negative_pairs, device=gene_exp.device)

                positive_logits = logits[positive_pairs[:, 0], positive_pairs[:, 1]]
                negative_logits = logits[negative_pairs[:, 0], negative_pairs[:, 1]]

                loss_isc = F.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits)) + F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits))
                    
        ########################### Reconstruction Loss ################################
        if method == 'reconstruction':
            loss_isc = torch.tensor(0.0, device=gene_exp.device)
        
        cross_atts_mask = torch.ones(batch_size, seq_length, image_seq_length, device=gene_exp.device, dtype=torch.float32)  # [batch_size, gene_length, image_seq_length]

        lm_output = self.spatialtip(
            input_ids=gene_id,
            gene_exp=gene_exp,
            attention_mask=attention_mask,
            labels=labels,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=cross_atts_mask,
            return_dict=True,
            is_decoder=True,
            batch_id=batch_id,
            output_cls=output_cls,
            output_attentions=output_attentions,
        )
        
        if mode == 'train':
            loss_lm = lm_output.loss
            loss = loss_lm + 0.5 * loss_isc
            return loss
        elif mode == 'test' and output_cls:
            return lm_output.logits, lm_output.hidden_states
        elif mode == 'test' and output_attentions:
            return lm_output.logits, lm_output.attentions
        else:
            return lm_output.logits
        
        