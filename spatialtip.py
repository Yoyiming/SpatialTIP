import math
import os
import warnings
from dataclasses import dataclass
from transformers.utils import logging
from typing import Optional, Tuple, Dict, Any
import torch
from torch import Tensor, device, nn
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.modeling_utils import apply_chunking_to_forward, PreTrainedModel
from transformers.models.bert.configuration_bert import BertConfig
logger = logging.get_logger(__name__)
from flash_attn import flash_attn_func


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x
    

class TissueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x
    

class AutoDiscretizationEmbedding2(nn.Module):
    """
    This class implements the auto-discretization embedding for gene expression data.
    It converts continuous gene expression values into discrete bins and embeds them into a high-dimensional space.
    when masked, it will use position embedding to represent the token embedding
    otherwise, it will use the position embedding + values embedding to represent the token embedding
    """
    def __init__(self, dim, bin_num=150, bin_alpha=1.0, mask_token_id=None):
        super().__init__()

        self.dim = dim
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha

        self.mlp = nn.Linear(1, self.bin_num)
        self.mlp2 = nn.Linear(self.bin_num, self.bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(self.dim)

        self.emb = nn.Embedding(self.bin_num, self.dim)
        self.emb_mask = nn.Embedding(1, self.dim)

        self.bin_num_idx = torch.tensor(range(self.bin_num))
        self.mask_token_id = mask_token_id

    def forward(self, x, output_weight=0):
        x_mask_idx = (x == self.mask_token_id).nonzero()

        x = self.mlp(x)  # [B,N,1] -> [B,N,H]
        x = self.LeakyReLU(x)  # [B,N,H]
        x_crosslayer = self.mlp2(x)  # [B,N,H]
        x = self.bin_alpha * x + x_crosslayer  # [B,N,H]
        weight = self.Softmax(x)  # [B, N, H]
        bin_num_idx = self.bin_num_idx.to(x.device)  # [H,]

        token_emb = self.emb(bin_num_idx)  # [H, D]
        x = torch.matmul(weight, token_emb)  # [B, N, D]

        tensor0 = torch.tensor(0, dtype=torch.long).to(x.device)
        mask_token_embed = self.emb_mask(tensor0).to(x.device).type(x.dtype)  # [1, D]

        x[x_mask_idx[:, 0], x_mask_idx[:, 1]] = mask_token_embed.repeat(x_mask_idx.size(0), 1)
        
        x = self.norm(x)

        if output_weight:
            return x, weight
        return x


class BertEmbeddings(nn.Module):
    """Construct the embeddings from gene expression embeddings and gene id embeddings."""

    def __init__(self, config):
        super().__init__()

        self.gene_encoder = GeneEncoder(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)

        self.value_encoder = AutoDiscretizationEmbedding2(config.hidden_size, mask_token_id=config.mask_token_id)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.config = config

    def forward(
            self,
            input_ids=None,
            gene_exp=None,
    ):  
        if input_ids is not None:
            gene_embeddings = self.gene_encoder(input_ids)
        else:
            raise ValueError("input_ids is None, please provide input_ids")

        if gene_exp is not None:
            value_embeddings = self.value_encoder(torch.unsqueeze(gene_exp, 2))
        else:
            raise ValueError("gene_exp is None, please provide gene_exp")

        cls_value = torch.zeros((value_embeddings.size(0), 1, value_embeddings.size(2)), device=value_embeddings.device)
        value_embeddings = torch.cat((cls_value, value_embeddings), 1)

        embeddings = gene_embeddings + value_embeddings
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config, is_cross_attention):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.is_cross_attention = is_cross_attention

        self.use_flash = True


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        if self.is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_probs = None
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        past_key_value = (key_layer, value_layer)

        if self.use_flash:
            query_layer = query_layer.permute(0, 2, 1, 3).contiguous()
            key_layer = key_layer.permute(0, 2, 1, 3).contiguous()
            value_layer = value_layer.permute(0, 2, 1, 3).contiguous()

            context_layer = flash_attn_func(query_layer, key_layer, value_layer)
        else:
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=value_layer.dtype)  # fp16 compatibility

            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_scores) if output_attentions else (context_layer,)

        outputs = outputs + (past_key_value,)
        return outputs

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.self = BertSelfAttention(config, is_cross_attention)
        self.output = BertSelfOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)  # residual connection

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.layer_num = layer_num
        if (
                self.config.add_cross_attention
                and layer_num % self.config.cross_attention_freq == 0
        ):
            self.crossattention = BertAttention(
                config, is_cross_attention=self.config.add_cross_attention
            )
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.intermediate_query = BertIntermediate(config)
        self.output_query = BertOutput(config)
        self.layernorm_after_self_attention = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after_cross_attention = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            query_length=0,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]

        attention_output = self.layernorm_after_self_attention(attention_output + hidden_states)

        present_key_value = self_attention_outputs[-1]

        if self.has_cross_attention and encoder_hidden_states is not None:
            
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                    outputs + cross_attention_outputs[1:-1]
            )  # add cross attentions if we output attention weights

            attention_output = self.layernorm_after_cross_attention(attention_output + self_attention_outputs[0])

            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )

        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        outputs = (layer_output,) + outputs

        outputs = outputs + (present_key_value,)
        # First element: layer_output, second: attention_output, third: cross_attention_output, last: present_key_value

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [BertLayer(config, i) for i in range(config.num_hidden_layers)]
        )

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            query_length=0,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        next_decoder_cache = () if use_cache else None

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:  # Gradient checkpointing, rarely used

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(
                            *inputs, past_key_value, output_attentions, query_length
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    query_length,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.hidden_size
        self.d_in = self.d_model * 2 if config.use_tissue else self.d_model
        self.decoder = nn.Sequential(
            nn.Linear(self.d_in, self.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, 1),
        )

    def forward(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)

        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_extended_attention_mask(
            self,
            attention_mask: Tensor,
            input_shape: Tuple[int],
            device: device,
            is_decoder: bool,
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        batch_size, seq_length = input_shape

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if is_decoder:
                # Create a mask for the decoder
                extended_attention_mask = torch.zeros((batch_size, seq_length, seq_length), device=device)
                for b in range(batch_size):
                    for i in range(seq_length):
                        if attention_mask[b, i] == 1:  # If the position is an unknown gene
                            extended_attention_mask[b, i, i] = 1  # only allow self-attention
                        else:
                            extended_attention_mask[b, :, i] = 1  # Allow attention to all positions
                extended_attention_mask = extended_attention_mask[:, None, :, :]
            else:
                # Provided a padding mask of dimensions [batch_size, seq_length]
                # - if the model is a decoder, apply a causal mask in addition to the padding mask
                # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            is_decoder=False,
            gene_exp=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is None:
            assert (
                    gene_exp is not None
            ), "You have to specify gene_exp when input_ids is None"

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] - self.config.query_length
            if past_key_values is not None
            else 0
        )

        query_length = 0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            gene_exp=gene_exp,
        )

        input_shape = embedding_output.size()[:-1]
        batch_size, seq_length = input_shape
        device = embedding_output.device

        if attention_mask is None:
            attention_mask = torch.zeros(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if is_decoder:
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device, is_decoder)
        else:
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device, is_decoder)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_attention_mask_shape = (encoder_batch_size, seq_length, encoder_sequence_length)

            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_attention_mask_shape, device=device)
            
            # Ensure the encoder_attention_mask is broadcastable to [batch_size, num_heads, decoder_seq_length, encoder_seq_length]
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None


        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            query_length=query_length,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertLMHeadModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)

        self.batch_encoder = TissueEncoder(num_embeddings=config.tissue_type, embedding_dim=config.hidden_size)

        self.predict_head = BertOnlyMLMHead(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            past_key_values=None,
            use_cache=True,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            return_logits=False,
            is_decoder=True,
            reduction="mean",
            gene_exp=None,
            batch_id=None,
            output_cls=False,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if labels is not None:
            use_cache = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
            gene_exp=gene_exp,
        )

        sequence_output = outputs[0]  # [batch_size, seq_length, hidden_size]

        if output_cls:
            cls_embed = sequence_output[:, 0, :]

        if batch_id is not None:
            batch_embeds = self.batch_encoder(batch_id)
            sequence_output = torch.cat([sequence_output, batch_embeds.unsqueeze(1).repeat(1, sequence_output.size(1), 1)], dim=2)

        prediction_scores = self.predict_head(sequence_output)  # [batch_size, seq_length, 1]

        if return_logits:
            return prediction_scores[:, :-1, :].contiguous()

        prediction_scores = prediction_scores.squeeze(-1)
        
        lm_loss = None
        if labels is not None:
            labels = labels.contiguous()
            mask = (gene_exp == -1)  # -1代表mask
            prediction_scores = prediction_scores[:, 1:]
            mask_prediction_scores = prediction_scores * mask
            mask_labels = labels * mask

            loss_fct = nn.MSELoss(reduction=reduction)
            lm_loss = loss_fct(
                mask_prediction_scores,
                mask_labels,
            )

            if reduction == "none":
                lm_loss = lm_loss.view(prediction_scores.size(0), -1).sum(1)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        if output_cls:
            return CausalLMOutputWithCrossAttentions(
                loss=lm_loss,
                logits=prediction_scores,
                past_key_values=outputs.past_key_values,
                hidden_states=cls_embed,
                attentions=outputs.attentions,
                cross_attentions=outputs.cross_attentions,
            )
        else:
            return CausalLMOutputWithCrossAttentions(
                loss=lm_loss,
                logits=prediction_scores,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                cross_attentions=outputs.cross_attentions,
            )
