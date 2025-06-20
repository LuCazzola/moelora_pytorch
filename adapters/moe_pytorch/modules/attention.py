from tkinter import W
import torch
from torch import nn, Tensor
from model.adapters.lora_pytorch.modules.attention import MultiheadAttentionLoRAModule

from typing import (
    NamedTuple,
    Optional,
    Tuple,
    Union,
    cast
)

class OutProj(NamedTuple):
    weight: Tensor
    bias: Optional[Tensor]


class MultiheadAttentionExpertModule(nn.Module):
    """
    A view object that masquerades as nn.MultiheadAttention but uses expert parameters.
    This allows us to call nn.MultiheadAttention.forward with expert-specific weights.
    """
    def __init__(self, base_expert: nn.MultiheadAttention, expert: Union[nn.MultiheadAttention, MultiheadAttentionLoRAModule]):
        super().__init__()
        # Store references - no copying, no __getattr__
        self.expert = expert
        # Parameters from the base expert to hijack into nn.MultiheadAttention
        self.base = {
            'embed_dim' : base_expert.embed_dim,
            'num_heads' : base_expert.num_heads,
            'dropout' : base_expert.dropout,
            'add_zero_attn' : base_expert.add_zero_attn,
            'batch_first' : getattr(base_expert, 'batch_first', False), # -> since project is on torch 1.7.1 batch_first is not available
            '_qkv_same_embed_dim' : base_expert._qkv_same_embed_dim,
            'bias_k' : base_expert.bias_k,
            'bias_v' : base_expert.bias_v,
            'in_proj_bias' : getattr(base_expert, 'in_proj_bias', None),
            'in_proj_weight': getattr(base_expert, 'in_proj_weight', None) is not None, # -> we don't want to store it, but check if it's present or not
        }
        
    def forward(
        self,
        expert_idx: int,
        gate_selection: Tensor,
        gate_weights: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        
        L, B, H = query.size()  # [Timesteps, Batch, Hidden Size]
        attn_output = torch.zeros_like(query).to(query.device)
        attn_weights = torch.zeros((B, L, L)).to(query.device) if need_weights else None
        
        # Get gate assignments for this expert
        batch_idx, nth_expert = torch.where(gate_selection == expert_idx)
        if len(batch_idx) == 0:
            # Return early if the expert has no assignments
            return attn_output, attn_weights
        
        # Handle optional arguments assignments
        exp_key_padding_mask, exp_attn_mask = None, None
        if key_padding_mask is not None:
            exp_key_padding_mask = key_padding_mask[batch_idx, :]
        if attn_mask is not None:
            exp_attn_mask = attn_mask[:, batch_idx]
        
        # parse Q, K, V and rest through the expert (only tokens assigned to this expert)
        exp_a, exp_a_w = nn.MultiheadAttention.forward( # -> the trick!
            cast(nn.MultiheadAttention, self),
            query[:, batch_idx], key[:, batch_idx], value[:, batch_idx],
            key_padding_mask=exp_key_padding_mask,
            need_weights=need_weights,
            attn_mask=exp_attn_mask,
            #average_attn_weights=average_attn_weights FIXME: you might uncomment this if you're using more recent torch version
            #is_causal=is_causal
        )
        
        # Stack per-expert contributions
        exp_weight = gate_weights[batch_idx, nth_expert]
        attn_output[:, batch_idx, :] += exp_weight[None, :, None] * exp_a
        if need_weights:
            attn_weights[batch_idx, :, :] += exp_weight[:, None, None] * exp_a_w
        
        return attn_output, attn_weights

    def merge_masks(
        self,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        query: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[int]]:
        return nn.MultiheadAttention.merge_masks(
            cast(nn.MultiheadAttention, self),
            attn_mask,
            key_padding_mask,
            query,
        )

    # Explicitly implement all the attributes that nn.MultiheadAttention.forward needs
    @property
    def embed_dim(self) -> int:
        return self.base['embed_dim']

    @property
    def num_heads(self) -> int:
        return self.base['num_heads']

    @property
    def dropout(self) -> float:
        return self.base['dropout']

    @property
    def add_zero_attn(self) -> bool:
        return self.base['add_zero_attn']

    #@property
    #def batch_first(self) -> bool:
    #    return self.base['batch_first']

    @property
    def _qkv_same_embed_dim(self) -> bool:
        return self.base['_qkv_same_embed_dim']

    # for Additive biases, use same of original module
    @property
    def bias_k(self) -> Optional[Tensor]:
        if self.base['bias_k'] is None:
            return None
        return self.base['bias_k']

    @property
    def bias_v(self) -> Optional[Tensor]:
        if self.base['bias_v'] is None:
            return None
        return self.base['bias_v']

    @property
    def in_proj_bias(self) -> Optional[Tensor]:
        bias = self.base['in_proj_bias']
        if bias is None:
            return None
        else:
            return bias.data.detach()
        # TODO: Add support for 'in_proj_bias' in MultiheadAttentionLoRAModule

    # Weight properties
    # NOTE: only expert weights are used, if the expert is a LoRA module, so are the weights
    @property
    def in_proj_weight(self) -> Optional[Tensor]:
        if not self.base['in_proj_weight']:
            return None
        return self.expert.in_proj_weight

    @property
    def q_proj_weight(self) -> Optional[Tensor]:
        if self.expert.q_proj_weight is None:
            return None
        return self.expert.q_proj_weight

    @property
    def k_proj_weight(self) -> Optional[Tensor]:
        if self.expert.k_proj_weight is None:
            return None
        return self.expert.k_proj_weight
    
    @property
    def v_proj_weight(self) -> Optional[Tensor]:
        if self.expert.v_proj_weight is None:
            return None
        return self.expert.v_proj_weight
    
    @property
    def out_proj(self) -> OutProj:
        weight = self.expert.out_proj.weight
        bias = self.expert.out_proj.bias
        return OutProj(weight, bias)


# NOTE: PyTorch MultiheadAttention uses parameter efficiency tricks:
# - Same embed_dim for Q,K,V → single in_proj_weight (combined)
# - Different embed_dims → separate q/k/v_proj_weight modules
# This affects LoRA expressivity: decomposing 1 vs 3 weight modules differs significantly.
# MultiheadAttentionLoRAModule always creates 3 separate LoRA modules internally,
# then concatenates them when in_proj_weight is requested for compatibility.