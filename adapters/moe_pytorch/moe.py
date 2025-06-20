from math import e
import torch
import torch.nn.functional as F

from argparse import Namespace
from dataclasses import dataclass

from copy import deepcopy
from torch import nn
from ..lora_pytorch import LoRA
from ..lora_pytorch.modules import MultiheadAttentionLoRAModule, BaseLoRAModule

from .modules import (
    MultiheadAttentionExpertModule
)

from .routing import get_routing_strategy, TopKRoutingConfig

from torch import Tensor, nn
from typing import (
    List,
    Union,
    Optional,
    Tuple,
    cast,
    overload,
)

@dataclass
class MoEOptions:
    num_experts: int = 3
    gate_type: str = "linear"
    gate_bias: bool = False
    routing_strategy: str = "topk"
    num_experts_per_tok: int = 2 # For topk routing
    lora_experts: bool = True
    lora_experts_rank: int = 5
    lora_experts_no_q: bool = False

def namespace_to_moe_opt(ns: Namespace) -> Tuple[MoEOptions, Namespace]:
    """Filter Namespace to create MoEOptions and return the unused fields."""
    valid_keys = MoEOptions.__dataclass_fields__.keys()
    filtered_dict = {k: v for k, v in vars(ns).items() if k in valid_keys}
    discarded_dict = {k: v for k, v in vars(ns).items() if k not in valid_keys}
    return MoEOptions(**filtered_dict), Namespace(**discarded_dict)

class MoE(nn.Module):
    """
    Basic Mixture of Experts (MoE) module for Linear layers
    * support LoRA experts 
    """
    def __init__(
        self,
        module: nn.Module,
        experts: List[Union[nn.Linear, BaseLoRAModule]],
        opt: Optional[MoEOptions]
    ):
        super().__init__()
        self.opt = opt if opt is not None else MoEOptions()
        assert len(experts) == self.opt.num_experts, f"Number of experts ({len(experts)}) does not match num_experts option ({self.opt.num_experts})"
        assert self.opt.num_experts > 0, "Select at least 1 expert when using MoE"
        
        self.base_expert = module # Frozen module on which MoE is built on
        self.experts = nn.ModuleList(experts) # List of experts, each is a copy of the module
        
        # Build Gate and Routing strategy
        self.route = get_routing_strategy(self.opt.routing_strategy, self.build_routing_config(self.opt.routing_strategy, opt))
        self.gate = self.build_gate_module(self.base_expert, self.opt.num_experts, type=self.opt.gate_type, bias=self.opt.gate_bias)

    def forward(self, inputs: Tensor) -> Tensor: 
        """
        inputs: [Timesteps, Batch, Hidden Size]
        """
        pooled_inputs = inputs.mean(dim=0) # group per batch

        gate_logits = self.gate(pooled_inputs) # Parse through gate
        weights, selected_experts = self.route(gate_logits) # decide routing

        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)

        if not self.opt.lora_experts:
            results = torch.zeros_like(inputs)
        else:
            results = self.base_expert(inputs) # Base expert contribution (for LoRA)
        
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            if len(batch_idx) == 0:
                continue  # Skip expert with no assignments
            
            expert_input = inputs[:, batch_idx]  # All tokens in the whole batch which are assigned to this expert
            expert_weight = weights[batch_idx, nth_expert, None] # Get the weights for this expert
            
            results[:, batch_idx] += expert_weight * expert(expert_input) # expert contribution
        
        return results
    
    @classmethod
    def _from_linear(cls, module: nn.Linear, opt: MoEOptions) -> nn.Module: 
        """Upcycle a linear module into a MoE module with multiple experts."""
        
        experts = []
        for _ in range(opt.num_experts):
            exp = deepcopy(module) # Copy Everything (weights included, we're upcycling)
            if opt.lora_experts:
                # NOTE: We keep only the lora_module !!!
                exp = LoRA._from_linear(module, opt.lora_experts_rank).lora_module
            experts.append(exp)
        
        return MoE(module, experts, opt)
    
    @classmethod
    def _from_multihead_attention(cls, module: nn.MultiheadAttention, opt: MoEOptions) -> nn.Module:
        """Upcycle a MultiheadAttention module into a MoE module with multiple experts."""
        
        experts = []
        for _ in range(opt.num_experts):
            exp = deepcopy(module) # Copy Everything (weights included, we're upcycling)
            if opt.lora_experts:
                # NOTE: We keep only the lora_module !!!
                exp = LoRA._from_multihead_attention(module, opt.lora_experts_rank, opt.lora_experts_no_q).lora_module
            experts.append(exp)
        
        return MultiheadAttentionMoE(module, experts, opt)

    @classmethod
    def from_module(
        cls,
        module: nn.Module, # Module to be converted into MoE
        opt: MoEOptions
    ) -> nn.Module:
        """Inject MoE recursively into an existing module. (Tested with TransformerEncoder)"""
        
        if isinstance(module, nn.Linear):
            return MoE._from_linear(module, opt)
        
        if isinstance(module, nn.MultiheadAttention):
            return MoE._from_multihead_attention(module, opt)
        
        for name, child in module.named_children():
            child = cast(nn.Module, child)
            module._modules[name] = cls.from_module(child, opt)  # type: ignore

        return module
    
    @classmethod
    def build_routing_config(self, routing_strategy, opt):
        """Build the routing configuration based on the provided arguments."""
        if routing_strategy == "topk":
            return TopKRoutingConfig(num_experts_per_tok=opt.num_experts_per_tok)
        else:
            raise ValueError(f"Unsupported routing strategy: {routing_strategy}")
    
    def build_gate_module(
        self,
        module: nn.Linear,
        num_experts: int,
        type: str = "linear",
        bias: bool = False
    ) -> Union[nn.Linear, nn.Sequential]:
        """Build the gate module based on the type specified."""

        if type == "linear":
            return nn.Linear(module.in_features, num_experts, bias=bias)
        elif type == "MLP":
                return nn.Sequential(
                    nn.Linear(module.in_features, module.in_features//2, bias=bias),
                    nn.ReLU(),
                    nn.Linear(module.in_features//2, num_experts, bias=bias)
                )
        else:
            raise ValueError(f"Unsupported gate type: {type}")


import time

class MultiheadAttentionMoE(nn.Module):
    """
    Mixture of Experts (MoE) module for Multihead Attention.
    Each expert is basically a clone of the base MultiheadAttention module.
    For practical use, experts should be LoRA modules (you can also use regular MultiheadAttention modules).
    """
    def __init__(
        self,
        module: nn.MultiheadAttention,
        experts: List[Union[nn.MultiheadAttention, MultiheadAttentionLoRAModule]],
        opt: Optional[MoEOptions]
    ):
        super().__init__()
        self.opt = opt if opt is not None else MoEOptions()
        assert len(experts) == self.opt.num_experts, f"Number of experts ({len(experts)}) does not match num_experts option ({self.opt.num_experts})"
        assert self.opt.num_experts > 0, "Select at least 1 expert when using MoE"
        
        self.base_expert = module#cast(nn.MultiheadAttention, module)
        self.experts = nn.ModuleList([
            MultiheadAttentionExpertModule(self.base_expert, exp)
            for exp in experts
        ])

        # Build Gate and Routing strategy
        self.route = get_routing_strategy(self.opt.routing_strategy, MoE.build_routing_config(self.opt.routing_strategy, opt))
        self.gate = self.build_gate_module(self.base_expert, self.opt.num_experts, type=self.opt.gate_type, bias=self.opt.gate_bias)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]: 
        """
        Input is assumed to be sequence data [Timesteps, Batch, Hidden Size]
        > Q, K, V : [Timesteps, Batch, Hidden Size]
        """
        # Route based on query representation
        gate_logits = self.gate(query.mean(dim=0))
        gate_weights, gate_selection = self.route(gate_logits)
        gate_weights = F.softmax(gate_weights, dim=1, dtype=torch.float).to(query.dtype)
        
        # Base Expert contribution
        attn_output, attn_weights = None, None
        
        if self.opt.lora_experts:
            # Base attention computation (for LoRA experts)
            attn_output, attn_weights = self.base_expert(
                query, key, value, 
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                #average_attn_weights=average_attn_weights,
                #is_causal=is_causal
            )
        else:
            # Initialize with zeros if not using LoRA
            attn_output = torch.zeros_like(query)
            if need_weights:
                L, B = query.size(0), query.size(1)
                attn_weights = torch.zeros(
                    B, L, L, 
                    device=query.device, dtype=query.dtype
                )

        # Compute expert contributions
        for exp_idx, expert in enumerate(self.experts):
            exp_a, exp_a_w = expert(
                exp_idx, gate_selection, gate_weights,
                query, key, value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal
            )
            attn_output += exp_a
            if need_weights and exp_a_w is not None:
                attn_weights += exp_a_w

        return attn_output, attn_weights
    
    def build_gate_module(
        self,
        module: nn.MultiheadAttention,
        num_experts: int,
        type: str = "linear",
        bias: bool = False
    ) -> Union[nn.Linear, nn.Sequential]:
        """Build the gate module based on the type specified."""
        if type == "linear":
            return nn.Linear(module.embed_dim, num_experts, bias=bias)
        elif type == "MLP":
            return nn.Sequential(
                    nn.Linear(module.embed_dim, module.embed_dim//2, bias=bias),
                    nn.ReLU(),
                    nn.Linear(module.embed_dim//2, num_experts, bias=bias)
                )
        else:
            raise ValueError(f"Unsupported gate type: {type}")