import torch
from dataclasses import dataclass
from typing import Union

@dataclass
class TopKRoutingConfig:
    num_experts_per_tok: int

def get_routing_strategy(
    name: str,
    config: TopKRoutingConfig #Union[TopKRoutingConfig, 1, 2,...]
):
    registry = {
        "topk": TopKRouting
    }
    if name not in registry:
        raise ValueError(f"Unsupported routing strategy: {name}")

    return registry[name](config)


class TopKRouting:
    def __init__(self, config: TopKRoutingConfig):
        assert config.num_experts_per_tok > 0
        self.k = config.num_experts_per_tok

    def __call__(self, gate_logits: torch.Tensor):
        return torch.topk(gate_logits, self.k)

