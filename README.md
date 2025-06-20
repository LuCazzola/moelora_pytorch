# MoeLoRA PyTorch

A lightweight PyTorch implementation for injecting Mixture of Experts (MoE) capabilities into existing models using LoRA adapters and weight upcycling.

## Features

- 🔧 **Plug-and-play**: Easily inject MoE into any existing PyTorch model
- ♻️ **Weight upcycling**: Efficiently repurpose existing model weights 
- 🎯 **LoRA-based experts**: Use low-rank adaptation instead of full matrices
- 🪶 **Zero dependencies**: Pure PyTorch implementation
- 🚀 **Memory efficient**: Minimal overhead compared to full expert layers

## Quick Start

```python
import torch
from moelora_pytorch import MoeLoRAAdapter

# Load your existing model
model = YourModel()

# Add MoE capabilities
moe_adapter = MoeLoRAAdapter(
    model=model,
    num_experts=8,
    rank=16,
    target_modules=["q_proj", "v_proj"]
)

# Use as normal
output = moe_adapter(input_tensor)
```

## Installation

```bash
pip install moelora-pytorch
```

Or install from source:
```bash
git clone https://github.com/yourusername/moelora_pytorch
cd moelora_pytorch
pip install -e .
```

## How it Works

MoeLoRA transforms existing model layers into mixture of experts by:

1. **Upcycling** existing weights as the base for all experts
2. **Adding LoRA adapters** as lightweight expert-specific modifications  
3. **Routing** inputs to the most relevant experts
4. **Combining** expert outputs with learned gating

This approach provides MoE benefits while keeping memory usage low and maintaining compatibility with existing models.

## Examples

See the `examples/` directory for complete usage examples with popular model architectures.

## License

MIT License - see LICENSE file for details.