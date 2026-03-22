# LoRA utilities (see paper Logic Collapse Horizon)
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, r: int = 4):
        super().__init__()
        d, k = original_layer.out_features, original_layer.in_features
        self.base_weight = nn.Parameter(original_layer.weight.data.clone(), requires_grad=False)
        self.base_bias = nn.Parameter(original_layer.bias.data.clone(), requires_grad=False) if original_layer.bias is not None else None
        self.lora_A = nn.Parameter(torch.randn(d, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, k))
        self.r = r

    @property
    def effective_weight(self):
        return self.base_weight + self.lora_A @ self.lora_B

    def forward(self, x):
        return F.linear(x, self.effective_weight, self.base_bias)


def inject_lora(model: nn.Module, r: int = 4):
    import copy
    model = copy.deepcopy(model)

    def replace(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r))
            else:
                replace(child)

    replace(model)
    return model
