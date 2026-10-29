import torch
import torch.nn as nn

class ResponsiveClipperFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min_value, max_value):
        result = torch.clip(x, min_value, max_value)
        dyp_mask = x >= min_value
        dyn_mask = x <= max_value
        ctx.save_for_backward(dyn_mask, dyp_mask)
        return result

    @staticmethod
    def backward(ctx, dy):
        dyn_mask, dyp_mask = ctx.saved_tensors
        dy_mask = (dy > 0.0) * dyp_mask + (dy < 0.0) * dyn_mask
        dy = dy * dy_mask
        return dy, None, None


def responsive_clip(x, min_value=float('-inf'), max_value=float('inf')):
    if torch.jit.is_tracing():
        return torch.clip(x, min_value, max_value)
    return ResponsiveClipperFunction.apply(x, min_value, max_value)


class ResponsiveClipper(nn.Module):
    def __init__(self, min_value=float('-inf'), max_value=float('inf'), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_x = min_value
        self.max_x = max_value

    def forward(self, x):
        return responsive_clip(x, self.min_x, self.max_x)