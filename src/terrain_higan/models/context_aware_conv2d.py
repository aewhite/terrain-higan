import torch
import torch.nn as nn

class ContextAwareConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, context_channels, kernel_size, stride=1, groups=1):
        super(ContextAwareConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups)
        self.linear = nn.Linear(context_channels, 2 * out_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.conv.no_auto_init = True
        self.linear.no_auto_init = True

    def forward(self, x, context=None):
        x = self.conv(x)
        gamma_beta = self.linear(context)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        x = x * (1.0 + gamma) + beta
        return x