import torch
import torch.nn as nn
import torch.nn.functional as F

def snip_forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)

def snip_forward_conv1d(self, x):
    return F.conv1d(x, self.weight * self.weight_mask, self.bias)

def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)
