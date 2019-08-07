import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import copy
from forward_mapping_functions import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

forward_mapping_dict = {
    'Linear': snip_forward_linear,
    'Conv2d': snip_forward_conv2d,
    'Conv1d': snip_forward_conv1d
}

class Prunner:

    def __init__(self, model, criterion, dataloader):
        self.model = copy.deepcopy(model).to(device)
        self.prun_model = copy.deepcopy(model).to(device)
        self.criterion = criterion.to(device)
        self.dataloader = dataloader
        self.variance_scaling_init()
        self.update_forward_pass()

    def apply_hook(self, masks):
        layers = filter(lambda l: type(l).__name__ in forward_mapping_dict, self.prun_model.modules())
        def apply_masking(mask):
            def hook(weight):
                return weight * mask
            return hook
        for layer, mask in zip(layers, masks):
            assert layer.weight.shape == mask.shape
            layer.weight.data = layer.weight.data * mask
            layer.weight.register_hook(apply_masking(mask))
        
    def prun(self, compression_factor=0.5, num_batch_sampling=1):
        grads, grads_list = self.compute_grads(num_batch_sampling)
        keep_params = int((1 - compression_factor) * len(grads))
        values, idxs = torch.topk(grads / grads.sum(), keep_params, sorted=True)
        threshold = values[-1]
        masks = [(grad / grads.sum() > threshold).float() for grad in grads_list]
        self.apply_hook(masks)
        return self.prun_model, masks

    def compute_grads(self, num_batch_sampling=1):
        moving_average_grads = 0
        for i, (data, labels) in enumerate(self.dataloader):
            if i == num_batch_sampling:
                break
            data, labels = data.to(device), labels.to(device)
            out = self.model(data)
            loss = self.criterion(out, labels)
            self.model.zero_grad()
            loss.backward()
            grads_list = []
            for layer in self.model.modules():
                if type(layer).__name__ in forward_mapping_dict:
                    grads_list.append(torch.abs(layer.weight_mask.grad))
            grads = torch.cat([torch.flatten(grad) for grad in grads_list])
            if i == 0:
                moving_average_grads = grads
                moving_average_grad_list = grads_list
            else:
                moving_average_grads = ((moving_average_grads * i) + grads) / (i + 1)
                moving_average_grad_list = [((mv_avg_grad * i) + grad) / (i + 1)
                                            for mv_avg_grad, grad in zip(moving_average_grad_list, grads_list)]
        return moving_average_grads, moving_average_grad_list

    def variance_scaling_init(self):
        for layer in self.model.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight).to(device))
                nn.init.xavier_normal_(layer.weight)
                layer.weight.requires_grad = False
    
    def update_forward_pass(self):
        for layer in self.model.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.forward = types.MethodType(forward_mapping_dict[type(layer).__name__], layer)

