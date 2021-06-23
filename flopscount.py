import torch
from thop import profile, clever_format
from models import *

# checkpoint = torch.load('results/resnet56_2020-01-07_21-08-16/checkpoint.pth.tar', map_loaction='cpu')
# checkpoint = torch.load('/home/syr/workspace/BinaryAttention_Pruning/results/resnet_bnat_pruned56_2020-02-25_11-44-52/checkpoint.pth.tar',
#                         map_location='cpu')
model = resnet(depth=56, dataset='cifar10')
# model = resnet_bnat_pruned(cfg= checkpoint['cfg'], depth=56, dataset='cifar10')
# model = resnet(depth=50, dataset='imagenet')

from torchvision.models.resnet import resnet50 as resnet50
# model = resnet50(pretrained=False)

# input = torch.rand(1, 3, 224, 224)
input = torch.rand(1, 3, 32, 32)
MAC, params = profile(model, inputs=(input, ))
print(MAC, params)
MAC, params = clever_format([MAC, params], '%.3f')
print(MAC, params)


from pytorch_tools import print_model_param_flops
# print_model_param_flops(model, [224,224])
print_model_param_flops(model, [32,32])  # use this!
num_parameters = sum([l.nelement() for l in model.parameters()])
print("number of parameters: ", num_parameters)

# from ptflops import get_model_complexity_info

# MAC, params = get_model_complexity_info(model, (3,32,32), as_strings=True, print_per_layer_stat=True)
# print(MAC, params)