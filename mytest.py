import torch
from thop import profile, clever_format
from models.binarized_modules import BinarizeAttention
import models
from pytorch_tools import print_model_param_flops
from models.resnet_hard_bnat import resnet_bnat_pruned, resnet_hard_prune

checkpoint = torch.load('/media/datasets/syr/Binary_pruning/imagenet/resnet_bnat18_2020-03-26_16-22-44/checkpoint.pth.tar',
                        map_location='cpu')



print(checkpoint['epoch'])
model_config_2 = { 'dataset': 'imagenet', 'depth': 18,
                  'mode': True}

pruned_model = resnet_bnat_pruned(**model_config_2)


pruned_model.load_state_dict(checkpoint['state_dict'])
for m in pruned_model.modules():
    if isinstance(m, BinarizeAttention):  # mask
        m.weight.org = m.weight.data.clone()
pruned_model, compress_ratio, cfg, _ = resnet_hard_prune(pruned_model, False, depth=18,
                                                         dataset='imagenet')  # 是否打开量化感知层

# test pruned model FLOPs
pruned_model.eval()  # now model has no bnan

input = torch.rand(1, 3, 224, 224)
MAC, params = profile(pruned_model, inputs=(input,))
print('MAC: {}, params: {}'.format(MAC, params))
MAC, params = clever_format([MAC, params], '%.6f')
print('MAC: {}, params: {}'.format(MAC, params))
print_model_param_flops(pruned_model, [224, 224])
print('successfully pruning ratio {} '.format(1 - compress_ratio))