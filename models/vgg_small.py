import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from models.binarized_modules import BinarizeAttention
import numpy as np
import torch
import logging
from models.vgg_bnat import vgg_bnat


__all__ = ['vgg_small', 'vgg_hard_prune']

model_urls = {
    '11': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',   # vgg11_bn
    '13': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',   #vgg13_bn
    '16': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',   #vgg16_bn
    '19': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',   # vgg19_bn
}

default_cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    # '16': [48, 48, 'M', 84, 84, 'M', 128, 128, 128, 'M', 420, 420, 420, 'M', 420, 420, 420, 'M'],    
'19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

default_cfg_ba = {
    '16': [0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
    '19': [0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
}

m_index = {
    '11': [1, 1, 2, 2, 2],
    '13': [2, 2, 2, 2, 2],
    '16': [2, 2, 3, 3, 3],
    '19': [2, 2, 4, 4, 4]
}


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_small_layers(cfg, cfg_ba,mode=False, batch_norm=False):
    layers = []
    in_channels = 3
    index_ba = 0 
    for i,v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                if mode:
                    layers += [conv2d, BinarizeAttention(v,1.0,cfg_ba[index_ba]), nn.BatchNorm2d(v),nn.ReLU()]
                else:
                    layers += [conv2d, nn.BatchNorm2d(v),nn.ReLU()]
            else:
                if mode:
                    layers += [conv2d, BinarizeAttention(v,1.0,cfg_ba[index_ba]),nn.ReLU()]
                else:
                    layers += [conv2d,nn.ReLU()]
        
            in_channels = v
            index_ba += 1
    return nn.ModuleList(layers)
    # return nn.Sequential(*layers)


class VGG_imagenet(VGG):
    def __init__(self, features, cfg, bnan_index, mode=True, num_classes=1000, init_weights=True):
        super(VGG_imagenet, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.bnan_index = bnan_index

        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(cfg[-2] * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()
        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-1, 'weight_decay': 5e-4, 'momentum': 0.9},
            30: {'lr': 1e-2},
            60: {'lr': 1e-3},
            90: {'lr': 5e-4},
            120: {'lr': 1e-4}
        }

    def forward(self, x):
        for k in range(len(self.features)):
            # if not self.mode and k in self.bnan_index:  # bnan层不经过
            #     # print(self.features[k])
            #     continue
            x = self.features[k](x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG_cifar100(VGG):
    def __init__(self, cfg,cfg_ba, mode=False, batch_norm=True, num_classes=100, init_weights=True):
        super(VGG_cifar100, self).__init__()
        self.cfg = cfg

        self.features = make_small_layers(cfg,cfg_ba, mode=mode, batch_norm=batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(cfg[-2], 512),
            nn.Linear(512,num_classes),
            # nn.BatchNorm1d(4096),
            # nn.ReLU(True),
            # nn.Dropout(),

            # nn.Linear(4096, 4096),
            # nn.BatchNorm1d(4096),
            # nn.ReLU(True),
            # nn.Dropout(),

            # nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-1, 'weight_decay': 5e-4, 'momentum': 0.9},
            30: {'lr': 1e-2},
            60: {'lr': 1e-3},
            90: {'lr': 5e-4},
            120: {'lr': 1e-4}
        }

    def forward(self, x):
        # x = self.features(x)
        for k in range(len(self.features)):
            # if not self.mode and k in self.bnan_index:  # bnan层不经过
            #     # print(self.features[k])
            #     continue
            x = self.features[k](x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def vgg_small(**kwargs):
    mode, cfg,cfg_ba, pretrained, num_classes, depth, dataset = map(
        kwargs.get, ['mode', 'cfg','cfg_ba', 'pretrained', 'num_classes', 'depth', 'dataset'])
    depth = depth or 16
    pretrained = pretrained or False
    mode = mode or False
    if cfg is None:
        cfg = default_cfg[str(depth)]
    if cfg_ba is None:
        cfg_ba = default_cfg_ba[str(depth)]        
    # bnan_index = find_bnan(depth)


    if pretrained:
        init_weights = False
    else:
        init_weights = True
    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        model = VGG_imagenet(cfg=cfg,cfg_ba=cfg_ba, mode=mode, batch_norm=True,
                             num_classes=num_classes, init_weights=init_weights)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls[str(depth)]))
        return model
    elif dataset == 'cifar100':
        num_classes = num_classes or 100
        model = VGG_cifar100(cfg=cfg,cfg_ba=cfg_ba, mode=mode, batch_norm=True,
                            num_classes=num_classes, init_weights=init_weights)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls[str(depth)]))
        return model
    else:
        print('Error! Now only support ImageNet and CIFAR100!')


def vgg_hard_prune(model, depth=19, dataset='cifar100'):
    print(model)

    cfg = []
    cfg_mask = []
    func = default_cfg[str(depth)]
    counter = 0
    mydct = []
    cfg_id = 0
    for i, (k, v) in enumerate(model.state_dict().items()):
        if len(v.size())==3:
            cfg.append(int(torch.sum(torch.abs(v))))
            cfg_mask.append(v.clone().squeeze())  # p.data就是mask
            counter += 1
            cfg_id += 1
            if func[counter] == 'M':
                cfg.append('M')
                counter += 1
                cfg_id += 1
        else:
            mydct.append(v)
    # 硬剪枝
    print('Pre-processing Successful!')
    print('Cfg: ', cfg)
   
    oldmodel = vgg_small(mode=False, cfg=func, depth=depth, dataset=dataset)
    print(oldmodel)
    oldstate = oldmodel.state_dict()
    # select all conv and fc param without bnan
    for i, (k, v) in enumerate(oldstate.items()):
        #print(k)
        #print(v.size(), mydct[i].size())
        oldstate[k] = mydct[i]
    oldmodel.load_state_dict(oldstate)
 
    newmodel = vgg_small(mode=False, cfg=cfg, depth=depth, dataset=dataset)
    print(newmodel)
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    fcfirst = True
    conv_prune_rate_per_layer = []

    for [m0, m1] in zip(oldmodel.modules(), newmodel.modules()):
        if fcfirst:
            if isinstance(m0, nn.BatchNorm2d):
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
            elif isinstance(m0, nn.Conv2d):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                b1 = m0.bias.data[idx1.tolist()].clone()
                m1.bias.data = b1.clone()
                per_com_ratio = (m1.weight.data.size(0)*m1.weight.data.size(1)*m1.weight.data.size(2)*
                                 m1.weight.data.size(3))/(m0.weight.data.size(0)*m0.weight.data.size(1)*
                                                          m0.weight.data.size(2)*m0.weight.data.size(3))
                logging.info('this conv layer pruning ratio: {} '.format(1 - per_com_ratio))
                conv_prune_rate_per_layer.append(per_com_ratio)
            elif isinstance(m0, BinarizeAttention):
                idx_out = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx_out.size == 1:
                    idx_out = np.resize(idx_out, (1,))
                m1.weight.data = m0.weight.data[idx_out.tolist()].clone()
                m1.weight.org = m0.weight.org[idx_out.tolist()].clone()
            elif isinstance(m0, nn.Linear):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, m1.weight.shape[0]))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                m1.weight.data = m0.weight.data[:, idx0].clone()
                m1.bias.data = m0.bias.data.clone()
                fcfirst = False
        else:
            if isinstance(m0, nn.BatchNorm1d):
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone() 
        
            elif isinstance(m0, nn.Linear):
                print('In shape: {:d}, Out shape {:d}.'.format(m1.weight.shape[1], m1.weight.shape[0]))
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()

    
    #for k, v in newmodel.state_dict().items():
        #if 'features.40.weight' in k:
            #print(k,v)
        # if 'classifier.0.weight' in k:
        #     print(k,v)

    #for k, v in model.state_dict().items():
        #if 'features.52.weight' in k:
            #print(k, v)
        # if 'classifier.0.weight' in k:
        #     print(k,v)
    #for k, v in oldmodel.state_dict().items():
        #if 'features.40.weight' in k:
            #print(k, v)
        # if 'classifier.0.weight' in k:
        #     print(k,v)

    # print(conv_prune_rate_per_layer)
    num_params_ori = sum([param.nelement() for param in model.parameters()])
    num_params_prune = sum([param.nelement() for param in newmodel.parameters()])
    compress_ratio = num_params_prune / num_params_ori
    # print('total pruning ratio {} '.format(1 - compress_ratio))
    # # logging.info('total pruning ratio {} '.format(1 - compress_ratio))
    #
    # float_conv = sum([p.nelement() if len(p.data.size())==4 else 0 for n, p in model.named_parameters()])
    # prune_conv = sum([p.nelement() if len(p.data.size())==4 else 0 for n, p in newmodel.named_parameters()])
    # compress_ratio = prune_conv / float_conv
    # print('total conv pruning ratio {} '.format(1-compress_ratio))
    return newmodel, cfg, cfg_mask, compress_ratio, conv_prune_rate_per_layer, oldmodel



def find_bnan(depth):
    bnan_index = []
    cnt = -3
    mindex = m_index[str(depth)]
    for k in mindex:
        for i in range(k):
            cnt += 4
            bnan_index.append(cnt)
        cnt += 1
    return bnan_index


if __name__ == '__main__':
    # model = vgg_small(dataset='imagenet', depth=16, mode=True)
    model = vgg_small(dataset='cifar100', depth=19, mode=True)
    print(model)
    x = torch.rand(2, 3, 32, 32)
    # x = torch.rand(2, 3, 224, 224)
    # model.eval()
    output = model.forward(x)
    print(output)
    # newmodel, cfg = vgg_hard_prune(model, 16, dataset='imagenet')
    newmodel, cfg, _, _, _, oldmodel= vgg_hard_prune(model, depth=19)
    newmodel.eval()
    # output = newmodel.forward(x)
    # print(output)
    # oldmodel.eval()
    # output = oldmodel.forward(x)
    # print(output)
    # model.eval()
    # output = model.forward(x)
    # print(output)
