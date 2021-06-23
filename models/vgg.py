import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = ['vgg']

model_urls = {
    '11': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',   # vgg11_bn
    '13': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',   #vgg13_bn
    '16': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',   #vgg16_bn
    '19': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',   # vgg19_bn
}

cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
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


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG_imagenet(VGG):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG_imagenet, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG_cifar100(VGG):
    def __init__(self, features, num_classes=100, init_weights=True):
        super(VGG_cifar100, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            #nn.Linear(512, 512),
            nn.Linear(512, num_classes)
            # nn.Linear(512, 4096),
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
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def vgg(**kwargs):
    pretrained, num_classes, depth, dataset = map(
        kwargs.get, ['pretrained', 'num_classes', 'depth', 'dataset'])
    depth = depth or 16
    pretrained = pretrained or False
    if pretrained:
        init_weights = False
    else:
        init_weights = True
    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        model = VGG_imagenet(make_layers(cfg[str(depth)], batch_norm=True),
                             num_classes=num_classes, init_weights=init_weights)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls[str(depth)]))
        return model
    elif dataset == 'cifar100':
        num_classes = num_classes or 100
        model = VGG_cifar100(make_layers(cfg[str(depth)], batch_norm=True),
                            num_classes=num_classes, init_weights=init_weights)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls[str(depth)]))
        return model
    else:
        print('Error! Now only support ImageNet and CIFAR100!')


if __name__ == '__main__':
    # model = vgg(dataset='imagenet', depth=16)
    model = vgg(dataset='cifar100', depth=19)

    import torch
    x = torch.rand(2, 3, 32, 32)
    # x = torch.rand(2, 3, 224, 224)
    output = model.forward(x)