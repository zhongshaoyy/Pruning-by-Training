import torch.nn as nn
import torch
import torchvision.transforms as transforms
import math
from models.binarized_modules import BinarizeAttention
# 按照官方结构建立的cifar100网络


__all__ = ['resnet_c100_bnat']
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bnan1 = BinarizeAttention(planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)    # 这一层先不加
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bnan1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bnan1 = BinarizeAttention(planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bnan2 = BinarizeAttention(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bnan1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bnan2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.ModuleList(layers)
        # return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for k in range(len(self.layer1)):
            x = self.layer1[k](x)
        for k in range(len(self.layer2)):
            x = self.layer2[k](x)
        for k in range(len(self.layer3)):
            x = self.layer3[k](x)
        for k in range(len(self.layer4)):
            x = self.layer4[k](x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet_cifar100(ResNet):
    def __init__(self, num_classes=100,
                 block=Bottleneck, layers=[3, 4, 6, 3]):
        super(ResNet_cifar100, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,   # !!!!!!!
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = lambda x: x
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # self.regime = {        # imagenet
        #     0: {'optimizer': 'SGD', 'lr': 1e-1,
        #         'weight_decay': 1e-4, 'momentum': 0.9},
        #     30: {'lr': 1e-2},
        #     60: {'lr': 1e-3, 'weight_decay': 0},
        #     90: {'lr': 1e-4}
        # }
        # self.regime = {    # he  ---best   cifar100  resnet50/32/18/101
        #     0: {'optimizer': 'SGD', 'lr': 0.1,  # 0.01
        #         'weight_decay': 5e-4, 'momentum': 0.9},
        #     150: {'lr': 0.01},
        #     225: {'lr': 0.001},
        # }
        self.regime = {  # he  ---best   cifar100  resnet50/32/18/101
            0: {'optimizer': 'SGD', 'lr': 0.1,  # 0.01
                'weight_decay': 5e-4, 'momentum': 0.9},
            60: {'lr': 0.02},
            120: {'lr': 0.004},
            160: {'lr': 0.0008}
        }
        # self.regime = {  #    cifar100  350epoch
        #     0: {'optimizer': 'SGD', 'lr': 0.1,  # 0.01
        #         'weight_decay': 5e-4, 'momentum': 0.9},
        #     150: {'lr': 0.01},
        #     250: {'lr': 0.001},
        #
        # }



def resnet_c100_bnat(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    depth = depth or 50
    assert dataset == 'cifar100', 'Error! not cifar100'
    num_classes = 100
    if depth == 18:
        return ResNet_cifar100(num_classes=num_classes,
                               block=BasicBlock, layers=[2, 2, 2, 2])
    if depth == 34:
        return ResNet_cifar100(num_classes=num_classes,
                               block=BasicBlock, layers=[3, 4, 6, 3])
    if depth == 50:
        return ResNet_cifar100(num_classes=num_classes,
                               block=Bottleneck, layers=[3, 4, 6, 3])
    if depth == 101:
        return ResNet_cifar100(num_classes=num_classes,
                               block=Bottleneck, layers=[3, 4, 23, 3])
    if depth == 152:
        return ResNet_cifar100(num_classes=num_classes,
                               block=Bottleneck, layers=[3, 8, 36, 3])



if __name__ == '__main__':
    model = resnet_c100_bnat(dataset='cifar100', depth = 50)
    print(model)
    x = torch.rand(1, 3, 32, 32)
    output = model.forward(x)
    print(output)