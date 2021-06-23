# model compression
Implementation with PyTorch. 

复现人：刘宇昂

## Data
### ImageNet
Download the ImageNet dataset from [here](http://image-net.org/download-images).
No need to split the val set into corresponding folders because we provide api codes to load data:
```
from mydataset.imagenet_dataset import ImageNetDataset
```
### CIFAR10
Use ```torchvision.datasets.CIFAR10()```

## Contents
model-compress




##训练全精度：
```
python3 main_binary.py --model vgg16 --save vgg16_cifar10 --dataset cifar10 --gpus 0 --epoch 300
python3 main_resnet_cifar10.py --model resnet  --depth 56 \
                               --dataset cifar10 --gpus 1 --epoch 200

```

##训练带attention量化系数：
###从零训练
```
python3 main_binary.py --model vgg16_cifar10_bnat --save vgg16_cifar10_bnat --dataset cifar10 --gpus 0 --epoch 300
python3 main_resnet_cifar10.py --model resnet_bnat  --depth 56 \
                               --dataset cifar10 --gpus 1 --epoch 200
                               
python main_resnet_cifar10.py --epochs 300 --model resnet_bnat --norm True      # norm就是加了一个L1正则项                         

```
###接着全精度模型带着量化层训练
```
python3 main_resnet_cifar10_pretrained.py --model resnet_bnat --depth 56 \
                                          --dataset cifar10 --gpus 1 --epoch 200 \
                                          --pretrained [path to float model]
                                          
python3 main_resnet_cifar10_pretrained.py --model resnet_bnat --depth 56 \
                                          --dataset cifar10 --gpus 1 --epoch 200 \
                                          --pretrained results/resnet_bnat56_2020-01-09_13-02-22/checkpoint.pth.tar

```

###根据训练好的量化层剪枝后微调
```
python3 finetune_binary.py --model resnet_bnat_pruned --depth 56 \
                           --dataset cifar10 --gpus 1 --epoch 200 \
                           --resume [path to model with bnan layers]


```



