import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
from models.vgg_small import vgg_hard_prune
from models.binarized_modules import BinarizeAttention
from thop import profile, clever_format
import numpy as np


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='/media/datasets/tgz/result/HZ_AQE_cifar100/VGG19/',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
################################
######### model config #########
################################
parser.add_argument('--dataset', metavar='DATASET', default='cifar100',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='vgg_bnat',
                    choices=model_names,
               	    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',type = str,
                    help='additional architecture configuration')
parser.add_argument('--depth', type=int, default=19,   #56
                    help='resnet depth')

################################
######### train config #########
################################
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=350, type=int, metavar='N',
                    help='number of total epochs to run (default: 300)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
#parser.add_argument('--schedule', default='[150, 225]',type=str,
parser.add_argument('--schedule', default='[ 100, 150, 200]',type=str,
#parser.add_argument('--schedule', default=[0, 80, 120],
                    help='initial learning rate')
#parser.add_argument('--gamma', default=[0.1,0.1,0.1],
parser.add_argument('--gamma', default='[0.1, 0.1, 0.1]',type=str,
                    help='Gamma update for SGD')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('--norm', type=bool, default=False,
                    help='whether open binary attention')  
parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
# parser.add_argument('--pretrain', default='resnet50.pth', type=str, metavar='PATH',
# parser.add_argument('--pretrain',
                    # default='/home/syr/Workspace/BinaryAttention_Pruning/results/c10_56_float/resnet56_2020-03-07_12-55-04/checkpoint.pth.tar',
                    # type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')




def main():
    global args, best_prec1
    
    best_prec1 = 0
    args = parser.parse_args()
    
    args.schedule = eval(args.schedule)
    args.gamma = eval(args.gamma)
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = args.model + str(args.depth) + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset, 'depth': args.depth}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            bnan_state = checkpoint['bnan']
            for n, p in list(model.named_parameters()):
                if 'bnan' in n:
                    p.org = bnan_state[n]
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    elif args.pretrain:
        if args.dataset == 'cifar100':
            list1 = [
                'conv1.weight',
                'bn1.weight',
                'bn1.bias',
                'bn1.running_mean',
                'bn1.running_var', ]

            checkpoint = torch.load(args.pretrain, map_location='cpu')
            mystate = model.state_dict()
            for k, v in mystate.items():
                if k in checkpoint and 'fc' not in k and k not in list1:
                    print(k)
                    mystate[k] = checkpoint[k]
            model.load_state_dict(mystate)
            logging.info('load pretrain model  %s', args.pretrain)
        else:
            checkpoint = torch.load(args.pretrain, map_location='cpu')['state_dict']
            mystate = model.state_dict()
            for k, v in mystate.items():
                if k in checkpoint:
                    mystate[k] = checkpoint[k]
            model.load_state_dict(mystate)
            logging.info('load pretrain model  %s', args.pretrain)

    num_parameters = sum([l.nelement () for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
                                           'lr': args.lr,
                                           'momentum': args.momentum,
                                           'weight_decay': args.weight_decay}})
    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()  # 交叉熵
    criterion.type(args.type)
    model.type(args.type)

    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, 0)
        logging.info('TEST: Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(val_loss=val_loss,
                             val_prec1=val_prec1,
                             val_prec5=val_prec5))
        return

    train_data = get_dataset(args.dataset, 'train', transform['train'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    bnan_params_id = list()
    bnan_params = list()
    for k in range(len(model.features)):
        if isinstance(model.features[k], BinarizeAttention):
            bnan_params_id += list(map(id, model.features[k].parameters()))
            for name, param in model.features[k].named_parameters():
                bnan_params.append(param)
    rest_params = filter(lambda x: id(x) not in bnan_params_id, model.parameters())
    params = [


        {'params': bnan_params, 'lr': args.lr*0.01},
        {'params': rest_params}
    ]

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam' or args.optimzer == 'Adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    logging.info('training regime: %s', regime)
    logging.info(optimizer)
    logging.info("schedule: ")
    logging.info(args.schedule)
    logging.info("gamma: ")
    logging.info(args.gamma)
    lamda_start = 0.5
    lamda_fin = 1.0
    lamda = lamda_start

    for epoch in range(args.start_epoch, args.epochs):
        optimizer = adjust_learning_rate(optimizer, epoch)
        logging.info("--------------------------------")
        logging.info("lr for epoch:%d is",epoch)
        for param_group in optimizer.param_groups:
                logging.info(param_group['lr'])
        logging.info("--------------------------------")
        # optimizer = adjust_optimizer(optimizer, epoch, regime)
        # train for one epoch

        if epoch > 249:
            lamda = 1.0
        else:
            lamda = 1/(1+np.exp(-epoch/10))
        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, criterion, lamda, epoch, optimizer)

        # evaluate on validation set
        val_loss, val_prec1, val_prec5 = validate(
            val_loader, model, criterion, lamda, epoch)
        

        # show now pruning result
        for m in model.modules():
            if isinstance(m,BinarizeAttention):
                #print(torch.sum(torch.where(m.weight.data==0,torch.full_like(m.weight.data,1),torch.full_like(m.weight.data,0))))
                print(torch.sum(m.weight.data))

        
        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)

        bnan = {}
        for n, p in list(model.named_parameters()):
            if hasattr(p, 'org'):
                bnan[n] = p.org
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'bnan': bnan,
            'config': args.model_config,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'regime': regime
        }, is_best, path=save_path)
        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))

        results.add(epoch=epoch + 1, alpha= lamda,train_loss=train_loss, val_loss=val_loss,
                    train_error1= train_prec1, val_error1=val_prec1)
        #results.plot(x='epoch', y=['train_loss', 'val_loss'],
        #             title='Loss', ylabel='loss')
        #results.plot(x='epoch', y=['train_error1', 'val_error1'],
        #             title='Error@1', ylabel='error %')
        #results.plot(x='epoch', y=['train_error5', 'val_error5'],
        #             title='Error@5', ylabel='error %')
        results.save()



    model_config_2 = {'input_size': args.input_size, 'dataset': args.dataset, 'depth': args.depth,
                    'mode': True}

    pruned_model = models.__dict__['vgg_small']
    pre_pruned_model = pruned_model(**model_config_2)
    print(model)
    print(pre_pruned_model)
    pre_pruned_model.load_state_dict(model.state_dict())
    for m in pre_pruned_model.modules():
        if isinstance(m, BinarizeAttention):  # mask
            m.weight.org = m.weight.data.clone()
    pruned_model, cfg, _, compress_ratio, conv_prune_rate_per_layer,_ = vgg_hard_prune(pre_pruned_model,
                                                                                        depth=args.depth,
                                                                                        dataset=args.dataset)

    # test pruned model FLOPs
    pre_pruned_model.eval()
    pruned_model.eval()  # now model has no bnan

    if args.dataset == 'imagenet':
        input = torch.rand(1, 3, 224, 224)
        MAC, params = profile(pruned_model.eval(), inputs=(input,))
        logging.info('MAC: {}, params: {}'.format(MAC, params))
        MAC, params = clever_format([MAC, params], '%.3f')
        logging.info('MAC: {}, params: {}'.format(MAC, params))
        
        input = torch.rand(1, 3, 224, 224)
        MAC, params = profile(pruned_model.eval(), inputs=(input,))
        logging.info('MAC: {}, params: {}'.format(MAC, params))
        MAC, params = clever_format([MAC, params], '%.3f')
        logging.info('MAC: {}, params: {}'.format(MAC, params))

    else:
        input = torch.rand(1, 3, 32, 32)
        MAC, params = profile(pre_pruned_model.eval(), inputs=(input,))
        logging.info('Before pruning: MAC: {}, params: {}'.format(MAC, params))
        MAC, params = clever_format([MAC, params], '%.3f')
        logging.info('Before pruning: MAC: {}, params: {}'.format(MAC, params))

        input = torch.rand(1, 3, 32, 32)
        MAC, params = profile(pruned_model.eval(), inputs=(input,))
        logging.info('After pruning: MAC: {}, params: {}'.format(MAC, params))
        MAC, params = clever_format([MAC, params], '%.3f')
        logging.info('After pruning: MAC: {}, params: {}'.format(MAC, params))

    logging.info('successfully pruning ratio {} '.format(1 - compress_ratio))

    pruned_model.type(args.type)
    val_loss, val_prec1, val_prec5 = validate(val_loader, pruned_model, criterion,1.0, 0)
    logging.info('TEST: Validation Loss {val_loss:.4f} \t'
                 'Validation Prec@1 {val_prec1:.3f} \t'
                 'Validation Prec@5 {val_prec5:.3f} \n'
                 .format(val_loss=val_loss,
                         val_prec1=val_prec1,
                         val_prec5=val_prec5))
    save_checkpoint({
        'model': 'vgg_bnat_pruned',
        'state_dict': pruned_model.state_dict(),  # contains BinarizeAttention weight org
        'cfg': cfg,  # prune cfg
        'prec1': val_prec1,
        'conv_prune_rate_per_layer': conv_prune_rate_per_layer,
    }, False, path=save_path, filename='pruned_' + str(val_prec1) + '.pth.tar')


def adjust_learning_rate(optimizer, epoch):
    #for param_group in optimizer.param_groups:
    #    param_group['lr'] = param_group['lr'] * args.gamma[i]
    for i in range(len(args.schedule)):
        if epoch == args.schedule[i]:
            for param_group in optimizer.param_groups:
                #print(param_group)
                param_group['lr'] = param_group['lr'] * args.gamma[i]
            break
    # if epoch in args.schedule:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = param_group['lr'] * args.gamma
    return optimizer


def forward(data_loader, model, criterion, lamda=0.5, epoch=0, training=True, optimizer=None):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    for m in model.modules():
        if isinstance(m, BinarizeAttention):
            m.alpha = lamda
            print (m.alpha)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.cuda()

        if not training:
            with torch.no_grad():
                input_var = Variable(inputs.type(args.type), volatile=not training)
                target_var = Variable(target)
                # compute output
                output = model(input_var)
        else:
            input_var = Variable(inputs.type(args.type), volatile=not training)
            target_var = Variable(target)
            # compute output
            output = model(input_var)
        #loss_weight = 0.
        #for m in model.modules():
        #    if isinstance(m,BinarizeAttention):
        #        loss_weight += torch.sum(torch.pow(m.weight.org,2)) 
        loss = criterion(output, target_var)# + loss_weight*0.001
        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.data.copy_(p.org)
            if args.norm:
                updateBnAn(model)
            optimizer.step()
            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.org.copy_(p.data)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def updateBnAn(model):
    for m in model.modules():
        if isinstance(m, BinarizeAttention):
            m.weight.grad.data.add_(0.0001*torch.sign(m.weight.grad.data)) 


def train(data_loader, model, criterion,lamda, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, lamda,epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion,lamda, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion,lamda, epoch,
                   training=False, optimizer=None)


if __name__ == '__main__':
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    main()