# Based on SLIP code bases
# https://github.com/facebookresearch/SLIP
# --------------------------------------------------------'
import argparse
from collections import OrderedDict
import json
import math
import os
import sys
import time
try:
    import wandb
except ImportError:
    wandb = None

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from datasets import get_dataset
import models
from tokenizer import SimpleTokenizer
from utils import AverageMeter, ProgressMeter, accuracy
import utils
from torchvision.datasets import ImageFolder
from utils import GaussianBlur, Solarize
from losses import ACLIPLoss, get_metric_names, CLIPLoss



def get_args_parser():
    parser = argparse.ArgumentParser(description='A-CLIP pre-training and evaluation', add_help=False)
    # Data
    parser.add_argument('--dataset', default='yfcc15m', type=str, choices=['yfcc15m', 'cc3m', 'cc12m', 'coco', 'redcaps'])
    parser.add_argument('--metadata', default='yfcc15m.pkl', type=str,
                    help='path to metadata file (see README for details)')
    parser.add_argument('--root', default='', type=str,
                        help='path to dataset root')
    parser.add_argument('--output-dir', default='./', type=str, help='path where to save, empty for no saving')
    # Model
    parser.add_argument('--model', default='ACLIP_VITB16', type=str)
    parser.add_argument('--mask-ratio', default=0., type=float)
    parser.add_argument('--ssl-mlp-dim', default=4096, type=int,
                        help='hidden dim of SimCLR mlp projection head')
    parser.add_argument('--ssl-emb-dim', default=256, type=int,
                        help='output embed dim of SimCLR mlp projection head')
    parser.add_argument('--ssl-scale', default=1.0, type=float,
                        help='loss scale for SimCLR objective')
    parser.add_argument('--ssl-temp', default=0.1, type=float,
                        help='softmax temperature for SimCLR objective')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    # Training
    parser.add_argument('--momentum-ema', default=0.996, type=float, help="""Base EMA
    parameter. The value is increased to 1 during training with cosine schedule.""")
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=2, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--base-lr', default=3e-3, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate', action='store_true', help='eval only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--descriptions', default='training', type=str)


    # For cluster
    parser.add_argument('--imagenet-val-zip', default='', type=str)
    parser.add_argument('--imagenet-val-txt', default='', type=str)
    return parser

def get_model(args):
    print("=> creating model: {}".format(args.model))
    model = getattr(models, args.model)(mask_ratio=args.mask_ratio)
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200,find_unused_parameters=False)

    return model

def get_optim(args, model):
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0},
                    ]

    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                    eps=args.eps, weight_decay=args.wd)
    return optimizer

def load_ckpt(args, model, optimizer, scaler):
    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            args.start_epoch = epoch
            result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(result)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            args.best_acc = checkpoint['best_acc']
            print("=> loaded resume checkpoint '{}' (epoch {})"
                  .format(args.resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # auto-resume from latest checkpoint in output directory
        latest = os.path.join(args.output_dir, 'checkpoint.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            args.start_epoch = latest_checkpoint['epoch']
            model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            args.best_acc = latest_checkpoint['best_acc']
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))
            
def get_loader(args, tokenizer):
    print("=> creating dataset")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([Solarize()], p=0.2),
            transforms.ToTensor(),
            normalize
        ])

    train_dataset = get_dataset(train_transform, tokenizer, args)
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd, 'dataset_catalog.json')) as f:
        root = json.load(f)['imagenet']['path']

    # val_dataset = ImageFolder(os.path.join(root, 'val'), val_transform)
    from datasets import ZipDataset
    val_dataset = ZipDataset(args.imagenet_val_zip, args.imagenet_val_txt, transform=val_transform)

    # dist eval resamples data to pad uneven batch sizes
    # make sure num_samples = 0 mod num_gpus for exact acc
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)
    
    return train_loader, train_sampler, val_loader
   
def main(args):
    utils.init_distributed_mode(args)
    cudnn.benchmark = True

    args.best_acc = 0

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create model
    model = get_model(args)

    # define loss function (criterion) and optimizer
    # criterion = ACLIPLoss(args.ssl_temp).cuda(args.gpu)
    criterion = CLIPLoss().cuda(args.gpu)

    optimizer = get_optim(args, model)
    scaler = amp.GradScaler(enabled=not args.disable_amp)

    load_ckpt(args, model, optimizer, scaler)
    
    # Data loading 
    tokenizer = SimpleTokenizer()
    train_loader, train_sampler, val_loader = get_loader(args, tokenizer)

    if args.evaluate:
        zero_stats = validate_zeroshot(val_loader, model, tokenizer, args)
        if utils.is_main_process():
            with open(os.path.join(args.output_dir, 'eval_log.txt'), 'a') as f:
                f.write(json.dumps(zero_stats) + '\n')
        return
    
    lr_schedule = utils.cosine_scheduler(args.lr, args.lr_end, args.epochs,
        len(train_loader) // args.update_freq, warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start)

    momentum_schedule = utils.cosine_scheduler(args.momentum_ema, 1, args.epochs, len(train_loader), 0)

    if utils.is_main_process() and args.wandb:
        wandb_id = os.path.split(args.output_dir)[-1]
        wandb.init(project='Mamba-CLIP', id=wandb_id, name=wandb_id, config=args, resume='resume')

    print(args)

    print("=> beginning training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_stats = train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, momentum_schedule ,args)

        if (epoch + 1) % args.eval_freq != 0:
            continue

        val_stats = validate_zeroshot(val_loader, model, tokenizer, args)
        ema_val_stats = validate_zeroshot(val_loader, model, tokenizer, args, ema=True)
        acc1 = val_stats['acc1']

        is_best = acc1 > args.best_acc
        args.best_acc = max(acc1, args.best_acc)

        print("=> saving checkpoint")
        utils.save_on_master({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc': args.best_acc,
                'args': args,
            }, is_best, args.output_dir,epoch+1,args.epochs)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     **{f'ema_test_{k}': v for k, v in ema_val_stats.items()},
                     'epoch': epoch}

        if utils.is_main_process():
            if args.wandb:
                wandb.log(log_stats)
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')


def train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, momentum_schedule, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = get_metric_names()
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it]
        
        online_inputs = [inputs[0], inputs[1]]
        online_inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in online_inputs]

        m = momentum_schedule[it]  # momentum parameter
        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(*online_inputs, m)
            loss_dict = criterion(outputs)

            loss = loss_dict['loss']
            loss /= args.update_freq

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()

        if (data_iter + 1) % args.update_freq != 0:
            continue

        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        # clamp logit scale to [0, 100]
        logit_scale_e = 0

        utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
        if hasattr(utils.get_model(model),'logit_scale_e'):
            utils.get_model(model).logit_scale_e.data.clamp_(0, 4.6052)
            logit_scale_e = utils.get_model(model).logit_scale_e.exp().item()

        logit_scale = utils.get_model(model).logit_scale.exp().item()

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()
        
        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            if utils.is_main_process() and args.wandb:
                wandb.log({**{k: v.item() for k, v in loss_dict.items()},
                        'scaler': scaler.get_scale(),
                        'logit': logit_scale,
                        'logit_e': logit_scale_e,
                        })
            progress.display(optim_iter)

    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}


def validate_zeroshot(val_loader, model, tokenizer, args, ema=False):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    print('=> encoding captions')
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd, 'templates.json')) as f:
        templates = json.load(f)['imagenet']

    with open(os.path.join(cwd, 'labels.json')) as f:
        labels = json.load(f)['imagenet']

    with torch.no_grad():
        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            texts = tokenizer(texts).cuda(args.gpu, non_blocking=True)
            class_embeddings = utils.get_model(model).encode_text(texts, ema=ema)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # encode images
            image_features = utils.get_model(model).encode_image(images, ema=ema)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_image = image_features @ text_features.t()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits_per_image, target, topk=(1, 5))
            acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    progress.synchronize()
    print('0-shot * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    return {'acc1': top1.avg, 'acc5': top5.avg}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('A-CLIP training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)



