import argparse
from collections import OrderedDict
import os

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from retrieval import coco_retrieval_evaluation, flickr_retrieval_evaluation
import models
from tokenizer import SimpleTokenizer


def get_args_parser():
    parser = argparse.ArgumentParser(description='A-CLIP retrieval evaluations', add_help=False)
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--batch-size', default=256, type=int, help='batch_size')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument(
        "--coco_data_dir", default="./data/coco2017", type=str, help="coco dataset"
    )
    parser.add_argument(
        "--flickr-data-dir", default="./data/flickr", type=str, help="flickr dataset"
    )
    return parser


def main(args):
    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        ckpt_path = args.resume
    elif os.path.isfile(os.path.join(args.output_dir, 'checkpoint_best.pt')):
        ckpt_path = os.path.join(args.output_dir, 'checkpoint_best.pt')
    else:
        raise Exception('no checkpoint found')

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    
    # create model
    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))
    model = getattr(models, old_args.model)()
    model.cuda()
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))

    cudnn.benchmark = True
    args.distributed = False
    
    tokenizer = SimpleTokenizer()
    val_transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            lambda x: x.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    res = coco_retrieval_evaluation(model, val_transform, tokenizer, args)
    print("COCO results:", res[0])
    res = flickr_retrieval_evaluation(model, val_transform, tokenizer, args)
    print("Flickr results:", res[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('A-CLIP retrieval evaluations', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
