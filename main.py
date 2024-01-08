from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
from ssl import *


def str2bool(v):

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int,
                        metavar='B', help='training batch size')
    parser.add_argument('--workers', default=4, type=int, help='workers')
    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    parser.add_argument('--resnet', default="resnet18", type=str, help="resnet")
    parser.add_argument('--normalize', default=True, action='store_true', help='normalize')
    parser.add_argument('--projection_dim', default=64, type=int, help='projection_dim')
    parser.add_argument('--optimizer', default="SGD", type=str, help="optimizer")
    parser.add_argument('--weight_decay', default=1.0e-6, type=float, help='weight_decay')
    parser.add_argument('--temperature', default=0.1, type=float, help='temperature')
    parser.add_argument('--model_path', default='log/', type=str,
                        help='model save path')
    parser.add_argument('--model_dir', default='checkpoint/', type=str,
                        help='model save path')

    parser.add_argument('--dataset', default='ImageNet-R')
    parser.add_argument('--gpu', default='0,1,2,3', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--trial', type=int, help='trial')
    parser.add_argument('--adv', default=False, action='store_true', help='adversarial exmaple')
    parser.add_argument('--eps', default=0.01, type=float, help='eps for adversarial')
    parser.add_argument('--bn_adv_momentum', default=0.01, type=float, help='batch norm momentum for advprop')
    parser.add_argument('--alpha', default=1.0, type=float, help='weight for contrastive loss with adversarial example')
    parser.add_argument('--debug', default=False, action='store_true', help='debug mode')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args_parser = parser.parse_args()

    from lib.utils.distributed import handle_distributed
    handle_distributed(args_parser, os.path.expanduser(os.path.abspath(__file__)))

    if args_parser.seed is not None:
        random.seed(args_parser.seed)
        torch.manual_seed(args_parser.seed)

    cudnn.enabled = True
    cudnn.benchmark = args_parser.cudnn

    configer = Configer(args_parser=args_parser)
    data_dir = configer.get('data', 'data_dir')
    if isinstance(data_dir, str):
        data_dir = [data_dir]
    abs_data_dir = [os.path.expanduser(x) for x in data_dir]
    configer.update(['data', 'data_dir'], abs_data_dir)

    project_dir = os.path.dirname(os.path.realpath(__file__))
    configer.add(['project_dir'], project_dir)

    if configer.get('logging', 'log_to_file'):
        log_file = configer.get('logging', 'log_file')
        new_log_file = '{}_{}'.format(log_file, time.strftime("%Y-%m-%d_%X", time.localtime()))
        configer.update(['logging', 'log_file'], new_log_file)
    else:
        configer.update(['logging', 'logfile_level'], None)

    Log.init(logfile_level=configer.get('logging', 'logfile_level'),
             stdout_level=configer.get('logging', 'stdout_level'),
             log_file=configer.get('logging', 'log_file'),
             log_format=configer.get('logging', 'log_format'),
             rewrite=configer.get('logging', 'rewrite'))

    Log.info('batch size: {}'.format(configer.get('train', 'batch_size')))

    model = None
    if configer.get('method') == 'fcn_segmentor':
        if configer.get('phase') == 'train':
            from segmentor.trainer_contrastive import Trainer
            model = Trainer(configer)
        elif configer.get('phase') == 'test':
            from segmentor.tester import Tester 
            model = Tester(configer)    
        elif configer.get('phase') == 'test_offset':
            from segmentor.tester_offset import Tester
            model = Tester(configer)
    else:
        Log.error('Method: {} is not valid.'.format(configer.get('task')))
        exit(1)

    if configer.get('phase') == 'train':
        model.train()
    elif configer.get('phase').startswith('test') and configer.get('network', 'resume') is not None:
        model.test()
    else:
        Log.error('Phase: {} is not valid.'.format(configer.get('phase')))
        exit(1)