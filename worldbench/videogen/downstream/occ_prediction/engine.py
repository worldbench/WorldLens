import os
from . import utils
import logging
import argparse
import importlib
import torch
import torch.distributed
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed, multi_gpu_test, single_gpu_test
from mmdet3d100rc6.datasets import build_dataset, build_dataloader
from mmdet3d100rc6.models import build_model

import sys
sys.path.append('worldbench/third_party/SparseOcc')

def evaluate(dataset, results):
    metrics = dataset.evaluate(results, jsonfile_prefix=None)

    logging.info('--- Evaluation Results ---')
    for k, v in metrics.items():
        logging.info('%s: %.4f' % (k, v))

    return metrics


def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(description='Validate a detector')
    parser.add_argument('--method_name', help='test config file path')
    parser.add_argument('--config', default='worldbench/third_party/SparseOcc/configs/r50_nuimg_704x256_8f_60e.py')
    parser.add_argument('--weights', default='pretrained_models/perception/sparseocc_r50_nuimg_704x256_8f_60e_v1.1.pth')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    
    if arg_list is not None:
        args = parser.parse_args(arg_list)
    else:
        args = parser.parse_args()

    return args

def run(args):

    # parse configs
    cfgs = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfgs.merge_from_dict(args.cfg_options)
    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')

    # MMCV, please shut up
    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger(__name__, logging.WARNING)

    # you need GPUs
    assert torch.cuda.is_available()

    # determine local_rank and world_size
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(args.world_size)

    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    if local_rank == 0:
        utils.init_logging(None, cfgs.debug)
    else:
        logging.root.disabled = True

    logging.info('Using GPU: %s' % torch.cuda.get_device_name(local_rank))
    torch.cuda.set_device(local_rank)

    if world_size > 1:
        logging.info('Initializing DDP with %d GPUs...' % world_size)
        dist.init_process_group('nccl', init_method='env://')

    logging.info('Setting random seed: 0')
    set_random_seed(0, deterministic=True)
    cudnn.benchmark = True

    logging.info('Loading validation set from %s' % cfgs.data.val.data_root)
    val_dataset = build_dataset(cfgs.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=world_size,
        dist=world_size > 1,
        shuffle=False,
        seed=0,
    )

    logging.info('Creating model: %s' % cfgs.model.type)
    model = build_model(cfgs.model)
    model.cuda()

    if world_size > 1:
        model = MMDistributedDataParallel(model, [local_rank], broadcast_buffers=False)
    else:
        model = MMDataParallel(model, [0])

    if os.path.isfile(args.weights):
        logging.info('Loading checkpoint from %s' % args.weights)
        load_checkpoint(
            model, args.weights, map_location='cuda', strict=True,
            logger=logging.Logger(__name__, logging.ERROR)
        )

    if world_size > 1:
        results = multi_gpu_test(model, val_loader, gpu_collect=True)
    else:
        results, tokens = single_gpu_test(model, val_loader)

    # save occ
    if local_rank == 0:
        save_path = f'generated_results/{args.method_name}/occ_predictions/'
        for token, predicted_occ in zip(tokens, results):
            out_file = os.path.join(save_path, f'{token}.pt')
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            torch.save(predicted_occ, out_file)
    return
    if local_rank == 0:
        evaluate(val_dataset, results)


if __name__ == '__main__':
    arg_list = ["--cfg-options", 
                "data.val.ann_file=generated_results/gt/nuscenes_infos_temporal_val_3keyframes_gen0.pkl",
                "data.workers_per_gpu=1"]

    args = parse_args(arg_list)
    run(args)
