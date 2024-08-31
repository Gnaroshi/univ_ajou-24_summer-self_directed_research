import os

import torch
from lightning import Fabric
from termcolor import colored

from .callback import CallBack
from .logger import WandBNCSVLogger


def setup_environ():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['OMP_NUM_THREADS'] = str(4)
    os.environ['HYDRA_FULL_ERROR'] = str(1)

#
def seed_everything(seed, local_rank=0, workers=0):
    import random
    import numpy

    each_seed = seed + local_rank

    os.environ['PL_GLOBAL_SEED'] = str(seed)
    os.environ['PL_SEED_WORKERS'] = str(workers)

    random.seed(each_seed)
    numpy.random.seed(each_seed)
    torch.manual_seed(each_seed)
    torch.cuda.manual_seed_all(each_seed)


def setup_fabric(cfg):
    setup_environ()

    if isinstance(cfg.gpu, int) and cfg.gpu > -1:
        cfg.gpu = [cfg.gpu]
    if cfg.gpu != -1 and len(cfg.gpu) == 1 and cfg.strategy != 'auto':
        print(colored('[Config Change]', 'yellow'), f'{cfg.strategy} is turned off. Running on single strategy')
        cfg.strategy = 'auto'

    fabric = Fabric(
        accelerator=cfg.accelerator,
        strategy=cfg.strategy,
        devices=cfg.gpu,
        precision=cfg.precision,
        num_nodes=1,
        loggers=[WandBNCSVLogger(cfg)],
        callbacks=[CallBack]
    )

    fabric.launch()
    seed_everything(cfg.train.seed, fabric.local_rank)

    cfg.is_master = True if fabric.global_rank == 0 else False
    cfg.world_size = fabric.world_size
    cfg.distributed = torch.distributed.is_initialized()

    return fabric
