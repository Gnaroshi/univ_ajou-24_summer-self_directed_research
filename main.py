import gc

import hydra
import torch
import wandb
from omegaconf import DictConfig
from timm.utils import ModelEmaV3

from src.data import get_dataloader
from src.engine import Engine
# from src.engine.fixmatch_engine import FixMatchEngine
from src.engine.fixmatch_simclr_engine import FixMatchSimCLREngine
# from src.engine.fixmatch_simclr_engine_ul import FixMatchSimCLREngine

from src.initialize import setup_fabric, ObjectFactory
from src.misc import print_meta_data
from src.utils import resume
from src.models import *


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    fabric = setup_fabric(cfg)
    loaders = get_dataloader(cfg.train.setup, cfg)
    factory = ObjectFactory(cfg, fabric)

    model = factory.create_model()

    model_ema = ModelEmaV3(model, cfg.train.ema) if cfg.train.ema else None

    optimizer, scheduler, n_epochs = factory.create_optimizer_and_scheduler(model, len(loaders[0]))
    criterion = factory.create_criterion()

    model, model_ema, optimizer, scheduler, start_epoch = resume(model, model_ema, optimizer, scheduler, cfg, fabric)
    model, optimizer = fabric.setup(model, optimizer)
    model_ema = model_ema if model_ema is None else fabric.setup_module(model_ema)

    cfg = factory.cfg
    fabric.loggers[0].update_config(cfg) if cfg.wandb else None
    print_meta_data(cfg, model, *loaders) if cfg.is_master else None
    engine = FixMatchSimCLREngine(cfg, fabric, model, model_ema, criterion, optimizer, scheduler, loaders,
                            (start_epoch, n_epochs))

    engine()

    if cfg.is_master:
        torch.cuda.empty_cache()
        gc.collect()
        wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
