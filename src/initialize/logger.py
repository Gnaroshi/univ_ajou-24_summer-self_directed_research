# Copyright SoongE.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import csv
import os
from argparse import Namespace
from glob import glob
from typing import Dict, Union, Any, Optional

import hydra
import wandb
from lightning.fabric.loggers.logger import Logger
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import OmegaConf


class WandBNCSVLogger(Logger):
    def __init__(self, cfg, csv_name='summary.csv'):
        super().__init__()
        self.cfg = cfg

        self.wandb = cfg.wandb
        self.step = 0
        self.id = None

        self.csv_name = csv_name
        self.write_header = True

        if self.wandb:
            if cfg.train.resume:
                base_path = hydra.utils.get_original_cwd()
                try:
                    self.id = glob(os.path.join(base_path, cfg.train.resume, 'wandb', 'run-*'))[0].rsplit('-', 1)[-1]
                except FileNotFoundError:
                    print(f'Wandb folder is not founded at {os.path.join(base_path, cfg.train.resume)}')
            self._init_wandb()

    @rank_zero_only
    def _init_wandb(self):
        wandb.init(project=self.cfg.info.project,  config=OmegaConf.to_container(self.cfg),
                   name=f"{self.cfg.name}", id=self.id, settings=wandb.Settings(_disable_stats=True), resume='allow')
        self.cfg.info.id = wandb.run.id

    @rank_zero_only
    def update_config(self, cfg):
        wandb.config.update(OmegaConf.to_container(cfg), allow_val_change=True)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int]):
        wandb.log(metrics) if self.wandb else None
        metrics['epoch'] = step
        self.log_csv(metrics)

    @rank_zero_only
    def log_csv(self, metrics):
        with open(self.csv_name, mode='a') as f:
            dw = csv.DictWriter(f, fieldnames=metrics.keys())
            if self.write_header:
                dw.writeheader()
            dw.writerow(metrics)
        self.write_header = False

    @rank_zero_only
    def log(self, logs):
        raise NotImplementedError("The `WandBNCSVLogger` does not yet support log methods.")

    @property
    def name(self) -> Optional[str]:
        return self.cfg.info.project

    @property
    def version(self) -> Optional[Union[int, str]]:
        return wandb.run.id

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("The `WandBNCSVLogger` does not yet support log hyperparams.")
