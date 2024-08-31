import logging
from datetime import timedelta

from lightning.fabric.utilities.rank_zero import rank_zero_only
from termcolor import colored


class CallBack:
    @staticmethod
    @rank_zero_only
    def on_epoch(tm, best_metric, best_epoch):
        logging.info(colored(f'*** Best {tm}: {best_metric:.5f} {best_epoch} ***', 'blue'))

    @staticmethod
    @rank_zero_only
    def on_train(epoch, update_idx, updates_per_epoch, loss, lr, duration, data_duration, batch_size):
        logging.info(f'{"Train":>5}: {epoch:>3} [{update_idx+1:>4d}/{updates_per_epoch}] '
                     f'({100. * (update_idx+1) / updates_per_epoch:>3.0f}%)]  '
                     f'Loss: {loss.item():#.3g}  '
                     f'LR: {lr:.3e}  '
                     f'Data: {data_duration:>5.2f}s  '
                     f'Batch: {duration:>5.2f}s  '
                     f'TP: {batch_size / duration:>7.2f}/s  '
                     f'ETA: {timedelta(seconds=int((updates_per_epoch - update_idx) * duration))}  '
                     )

    @staticmethod
    @rank_zero_only
    def on_eval(metrics, epoch, num_iter, max_iter, ema):
        log = f'{"Eval" if ema is None else "EMA":>5}: {epoch:>3}: [{num_iter:>4d}/{max_iter}]  '
        if "ConfusionMatrix" in metrics:
            metrics.pop('ConfusionMatrix')
        for k, v in metrics.items():
            log += f'{k}: {v.item():.4f}  '
        log = log[:-3]

        logging.info(log)
