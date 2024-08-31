'''
    modified by Mingyu Jung
'''

from time import perf_counter

import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric

# from base_engine import Engine
from src.engine import Engine
import src.data.simclr_augmentation as sa


class FixMatchSimCLREngine(Engine):
    def __init__(self, cfg, fabric, model, model_ema, criterion, optimizer, scheduler, loaders, epochs):
        self.local_rank = fabric.local_rank
        self.world_size = fabric.world_size
        self.device = fabric.device
        self.fabric = fabric
        self.cfg = cfg

        self.distributed = True if fabric.world_size > 1 else False
        self.dist_bn = cfg.train.dist_bn

        self.start_epoch, self.num_epochs = epochs
        self.logging_interval = cfg.train.log_interval
        self.model_name = cfg.model.model_name
        self.num_classes = cfg.dataset.num_classes
        self.sample_size = cfg.train.batch_size * cfg.train.optimizer.grad_accumulation * self.world_size

        self.model = model
        self.model_ema = model_ema
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_accumulation = cfg.train.optimizer.grad_accumulation
        self.train_criterion, self.val_criterion = criterion
        self.labeled_train_loader, self.unlabeled_train_loader, self.val_loader = loaders

        self.cm = cfg.train.criteria_metric
        self.data_duration = MeanMetric().to(self.device)
        self.duration = MeanMetric().to(self.device)
        self.losses = MeanMetric().to(self.device)
        self.metric_fn = self._init_metrics(cfg.dataset.task, cfg.train.eval_metrics,
                                            0.5, self.num_classes, self.num_classes, 'macro')
        self.best_metric = self.best_epoch = 0
        self.lambda_u = cfg.train.lambda_u

        self.clvg = sa.ContrastiveLearningViewGenerator(sa.simclr_transform(size=32, device=self.device),
                                                        n_views=self.cfg.fs.n_views)
        self.sa_criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def consistency_loss(self, logits_w, logits_s, T=1.0, p_cutoff=0.95, hard_labels=True):
        logits_w = logits_w.detach()
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()

        if hard_labels:
            masked_loss = torch.nn.functional.cross_entropy(logits_s, max_idx, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            log_pred = torch.nn.functional.log_softmax(logits_s, dim=-1)
            nll_loss = torch.sum(-pseudo_label * log_pred, dim=1)
            masked_loss = nll_loss * mask
        return masked_loss.mean(), mask.mean()

    '''
        info_nce_loss, code from
        https://github.com/sthalles/SimCLR/blob/master/simclr.py
        modified by Mingyu Jung
    '''

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(features.shape[0]) for i in range(self.cfg.fs.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.cfg.fs.temperature
        return logits, labels

    def simclr_loss(self, weak_features, strong_features):
        weak_features = F.normalize(weak_features, dim=1)
        strong_features = F.normalize(strong_features, dim=1)

        similarity_matrix = torch.matmul(weak_features, strong_features.T)

        batch_size = weak_features.size(0)
        labels = torch.arange(batch_size).to(weak_features.device)

        mask = torch.eye(batch_size, dtype=torch.bool).to(weak_features.device)
        positives = similarity_matrix[mask].view(batch_size, -1)
        negatives = similarity_matrix[~mask].view(batch_size, -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.cfg.fs.temperature

        loss = F.cross_entropy(logits, labels)
        return loss

    # unlabeled data에 SimCLR 적용
    # u_w, u_s를 그대로 사용 (feature)
    def iterate(self, model, data, criterion):
        x, y = map(lambda a: a.to(self.device), data[0])
        u_w, u_s, _ = map(lambda a: a.to(self.device), data[1])
        num_x = x.shape[0]
        x = torch.cat((x, u_w, u_s))
        x = x.to(memory_format=torch.channels_last)

        with self.fabric.autocast():
            logits = self.model(x)
            logits_x = logits[:num_x]
            logits_u_w, logits_u_s = logits[num_x:].chunk(2)
            del logits

            Lx = self.train_criterion(logits_x, y)
            Lu, _ = self.consistency_loss(logits_u_w, logits_u_s)
            Lsa = self.simclr_loss(logits_u_w, logits_u_s)
            loss = Lx + self.lambda_u * (Lu + Lsa)

        return loss, logits_x, y

    def train(self, epoch):
        self._reset_metric()

        total_len = len(self.labeled_train_loader)
        accum_steps = self.grad_accumulation
        num_updates = epoch * (update_len := (total_len + accum_steps - 1) // accum_steps)
        total_len = total_len - 1

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        data_start = start = perf_counter()
        for i, data in enumerate(zip(self.labeled_train_loader, self.unlabeled_train_loader)):
            self.data_duration.update(perf_counter() - data_start)
            is_accumulating = (i + 1) % accum_steps != 0 and i != total_len
            update_idx = i // accum_steps

            with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                loss, prob, target = self.iterate(self.model, data, self.train_criterion)
                self.fabric.backward(loss)

            self.losses.update(loss)

            if is_accumulating:
                data_start = perf_counter()
                continue

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.model_ema is not None: self.model_ema.update(self.model, step=num_updates)

            self.duration.update(perf_counter() - start)
            start = perf_counter()
            num_updates += 1
            if update_idx % self.logging_interval == 0 or i == total_len:
                lrl = [param_group['lr'] for param_group in self.optimizer.param_groups]
                lr = sum(lrl) / len(lrl)
                self.fabric.call('on_train', epoch, update_idx, update_len, loss, lr, self.duration.compute().item(),
                                 self.data_duration.compute().item(), self.sample_size)

            self.scheduler.step_update(num_updates=num_updates, metric=self.losses.compute())
            data_start = perf_counter()

        if hasattr(self.optimizer, 'sync_lookahead'):
            self.optimizer.sync_lookahead()

        return {'loss': self.losses.compute().item()}

    def val_iterate(self, model, data, criterion):
        x, y = map(lambda a: a.to(self.device), data)
        x = x.to(memory_format=torch.channels_last)

        with self.fabric.autocast():
            prob = model(x)
            loss = criterion(prob, y)

        return loss, prob, y

    @torch.no_grad()
    def validate(self, epoch):
        self._reset_metric()
        total = len(self.val_loader) - 1

        model = self.model_ema if self.model_ema is not None else self.model
        model.eval()
        for i, data in enumerate(self.val_loader):
            loss, prob, target = self.val_iterate(model, data, self.val_criterion)
            self._update_metric(loss, prob, target)

            if i % self.logging_interval == 0 or i == total:
                self.fabric.call('on_eval', self._metrics(), epoch, i, total, self.model_ema)

        return self._metrics()
