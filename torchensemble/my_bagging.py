"""
  In bagging-based ensemble, each base estimator is trained independently.
  In addition, sampling with replacement is conducted on the training data
  batches to encourage the diversity between different base estimators in
  the ensemble.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
from joblib import Parallel, delayed

from ._base import BaseClassifier, BaseRegressor
from ._base import torchensemble_model_doc
from .utils import io
from .utils import set_module
from .utils import operator as op
from .bagging import BaggingClassifier, BaggingRegressor


__all__ = ["MyBaggingRegressor"]


def _parallel_fit_per_epoch(
    train_loader,
    estimator,
    cur_lr,
    optimizer,
    criterion,
    idx,
    epoch,
    log_interval,
    device,
    is_classification,
):
    """
    Private function used to fit base estimators in parallel.

    WARNING: Parallelization when fitting large base estimators may cause
    out-of-memory error.
    """
    if cur_lr:
        # Parallelization corrupts the binding between optimizer and scheduler
        set_module.update_lr(optimizer, cur_lr)

    for batch_idx, elem in enumerate(train_loader):
        data, target = io.split_data_target(elem, device)
        batch_size = data[0].size(0)

        optimizer.zero_grad()
        output = estimator(*data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Print training status
        if batch_idx % log_interval == 0:
            msg = "Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d}" " | Loss: {:.5f}"
            print(msg.format(idx, epoch, batch_idx, loss))

    return estimator, optimizer, loss


class MyBaggingRegressor(BaggingRegressor):
    def __init__(
        self,
        estimator,
        n_estimators,
        estimator_args=None,
        cuda=True,
        n_jobs=None,
    ):
        super(MyBaggingRegressor, self).__init__(estimator, n_estimators, estimator_args, cuda, n_jobs)
        self.total_epochs = 0
        for _ in range(self.n_estimators):
            self.estimators_.append(self._make_estimator())

    def forward(self, *x):
        # Average over predictions from all base estimators.
        outputs = [estimator(*x) for estimator in self.estimators_]
        pred = op.average(outputs)

        return pred

    def fit(
        self,
        train_loader,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):
        self._validate_parameters(epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)

        # Instantiate a pool of base estimators, optimizers, and schedulers.
        estimators = []
        for i in range(self.n_estimators):
            estimators.append(self.estimators_[i])

        optimizers = []
        for i in range(self.n_estimators):
            optimizers.append(set_module.set_optimizer(estimators[i], self.optimizer_name, **self.optimizer_args))

        if self.use_scheduler_:
            scheduler_ = set_module.set_scheduler(optimizers[0], self.scheduler_name, **self.scheduler_args)

        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = nn.MSELoss()

        # Utils
        best_loss = float("inf")

        # Internal helper function on pesudo forward
        def _forward(estimators, *x):
            outputs = [estimator(*x) for estimator in estimators]
            pred = op.average(outputs)

            return pred

        # Turn train_loader into a list of train_loaders,
        # sampling with replacement
        train_loader = _get_bagging_dataloaders(train_loader, self.n_estimators)

        # Maintain a pool of workers
        with Parallel(n_jobs=self.n_jobs) as parallel:
            # Training loop
            for epoch in range(epochs):
                self.train()

                if self.use_scheduler_:
                    if self.scheduler_name == "ReduceLROnPlateau":
                        cur_lr = optimizers[0].param_groups[0]["lr"]
                    else:
                        cur_lr = scheduler_.get_last_lr()[0]
                else:
                    cur_lr = None

                if self.n_jobs and self.n_jobs > 1:
                    msg = "Parallelization on the training epoch: {:03d}"
                    self.logger.info(msg.format(epoch))

                rets = parallel(
                    delayed(_parallel_fit_per_epoch)(
                        dataloader,
                        estimator,
                        cur_lr,
                        optimizer,
                        self._criterion,
                        idx,
                        epoch,
                        log_interval,
                        self.device,
                        False,
                    )
                    for idx, (estimator, optimizer, dataloader) in enumerate(zip(estimators, optimizers, train_loader))
                )

                estimators, optimizers, losses = [], [], []
                for estimator, optimizer, loss in rets:
                    estimators.append(estimator)
                    optimizers.append(optimizer)
                    losses.append(loss)

                # Validation
                if test_loader:
                    self.eval()
                    with torch.no_grad():
                        val_loss = 0.0
                        for _, elem in enumerate(test_loader):
                            data, target = io.split_data_target(elem, self.device)
                            output = _forward(estimators, *data)
                            val_loss += self._criterion(output, target)
                        val_loss /= len(test_loader)

                        if val_loss < best_loss:
                            best_loss = val_loss
                            self.estimators_ = nn.ModuleList()
                            self.estimators_.extend(estimators)
                            if save_model:
                                io.save(self, save_dir, self.logger)

                        msg = "Epoch: {:03d} | Validation Loss:" " {:.5f} | Historical Best: {:.5f}"
                        self.logger.info(msg.format(epoch, val_loss, best_loss))
                        if self.tb_logger:
                            self.tb_logger.add_scalar("bagging/Validation_Loss", val_loss, epoch)
                # No validation
                else:
                    self.estimators_ = nn.ModuleList()
                    self.estimators_.extend(estimators)
                    if save_model:
                        io.save(self, save_dir, self.logger)

                # Update the scheduler
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)

                    if self.use_scheduler_:
                        if self.scheduler_name == "ReduceLROnPlateau":
                            if test_loader:
                                scheduler_.step(val_loss)
                            else:
                                loss = torch.mean(torch.tensor(losses))
                                scheduler_.step(loss)
                        else:
                            scheduler_.step()

        self.total_epochs += epochs

        # evaluate
        self.eval()
        losses = []

        for i in range(self.n_estimators):
            with torch.no_grad():
                loss = 0.0
                for _, elem in enumerate(train_loader[i]):
                    data, target = io.split_data_target(elem, self.device)
                    output = self.estimators_[i](*data)
                    loss += self._criterion(output, target) * len(data[0])
                loss /= len(train_loader[i])
                losses.append(loss.item())

        return {
            "epoch": self.total_epochs,
            "loss": sum(losses) / len(losses),
            "max_loss": max(losses),
            "min_loss": min(losses),
        }


def _get_bagging_dataloaders(original_dataloader, n_estimators):
    dataset = original_dataloader.dataset
    dataloaders = []
    for i in range(n_estimators):
        # sampling with replacement
        indices = torch.randint(high=len(dataset), size=(len(dataset),), dtype=torch.int64)
        sub_dataset = torch.utils.data.Subset(dataset, indices)
        dataloader = torch.utils.data.DataLoader(
            sub_dataset,
            batch_size=original_dataloader.batch_size,
            num_workers=original_dataloader.num_workers,
            collate_fn=original_dataloader.collate_fn,
            shuffle=True,
        )
        dataloaders.append(dataloader)
    return dataloaders
