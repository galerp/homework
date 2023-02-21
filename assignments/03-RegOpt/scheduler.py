from typing import List
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    This is a custom scheduler for training of a CNN.
    This is a wrapper around the PyTorch _LRScheduler class.
    """

    def __init__(self, optimizer, last_epoch=-1, **kwargs):
        """
        Initialize custom scheduler.

        Args:
            optimizer (function): Opimizer employed.
            last_epoch (int): Last epoch. Defaults to -1.
        """
        self.last_epoch = last_epoch
        self.optimizer = optimizer
        self._step_count = 0
        self.lr_lambdas = lambda epoch: (1 - (epoch / 2)) ** 0.9

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def step(self) -> None:
        """
        Step counting for each epoch.
        """
        self._step_count += 1

    def get_lr(self) -> List[float]:
        """
        Sets learning rate.

        Returns:
            List[float]: Learning rate.
        """
        if self._step_count > 0:
            return [
                param_group["lr"] * self.lr_lambdas(self._step_count)
                for param_group in self.optimizer.param_groups
            ]
        else:
            return [param_group["lr"] for param_group in self.optimizer.param_groups]
