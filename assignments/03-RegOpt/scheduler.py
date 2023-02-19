from typing import List
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, **kwargs):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        self.last_epoch = last_epoch
        self.optimizer = optimizer
        self._step_count = 0
        self.lr_lambdas = lambda epoch: (1 - (epoch / 2)) ** 1.0

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        """Step counting for each epoch

        Arguments:
            epoch (integer): number of epochs. Defaults to None.
        """
        self._step_count += 1

    def get_lr(self) -> List[float]:
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # Here's our dumb baseline implementation:
        # if self.last_epoch not in self.milestones:
        #     return [group['lr'] for group in self.optimizer.param_groups]
        # return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
        #         for group in self.optimizer.param_groups]
        if self._step_count > 0:
            return [
                param_group["lr"] * self.lr_lambdas(self._step_count)
                for param_group in self.optimizer.param_groups
            ]
        else:
            return [param_group["lr"] for param_group in self.optimizer.param_groups]
