from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:

    batch_size = 64
    num_epochs = 15
    initial_learning_rate = 0.001
    initial_weight_decay = 0.01
    lambda1 = lambda epoch: (1 - (epoch / 2)) ** 1.0
    lr_lambda = [lambda1]

    lrs_kwargs = {
        "initial_weight_decay": initial_weight_decay,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "lr_lambda_l": lr_lambda,
        "initial_learning_rate": initial_learning_rate,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [
            ToTensor(),
        ]
    )
