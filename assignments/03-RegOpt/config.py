from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:
    batch_size = 64
    num_epochs = 7
    initial_learning_rate = 0.001
    initial_weight_decay = 0.01

    lrs_kwargs = {}

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [ToTensor(), Normalize(mean=[0.485, 0.456, 0.4], std=[0.229, 0.224, 0.2])]
    )
