import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    """Convolution Neural Net Model"""

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """Initialize the model.
        Arguments:
            num_channels (int): The number of channels in the input.
            num_classes (int): The number of classes in the dataset.
        Returns:
            None
        """

        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 14, 5)
        self.fc1 = nn.Linear(14 * 5 * 5, 100)
        self.fc2 = nn.Linear(100, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        Arguments:
            x (torch.Tensor): The input tensor.
        Returns:
            output (torch.Tensor): The output of the model.
        """

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output
