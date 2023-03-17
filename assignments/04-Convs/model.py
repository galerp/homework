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

        self.conv1 = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size=5,
            padding=1,
            dilation=1,
            groups=num_channels,
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(
            num_channels,
            num_channels * 6,
            kernel_size=5,
            padding=1,
            dilation=1,
            groups=num_channels,
        )
        self.fc1 = nn.Linear(num_channels * 6 * 6 * 6, 184)
        nn.init.kaiming_normal_(self.fc1.weight)
        # self.fc2 = nn.Linear(88, 55)
        # nn.init.kaiming_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(184, num_classes)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        Arguments:
            x (torch.Tensor): The input tensor.
        Returns:
            output (torch.Tensor): The output of the model.
        """

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
