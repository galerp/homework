import torch
from typing import Callable
import torch.nn


class MLP(torch.nn.Module):
    """
    MLP model using pytorch.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.activation = activation()
        self.hidden_count = hidden_count
        self.initializer = initializer
        self.layers = torch.nn.ModuleList()

        for i in range(hidden_count):
            self.layers += [torch.nn.Linear(input_size, i)]
            num_inputs = i

        self.out = torch.nn.Linear(num_inputs, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initilizes weights of layers
        Arguments:
            None

        Returns:
            None
        """
        for m in self.layers:
            self.initializer(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x (torch.Tensor): The input layers

        Returns:
            x (torch.Tensor): The output of the network.
        """

        x = x.view(x.shape[0], -1)

        for layer in self.layers:
            x = self.activation(layer(x))

        x = self.out(x)

        return x
