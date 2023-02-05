import torch
from typing import Callable
import torch.nn


class MLP(torch.nn.Module):
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

        # self.ln1 = torch.nn.Linear(input_size, hidden_size)
        # self.ln2 = torch.nn.Linear(hidden_size, hidden_size)
        self.out = torch.nn.Linear(num_inputs, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.layers:
            self.initializer(m.weight)
            # torch.nn.init.ones_(m.weight)

    def forward(self, x):
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        # x_f = torch.flatten(x, 1)
        # x = x.view(x.shape[0], -1)
        # x = self.out(x)
        x = x.view(x.shape[0], -1)
        # actv = self.activation
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.out(x)

        return x
