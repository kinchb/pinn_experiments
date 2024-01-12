import torch
import torch.nn as nn


class SimplePINN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size,
    ):
        super(SimplePINN, self).__init__()
        if len(hidden_layers) == 0:
            raise ValueError("hidden_layers must have at least one element")
        hidden_layers.insert(0, input_size)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(hidden_layers[i - 1], hidden_layers[i])
                for i in range(1, len(hidden_layers))
            ]
        )
        self.head = nn.Linear(hidden_layers[-1], output_size)

        # optional, loss weighting parameters
        self.loss_params = nn.Parameter(torch.ones(3))
        self.loss_weights = torch.softmax(self.loss_params, dim=0)

    def forward(self, x):
        self.loss_weights = torch.softmax(self.loss_params, dim=0)
        for layer in self.hidden_layers:
            x = layer(x)
            x = torch.tanh(x)
        x = self.head(x)
        return x
