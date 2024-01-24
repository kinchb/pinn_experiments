import torch
import torch.nn as nn


class SimplePINN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size,
        activation=nn.ReLU(),
        use_bias_in_output_layer=False,
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
        self.head = nn.Linear(
            hidden_layers[-1], output_size, bias=use_bias_in_output_layer
        )

        self.activation = activation

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        x = self.head(x)
        return x


# this version forces B_x equal to some supplied constant, as in the Brio and Wu shock tube problem,
# obviating the need for the monopole loss
class BrioAndWuPINN(SimplePINN):
    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size,
        B_x=0.75,
        activation=nn.ReLU(),
        use_bias_in_output_layer=False,
    ):
        super(BrioAndWuPINN, self).__init__(
            input_size,
            hidden_layers,
            output_size - 1,
            activation,
            use_bias_in_output_layer,
        )
        self.B_x = B_x

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        x = self.head(x)
        B_x = self.B_x * torch.ones(x.shape[:-1]).unsqueeze(-1).to(x.device)
        x = torch.cat([x[..., :4], B_x, x[..., 4:]], dim=-1)
        return x


if __name__ == "__main__":
    mhd_state_variables_nn = BrioAndWuPINN(
        2,
        [32, 32, 32, 32, 32],
        8,
        activation=nn.Softplus(),
    )

    Nt = 101
    Nx = 301

    t = torch.linspace(0.0, 0.2, Nt, requires_grad=True)
    x = torch.linspace(-1.0, 1.0, Nx, requires_grad=True)
    T, X = torch.meshgrid(t, x, indexing="ij")
    inputs = torch.stack([T, X], dim=len(T.shape))

    mhd_state_variables = mhd_state_variables_nn(inputs)
