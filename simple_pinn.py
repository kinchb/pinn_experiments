import torch
import torch.nn as nn


class SimplePINN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size,
        activation=nn.ReLU(),
        use_bias_in_output_layer=True,
    ):
        super().__init__()
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


class DirichletPINN(SimplePINN):
    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size,
        mesh,
        ic_state_vec_evaluation,
        eos,
        activation=nn.ReLU(),
        use_bias_in_output_layer=True,
    ):
        super().__init__(
            input_size,
            hidden_layers,
            output_size,
            activation,
            use_bias_in_output_layer,
        )
        self.t_domain = mesh.t_domain
        self.x_domain = mesh.x_domain
        self.ic_state_vec_evaluation = ic_state_vec_evaluation
        self.eos = eos

    def forward(self, input):
        t = input[..., 0]
        x = input[..., 1]
        is_ic_or_bc = torch.logical_or(
            t <= self.t_domain[0],
            torch.logical_or(x <= self.x_domain[0], x >= self.x_domain[1]),
        )
        output = super().forward(input)
        ic_state_vec = self.ic_state_vec_evaluation(input, self.eos)
        output = torch.where(is_ic_or_bc.unsqueeze(-1), ic_state_vec, output)
        return output


# this version forces B_x equal to some supplied constant, as in the Brio and Wu shock tube problem,
# obviating the need for the monopole loss
class BrioAndWuPINN(DirichletPINN):
    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size,
        mesh,
        ic_state_vec_evaluation,
        eos,
        B_x=0.75,
        activation=nn.ReLU(),
        use_bias_in_output_layer=True,
    ):
        super().__init__(
            input_size,
            hidden_layers,
            output_size,
            mesh,
            ic_state_vec_evaluation,
            eos,
            activation,
            use_bias_in_output_layer,
        )
        self.B_x = B_x

    def forward(self, x):
        x = super().forward(x)
        B_x = self.B_x * torch.ones(x.shape[:-1]).unsqueeze(-1).to(x.device)
        x = torch.cat([x[..., :4], B_x, x[..., 5:]], dim=-1)
        return x
