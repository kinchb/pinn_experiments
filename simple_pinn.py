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
        # print out the names and shapes of the tensors involved in this function
        # print("input", input.shape)
        # print("output", output.shape)
        # print("ic_state_vec", ic_state_vec.shape)
        # print("is_ic_or_bc", is_ic_or_bc.shape)
        output = torch.where(is_ic_or_bc.unsqueeze(-1), ic_state_vec, output)
        return output


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
        super().__init__(
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


# this version forces all B components to 0, and v_y and v_z to zero as well
class SodPINN(SimplePINN):
    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size,
        activation=nn.ReLU(),
        use_bias_in_output_layer=False,
    ):
        super().__init__(
            input_size,
            hidden_layers,
            output_size - 1,
            activation,
            use_bias_in_output_layer,
        )

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        x = self.head(x)
        v_y = torch.zeros(x.shape[:-1]).unsqueeze(-1).to(x.device)
        v_z = torch.zeros(x.shape[:-1]).unsqueeze(-1).to(x.device)
        B_x = torch.zeros(x.shape[:-1]).unsqueeze(-1).to(x.device)
        B_y = torch.zeros(x.shape[:-1]).unsqueeze(-1).to(x.device)
        B_z = torch.zeros(x.shape[:-1]).unsqueeze(-1).to(x.device)
        x = torch.cat(
            [x[..., :2], v_y, v_z, B_x, B_y, B_z, x[..., -1].unsqueeze(-1)], dim=-1
        )
        return x


if __name__ == "__main__":
    mhd_state_variables_nn = SodPINN(
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
