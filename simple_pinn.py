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
        """
        Initialize a SimplePINN object. Basically just a feedforward dense neural network.

        Args:
            input_size (int): The input size; e.g., 2 for a 2D (1 time and 1 space) input point, like (t, x).
            hidden_layers (list): A list of integers representing the sizes of the hidden layers.
            output_size (int): The size of the output layer. Typically how many components are in the state vector.
            activation (torch.nn.Module, optional): The activation function to use in the hidden layers. Defaults to nn.ReLU().
            use_bias_in_output_layer (bool, optional): Whether to use bias in the output layer. Defaults to True.
        """
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
        """
        Initialize a slightly more advanced PINN object that enforces Dirichlet boundary conditions.

        Args:
            input_size (int): The input size; e.g., 2 for a 2D (1 time and 1 space) input point, like (t, x).
            hidden_layers (list): A list of integers representing the sizes of the hidden layers.
            output_size (int): The size of the output layer. Typically how many components are in the state vector.
            mesh (CVMesh): An instance of the Mesh class representing the mesh used in the problem.
            ic_state_vec_evaluation (callable): A function that evaluates the initial condition state vector.
            eos (callable): The equation of state; its inputs and outputs are arbitrary with respect to this class,
                so long as it's compatible with ic_state_vec_evaluation.
            activation (torch.nn.Module, optional): The activation function to use in the hidden layers. Defaults to nn.ReLU().
            use_bias_in_output_layer (bool, optional): Whether to use bias in the output layer. Defaults to True.
        """
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
        # test if any of the inputs, (t, x), are on the time or space boundary, and use the initial condition state vector if so
        is_ic_or_bc = torch.logical_or(
            t <= self.t_domain[0],
            torch.logical_or(x <= self.x_domain[0], x >= self.x_domain[1]),
        )
        output = super().forward(input)
        ic_state_vec = self.ic_state_vec_evaluation(input, self.eos)
        output = torch.where(is_ic_or_bc.unsqueeze(-1), ic_state_vec, output)
        return output


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
        """
        Initialize a slightly specialized PINN object that both enforces Dirichlet boundary conditions and outputs a
        constant magnetic field in the x-direction. Note that without this, we'd have to calculate the divergence of
        the magnetic field and constrain it to be zero (i.e., to require no monopoles).

        Args:
            input_size (int): The input size; e.g., 2 for a 2D (1 time and 1 space) input point, like (t, x).
            hidden_layers (list): A list of integers representing the sizes of the hidden layers.
            output_size (int): The size of the output layer. Typically how many components are in the state vector.
            mesh (CVMesh): An instance of the Mesh class representing the mesh used in the problem.
            ic_state_vec_evaluation (callable): A function that evaluates the initial condition state vector.
            eos (callable): The equation of state; its inputs and outputs are arbitrary with respect to this class,
                so long as it's compatible with ic_state_vec_evaluation.
            B_x (float, optional): The constant magnetic field in the x-direction. Defaults to 0.75.
            activation (torch.nn.Module, optional): The activation function to use in the hidden layers. Defaults to nn.ReLU().
            use_bias_in_output_layer (bool, optional): Whether to use bias in the output layer. Defaults to True.
        """
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
        # plug in the constant magnetic field in the x-direction
        B_x = self.B_x * torch.ones(x.shape[:-1]).unsqueeze(-1).to(x.device)
        x = torch.cat([x[..., :4], B_x, x[..., 5:]], dim=-1)
        return x
