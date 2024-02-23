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
        dropout_rate=0,
    ):
        """
        Initialize a SimplePINN object. Basically just a feedforward dense neural network.

        Args:
            input_size (int): The input size; e.g., 2 for a 2D (1 time and 1 space) input point, like (t, x).
            hidden_layers (list): A list of integers representing the sizes of the hidden layers.
            output_size (int): The size of the output layer. Typically how many components are in the state vector.
            activation (torch.nn.Module, optional): The activation function to use in the hidden layers. Defaults to nn.ReLU().
            use_bias_in_output_layer (bool, optional): Whether to use bias in the output layer. Defaults to True.
            dropout_rate (float, optional): The dropout rate to use in the hidden layers. Defaults to 0 (i.e., no dropout).
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
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.head = nn.Linear(
            hidden_layers[-1], output_size, bias=use_bias_in_output_layer
        )

        self.activation = activation

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
            if self.dropout_layer.p > 0:
                x = self.dropout_layer(x)
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
        upwind_only=True,
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
            upwind_only (bool, optional): Whether to enforce the Dirichlet boundary conditions only in the upwind (t = 0) direction.
                Defaults to True.
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
        self.upwind_only = upwind_only

    def overwrite_ics_and_bcs(self, input, output):
        t = input[..., 0]
        x = input[..., 1]
        # test if any of the inputs, (t, x), are on the time or space boundary, and use the initial condition state vector if so
        if self.upwind_only:
            is_ic_or_bc = torch.logical_or(
                t <= self.t_domain[0],
                torch.logical_or(x <= self.x_domain[0], x >= self.x_domain[1]),
            )
        else:
            is_ic_or_bc = torch.logical_or(
                torch.logical_or(
                    t <= self.t_domain[0], t >= self.t_domain[1]
                ),  # time boundary
                torch.logical_or(
                    x <= self.x_domain[0], x >= self.x_domain[1]
                ),  # space boundary
            )
        ic_state_vec = self.ic_state_vec_evaluation(input, self.eos)
        output = torch.where(is_ic_or_bc.unsqueeze(-1), ic_state_vec, output)
        return output

    def forward(self, input):
        output = super().forward(input)
        output = self.overwrite_ics_and_bcs(input, output)
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
        use_vector_potential=False,
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
            use_vector_potential (bool, optional): Whether to use the vector potential to calculate the magnetic field. Defaults to False.
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
        self.use_vector_potential = use_vector_potential

    def forward(self, input):
        if not self.use_vector_potential:
            output = super().forward(input)
            # plug in the constant magnetic field in the x-direction
            B_x = self.B_x * torch.ones(output.shape[:-1]).unsqueeze(-1).to(
                output.devices
            )
            output = torch.cat([output[..., :4], B_x, output[..., 5:]], dim=-1)
            return output
        else:
            state_vec = super().forward(input)
            A_x = state_vec[..., 4]
            A_y = state_vec[..., 5]
            A_z = state_vec[..., 6]
            # B_x = self.B_x * torch.ones(state_vec.shape[:-1]).unsqueeze(-1).to(
            #     state_vec.device
            # )
            B_x = A_x.unsqueeze(-1)
            B_y = -torch.autograd.grad(
                A_z,
                input,
                torch.ones_like(A_z),
                retain_graph=True,
                create_graph=True,
            )[0][..., 1].unsqueeze(-1)
            B_z = torch.autograd.grad(
                A_y,
                input,
                torch.ones_like(A_y),
                retain_graph=True,
                create_graph=True,
            )[0][..., 1].unsqueeze(-1)
            state_vec = torch.cat(
                [state_vec[..., :4], B_x, B_y, B_z, state_vec[..., 7:]], dim=-1
            )
            state_vec = self.overwrite_ics_and_bcs(input, state_vec)
            return state_vec


class SegmentationPINN(SimplePINN):
    def __init__(
        self,
        input_size,
        hidden_layers,
        problem_parameters,
        solve_for_parameters=False,
        activation=nn.ReLU(),
        dropout_rate=0,
    ):
        """
        Initialize a neural network that predicts a parameter (or vector of parameters) for a model such that the predicted
        parameter(s) is a linear (with weights summing to one) combination of a supplied set of parameters.

        Args:
            input_size (int): The input size; e.g., 2 for a 2D (1 time and 1 space) input point, like (t, x).
            hidden_layers (list): A list of integers representing the sizes of the hidden layers.
            problem_parameters (torch.Tensor): Possible parameters for the problem, organized like (number of parameter possibilities, ...).
            solve_for_parameters (bool, optional): Whether to solve for the problem parameters (as opposed to treating them as constants). Defaults to False.
            activation (torch.nn.Module, optional): The activation function to use in the hidden layers. Defaults to nn.ReLU().
            dropout_rate (float, optional): The dropout rate to use in the hidden layers. Defaults to 0 (i.e., no dropout).
        """
        super().__init__(
            input_size,
            hidden_layers,
            problem_parameters.shape[0],
            activation,
            use_bias_in_output_layer=False,
            dropout_rate=dropout_rate,
        )

        self.problem_parameters = nn.Parameter(
            problem_parameters, requires_grad=solve_for_parameters
        )

    def forward(self, x, return_p=False):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.
            return_p (bool, optional): Whether to return the probability-per-class tensor along with the prediction.
                Defaults to False.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Probability-per-class tensor (if return_p is True).
        """
        x = super().forward(x)
        p = torch.softmax(x, dim=-1)
        K = torch.matmul(p, self.problem_parameters)
        if return_p:
            return K, p
        return K


class CombinedPINN(nn.Module):
    def __init__(self, pinn, segmentation_pinn):
        super().__init__()
        self.pinn = pinn
        self.segmentation_pinn = segmentation_pinn

    def forward(self, x):
        K = self.segmentation_pinn(x)
        u = self.pinn(x)
        return torch.cat([K, u], dim=-1)
