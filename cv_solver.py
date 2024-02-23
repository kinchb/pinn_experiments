import torch
import matplotlib.pyplot as plt
from pyrecorder.recorder import Recorder
from pyrecorder.writers.gif import GIF


class CVSolver:
    def __init__(
        self,
        mesh,
        model,
        state_vec_to_fluxes,
        ic_state_vec_evaluation,
        eos,
        rhs=None,
        state_vec_to_entropy_fluxes=None,
        model_to_data_comparison=None,
        analytic_soln=None,
        use_rba=False,
        rba_learning_rate=1e-3,
        rba_decay=0.9,
        component_names=[],
    ):
        """
        Initializes a CVSolver object.

        This class computes losses on user-supplied control volumes given a user-supplied model.
        It also contains some handy helper functions for plotting and animating the model's outputs.

        Args:
            mesh (CVMesh): The mesh object representing the computational domain. Currently assumed to be 2D (1 in time, 1 in space).
            model (torch.nn.Module): The model object representing the physics model, which maps input points to state vectors.
            state_vec_to_fluxes (callable): A function that maps the state vector to fluxes, in both the t- and x-directions.
            ic_state_vec_evaluation (callable): A function that evaluates the initial condition state vector.
                This is used for pre-training only, as the supplied model is assumed to enforce its own boundary conditions.
            eos (callable): The equation of state; its inputs and outputs are arbitrary with respect to this class, so long as the
                other supplied functions are compatible with it.
            rhs (callable): The RHS of the "Gauss's law"-style expression which is ultimately what this class computes. Defaults to None (i.e., zero).
            state_vec_to_entropy_fluxes (callable, optional): A function that maps the state vector to entropy fluxes. Defaults to None.
                Required to compute to the entropy loss.
            model_to_data_comparison (callable, optional): A function that both supplies data for comparison (from the user, somehow) and transforms
                the model's outputs into the same format as the data (or vice-versa, or some combination thereof). Defaults to None.
                Required to compute the data loss.
            analytic_soln (callable, optional): A function that provides the analytic (or simply well-accepted) solution for comparison. Defaults to None.
                Required to plot the analytic solution.
            use_rba (bool, optional): Whether to use "residual-based attention" loss scaling. Defaults to False.
            rba_learning_rate (float, optional): The learning rate for the RBA weights. Defaults to 1e-3.
            rba_decay (float, optional): The decay rate for the RBA weights. Defaults to 0.9.
            component_names (list, optional): A list of component names. Defaults to []. Is used to label the components in the plots, if supplied.
        """
        self.mesh = mesh
        self.model = model
        self.state_vec_to_fluxes = state_vec_to_fluxes
        self.ic_state_vec_evaluation = ic_state_vec_evaluation
        self.eos = eos

        # set self.rhs to a lambda function that returns zero if it's None
        if rhs is None:
            self.rhs = lambda: 0
        else:
            self.rhs = rhs
        self.state_vec_to_entropy_fluxes = state_vec_to_entropy_fluxes
        self.model_to_data_comparison = model_to_data_comparison
        self.analytic_soln = analytic_soln
        self.component_names = component_names

        self.pre_train = False
        if state_vec_to_entropy_fluxes is not None:
            self.compute_entropy_loss = True
        else:
            self.compute_entropy_loss = False
        if model_to_data_comparison is not None:
            self.compute_data_loss = True
        else:
            self.compute_data_loss = False

        self.pre_train_loss_history = []
        self.cv_pde_loss_history = []
        self.cv_entropy_loss_history = []
        self.data_loss_history = []
        self.relative_l2_error_history = []

        self.use_rba = use_rba
        self.rba_learning_rate = rba_learning_rate
        self.rba_decay = rba_decay
        self.rba_weights = {"cv_pde": None, "cv_entropy": None, "data": None}

    def update_rba_weights(self, residuals, loss_type):
        """
        Updates the RBA weights based on the residuals.

        Args:
            residuals (torch.Tensor): The residuals to use for the update.
        """
        if loss_type not in ["cv_pde", "cv_entropy", "data"]:
            raise ValueError("loss_type must be one of cv_pde, cv_entropy, or data")
        detached_residuals = residuals.detach()
        if (not self.use_rba) or self.rba_weights[loss_type] is None:
            self.rba_weights[loss_type] = torch.ones_like(detached_residuals)
        else:
            self.rba_weights[loss_type] = (
                self.rba_decay * self.rba_weights[loss_type]
                + self.rba_learning_rate
                * torch.abs(detached_residuals)
                / detached_residuals.max()
            )

    def forward(self, top_k=None, batch_size=None):
        """
        Performs the forward pass of the control volume solver.

        Args:
            top_k (int, optional): If not None, only the top k losses will be used in the loss calculation. Defaults to None.

        Returns:
            If `pre_train` is True, returns the pre-training loss, which is the mean squared difference
            between the model's output and the initial conditions at the evaluation points.
            Otherwise, returns a tuple of losses, including the per-control volume PDE loss, the entropy loss
            (if `compute_entropy_loss` is True), and the data loss (if `compute_data_loss` is True).
        """
        (
            F_t_eval_points,
            F_x_eval_points,
            F_t_quad_weights,
            F_x_quad_weights,
            dT,
            dX,
        ) = self.mesh.get_training_eval_points_and_weights()
        # how to reduce the losses to a scalar... I don't think this should matter in principle, but it might in practice
        loss_aggregation_func = torch.mean

        # "pre-training" attempts to match the model's output at any input time to the initial conditions;
        # uses the same evaluation points as would be used for the full control volume PDE loss
        if self.pre_train:
            F_t_eval_points_state_vec = self.ic_state_vec_evaluation(
                F_t_eval_points, self.eos
            )
            F_x_eval_points_state_vec = self.ic_state_vec_evaluation(
                F_x_eval_points, self.eos
            )
            F_t_eval_points_model = self.model(F_t_eval_points)
            F_x_eval_points_model = self.model(F_x_eval_points)
            self.pre_train_loss = torch.mean(
                torch.square(F_t_eval_points_model - F_t_eval_points_state_vec)
            ) + torch.mean(
                torch.square(F_x_eval_points_model - F_x_eval_points_state_vec)
            )
            self.pre_train_loss_history.append(self.pre_train_loss.item())
            return self.pre_train_loss

        # calculate the per-control volume loss, which is the integral of the fluxes
        # over the control volume's surface---in other words, it's the contribution to
        # the overall loss due to satisfying the PDE which expresses the system's
        # conservation laws

        # calculate the value of the model's outputs at the evaluation points for the
        # t- and x-directed fluxes, as already computed by the mesh
        F_t_eval_points_state_vec = self.model(F_t_eval_points)
        F_x_eval_points_state_vec = self.model(F_x_eval_points)
        # turn these state vectors into fluxes
        F_t, _ = self.state_vec_to_fluxes(
            F_t_eval_points_state_vec, self.eos, F_t_eval_points
        )
        _, F_x = self.state_vec_to_fluxes(
            F_x_eval_points_state_vec, self.eos, F_x_eval_points
        )

        # these fluxes are organized like F[i, j, k, l], where i and j index the time and space location
        # of the control volume, k indexes the evaluation point on the surface of the control volume
        # (set during mesh construction according to the quadrature scheme), and l indexes the component
        # of the state vector (e.g., density, pressure, etc.);
        # the quadrature weights are organized like w[i, j, k], where i, j, and k index the time, space,
        # and evaluation point locations, respectively, just like the fluxes;
        # dT and dX are the size of the control volumes (also indexed by the same i and j);
        # thus the expressions below are:
        # \sum_{k} F_t[i, j, k, l] * w[i, j, k] * dX[i, j] ~ the integral of the t-directed fluxes over the t-directed surfaces of the control volume
        # \sum_{k} F_x[i, j, k, l] * w[i, j, k] * dT[i, j] ~ the integral of the x-directed fluxes over the x-directed surfaces of the control volume
        F_t_integrated = torch.einsum(
            "ijkl,ijk->ijl", F_t, F_t_quad_weights
        ) * dX.unsqueeze(-1)
        F_x_integrated = torch.einsum(
            "ijkl,ijk->ijl", F_x, F_x_quad_weights
        ) * dT.unsqueeze(-1)
        summed_fluxes = F_t_integrated + F_x_integrated - self.rhs()
        self.cv_pde_loss_structure = summed_fluxes
        if batch_size is not None:
            shape = self.cv_pde_loss_structure.shape
            self.cv_pde_loss_structure = self.cv_pde_loss_structure.reshape(-1)
            indices = torch.randint(
                0, self.cv_pde_loss_structure.shape[0], (batch_size,)
            ).to(self.cv_pde_loss_structure.device)
            self.cv_pde_loss_structure = (
                torch.zeros_like(self.cv_pde_loss_structure)
                .scatter_(0, indices, self.cv_pde_loss_structure[indices])
                .reshape(shape)
            )
        if top_k is not None:
            shape = self.cv_pde_loss_structure.shape
            values, indices = torch.topk(
                torch.abs(self.cv_pde_loss_structure.reshape(-1)), top_k
            )
            self.cv_pde_loss_structure = (
                torch.zeros_like(self.cv_pde_loss_structure)
                .reshape(-1)
                .scatter_(0, indices, values)
                .reshape(shape)
            )
        self.update_rba_weights(self.cv_pde_loss_structure, "cv_pde")
        self.cv_pde_loss = loss_aggregation_func(
            torch.square(self.rba_weights["cv_pde"] * self.cv_pde_loss_structure)
        )
        self.cv_pde_loss_history.append(self.cv_pde_loss.item())

        if self.compute_entropy_loss:
            # do the same as above, but for the "entropy" fluxes, as defined by the supplied function and equation of state
            F_t, _ = self.state_vec_to_entropy_fluxes(
                F_t_eval_points_state_vec, self.eos
            )
            _, F_x = self.state_vec_to_entropy_fluxes(
                F_x_eval_points_state_vec, self.eos
            )

            F_t_integrated = torch.einsum(
                "ijkl,ijk->ijl", F_t, F_t_quad_weights
            ) * dX.unsqueeze(-1)
            F_x_integrated = torch.einsum(
                "ijkl,ijk->ijl", F_x, F_x_quad_weights
            ) * dT.unsqueeze(-1)
            summed_fluxes = torch.nn.functional.relu(F_t_integrated + F_x_integrated)
            self.cv_entropy_loss_structure = summed_fluxes
            self.update_rba_weights(self.cv_entropy_loss_structure, "cv_entropy")
            self.cv_entropy_loss = loss_aggregation_func(
                torch.square(
                    self.rba_weights["cv_entropy"] * self.cv_entropy_loss_structure
                )
            )
            self.cv_entropy_loss_history.append(self.cv_entropy_loss.item())

        if self.compute_data_loss:
            # compare the model's outputs to some data, as defined by the supplied function
            model_output, data = self.model_to_data_comparison(self.model, self.eos)
            self.data_loss_structure = model_output - data
            self.update_rba_weights(self.data_loss_structure, "data")
            self.data_loss = loss_aggregation_func(
                torch.square(self.rba_weights["data"] * self.data_loss_structure)
            )
            self.data_loss_history.append(self.data_loss.item())
            # compute relative L2 error
            self.relative_l2_error = torch.norm(data - model_output, p=2) / torch.norm(
                data, p=2
            )
            self.relative_l2_error_history.append(self.relative_l2_error.item())

        losses = [self.cv_pde_loss]
        if self.compute_entropy_loss:
            losses.append(self.cv_entropy_loss)
        if self.compute_data_loss:
            losses.append(self.data_loss)
        return tuple(losses)

    def plot_loss_history(self):
        """
        Plots the loss history of the solver.

        If `pre_train` is True, it plots the pre-training loss history.
        Otherwise, it plots the CV PDE loss history and CV Entropy loss history.
        Will also plot the data loss history if `compute_data_loss` is True.
        """
        plt.figure()
        if self.pre_train:
            plt.plot(self.pre_train_loss_history, label="Pre-Train Loss")
        else:
            plt.plot(self.cv_pde_loss_history, label="CV PDE Loss")
            plt.plot(self.cv_entropy_loss_history, label="CV Entropy Loss")
        if self.compute_data_loss:
            plt.plot(self.data_loss_history, label="Data Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss History")
        plt.legend()
        plt.semilogy()
        plt.show()

    def plot_relative_l2_error_history(self):
        """
        Plots the relative L2 error history of the solver.
        """
        plt.figure()
        plt.plot(self.relative_l2_error_history, label="Relative L2 Error")
        plt.xlabel("Epoch")
        plt.ylabel("Relative L2 Error")
        plt.title("Relative L2 Error History")
        plt.legend()
        plt.semilogy()
        plt.show()

    def plot_components(self, t_ndx, **kwargs):
        """
        Plots the components of the model's outputs at a specific time index.

        Args:
            t_ndx (int): The index of the time step to plot.
            **kwargs: Additional keyword arguments for customization.

        Keyword Args:
            animating (bool): Whether the plot is for animation. Default is False.
            num_rows (int): Number of rows in the subplot grid. Default is 1.
            num_cols (int): Number of columns in the subplot grid. Default is None.
            centered (bool): Whether to center the evaluation points. Default is False.
            with_ics (bool): Whether to plot the initial conditions. Default is False.
            with_analytic_soln (bool): Whether to plot the analytic solution. Default is False.
            loss_to_plot (str): The type of loss to plot. Options are "PDE", "Entropy", or None. Default is None.

        Raises:
            ValueError: If loss_to_plot is not None, "PDE", or "Entropy".
        """
        animating = kwargs.get("animating", False)
        num_rows = kwargs.get("num_rows", 1)
        num_cols = kwargs.get("num_cols", None)
        centered = kwargs.get("centered", False)
        with_ics = kwargs.get("with_ics", False)
        with_analytic_soln = kwargs.get("with_analytic_soln", False)
        loss_to_plot = kwargs.get("loss_to_plot", None)

        if num_cols is None:
            num_cols = self.model.head.out_features

        t, x, inputs = self.mesh.get_eval_points(centered=centered)
        self.model.eval()
        outputs = self.model(inputs)
        self.model.train()

        t = t.to("cpu").detach().numpy()
        x = x.to("cpu").detach().numpy()
        outputs = outputs.to("cpu").detach().numpy()

        fig_width = 4 * num_cols
        fig_height = 3 * num_rows

        fig, axs = plt.subplots(
            num_rows, num_cols, figsize=(fig_width, fig_height), squeeze=False
        )
        fig.suptitle("Component Plots at t = {t:.2f}".format(t=t[t_ndx]))

        if len(self.component_names) == 0:
            tmp_component_names = [f"Component {i}" for i in range(outputs.shape[-1])]
            component_names = tmp_component_names
        else:
            component_names = self.component_names

        for i, component in enumerate(component_names):
            ax = axs[i // num_cols, i % num_cols]
            ax.plot(x, outputs[t_ndx, :, i])
            ax.set_xlabel("x")
            ax.set_ylabel(component)

            if with_ics:
                ic_state_vec = self.ic_state_vec_evaluation(inputs, self.eos)
                ic_state_vec = ic_state_vec.to("cpu").detach().numpy()
                ax.plot(x, ic_state_vec[t_ndx, :, i], ":")

            if with_analytic_soln:
                analytic_soln = self.analytic_soln(inputs, self.eos)
                analytic_soln = analytic_soln.to("cpu").detach().numpy()
                ax.plot(x, analytic_soln[t_ndx, :, i], "--")

            if loss_to_plot is not None:
                if loss_to_plot.lower() == "pde":
                    loss = self.cv_pde_loss_structure
                    loss = loss[t_ndx, :, i].to("cpu").detach().numpy()
                elif loss_to_plot.lower() == "entropy":
                    loss = self.cv_entropy_loss_structure
                    loss = loss[t_ndx, :, 0].to("cpu").detach().numpy()
                else:
                    raise ValueError("loss_to_plot must be pde or entropy")
                x_c = self.mesh.get_eval_points(centered=True)[1]
                x_c = x_c.to("cpu").detach().numpy()
                ax2 = ax.twinx()
                ax2.plot(x_c, loss, color="red")
                ax2.set_ylabel(f"{loss_to_plot} Loss", color="red")

        plt.tight_layout()
        if not animating:
            plt.show()

    def animate_components(self, filename, fps=10, **kwargs):
        """
        Animates the components of the model's outputs, and saves them as a GIF file.

        Parameters:
            filename (str): The name of the GIF file to be saved.
            fps (int, optional): Frames per second for the animation. Defaults to 10.
            **kwargs: Additional keyword arguments to be passed to the plot_components method.
        """
        centered = kwargs.get("centered", False)
        t, _, _ = self.mesh.get_eval_points(centered=centered)
        duration = 1.0 / fps
        print(f"duration = {duration}")
        with Recorder(GIF(filename, duration=duration)) as rec:
            for t_ndx in range(len(t)):
                print(f"t_ndx = {t_ndx}/{len(t)}")
                self.plot_components(t_ndx, animating=True, **kwargs)
                rec.record()
        rec.close()
