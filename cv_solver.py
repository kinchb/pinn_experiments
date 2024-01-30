import torch
import matplotlib.pyplot as plt


class CVSolver:
    def __init__(
        self,
        mesh,
        model,
        state_vec_to_fluxes,
        state_vec_to_entropy_fluxes,
        ic_state_vec_evaluation,
        analytic_soln,
        eos,
        component_names=[],
    ):
        self.mesh = mesh
        self.model = model
        self.state_vec_to_fluxes = state_vec_to_fluxes
        self.state_vec_to_entropy_fluxes = state_vec_to_entropy_fluxes
        self.ic_state_vec_evaluation = ic_state_vec_evaluation
        self.analytic_soln = analytic_soln
        self.eos = eos
        self.component_names = component_names

        self.pre_train_loss_history = []
        self.cv_pde_loss_history = []
        self.cv_entropy_loss_history = []

        self.pre_train = False

    def forward(self):
        (
            F_t_eval_points,
            F_x_eval_points,
            F_t_quad_weights,
            F_x_quad_weights,
        ) = self.mesh.get_training_eval_points_and_weights()

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

        F_t_eval_points_state_vec = self.model(F_t_eval_points)
        F_x_eval_points_state_vec = self.model(F_x_eval_points)
        F_t, _ = self.state_vec_to_fluxes(F_t_eval_points_state_vec, self.eos)
        _, F_x = self.state_vec_to_fluxes(F_x_eval_points_state_vec, self.eos)

        F_t_integrated = torch.einsum(
            "ijkl,ijk->ijl", F_t, F_t_quad_weights
        ) * self.mesh.dX.unsqueeze(-1)
        F_x_integrated = torch.einsum(
            "ijkl,ijk->ijl", F_x, F_x_quad_weights
        ) * self.mesh.dT.unsqueeze(-1)
        self.cv_pde_loss_structure = torch.square(F_t_integrated + F_x_integrated)
        self.cv_pde_loss = torch.sum(self.cv_pde_loss_structure)
        self.cv_pde_loss_history.append(self.cv_pde_loss.item())

        F_t, _ = self.state_vec_to_entropy_fluxes(F_t_eval_points_state_vec, self.eos)
        _, F_x = self.state_vec_to_entropy_fluxes(F_x_eval_points_state_vec, self.eos)

        F_t_integrated = torch.einsum(
            "ijkl,ijk->ijl", F_t, F_t_quad_weights
        ) * self.mesh.dX.unsqueeze(-1)
        F_x_integrated = torch.einsum(
            "ijkl,ijk->ijl", F_x, F_x_quad_weights
        ) * self.mesh.dT.unsqueeze(-1)
        self.cv_entropy_loss_structure = torch.square(
            torch.nn.functional.relu(F_t_integrated + F_x_integrated)
        )
        self.cv_entropy_loss = torch.sum(self.cv_entropy_loss_structure)
        self.cv_entropy_loss_history.append(self.cv_entropy_loss.item())

        return self.cv_pde_loss, self.cv_entropy_loss

    def plot_loss_history(self):
        plt.figure()
        if self.pre_train:
            plt.plot(self.pre_train_loss_history, label="Pre-Train Loss")
        else:
            plt.plot(self.cv_pde_loss_history, label="CV PDE Loss")
            plt.plot(self.cv_entropy_loss_history, label="CV Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss History")
        plt.legend()
        plt.semilogy()
        plt.show()

    def plot_components(
        self,
        t_ndx,
        num_rows=1,
        num_cols=None,
        centered=False,
        with_ics=False,
        with_analytic_soln=False,
        loss_to_plot=None,
    ):
        if num_cols is None:
            num_cols = self.model.head.out_features

        t, x, inputs = self.mesh.get_eval_points(centered=centered)
        self.model.eval()
        outputs = self.model(inputs)
        self.model.train()

        t = t.to("cpu").detach().numpy()
        x = x.to("cpu").detach().numpy()
        outputs = outputs.to("cpu").detach().numpy()

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8), squeeze=False)
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
        plt.show()
