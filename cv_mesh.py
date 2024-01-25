import scipy.special as sp
import torch
import matplotlib.pyplot as plt


class CVMesh:
    def __init__(
        self, t_domain, x_domain, Nt, Nx, quad_dict, quad_pts=None, requires_grad=False
    ):
        """
        Initializes a CV_Mesh object.

        Class generates a uniformly spaced Cartesian control volume mesh in two dimensions (t and x).
        The mesh spans from t_domain[0] to t_domain[1] and from x_domain[0] to x_domain[1].
        The mesh is composed of Nt-1 cells in the t direction and Nx-1 cells in the x direction.
        Quad_dict is assumed to contain the quadrature weights and nodes for the t and x directions,
        where quad_dict["t"] contains two tuples, (t_nodes, t_weights), and similarly for quad[x"].
        These tuples are expected to be PyTorch tensors, but ~will~ *should* be converted if they are not (TODO: this).
        The quadrature nodes are assumed to lie in the interval [-1, 1] and the weights are assumed to sum to 2.
        If quad_pts is not None, then it is assumed to be a tuple corresponding to the number of quadrature points
        desired in the (t, x) directions, and the quadrature weights and nodes are computed using the Gauss-Legendre
        quadrature rule instead of using the supplied dictionary.

        Args:
            t_domain (tuple): A tuple representing the domain of the t-coordinate.
            x_domain (tuple): A tuple representing the domain of the x-coordinate.
            Nt (int): The number of grid points in the t-direction.
            Nx (int): The number of grid points in the x-direction.
            quad_dict (dict): A dictionary containing the quadrature points and weights, as PyTorch tensors.
            quad_pts (tuple, optional): A tuple representing the number of quadrature points in the t and x directions. Defaults to None.
            requires_grad (bool, optional): Whether the evaluation points require gradients. Defaults to False.
        """
        self.t_domain = t_domain
        self.x_domain = x_domain
        self.Nt = Nt
        self.Nx = Nx
        self.quad_dict = quad_dict
        if quad_pts:
            self.quad_dict = {}
            if isinstance(quad_pts, tuple) and len(quad_pts) == 2:
                num_quad_pts_t = quad_pts[0]
                num_quad_pts_x = quad_pts[1]
                t_gl = sp.roots_legendre(num_quad_pts_t)
                self.quad_dict["t"] = (torch.tensor(t_gl[0]), torch.tensor(t_gl[1]))
                x_gl = sp.roots_legendre(num_quad_pts_x)
                self.quad_dict["x"] = (torch.tensor(x_gl[0]), torch.tensor(x_gl[1]))
        num_quad_pts_t = len(self.quad_dict["t"][0])
        num_quad_pts_x = len(self.quad_dict["x"][0])
        self.t = torch.linspace(t_domain[0], t_domain[1], Nt)
        self.x = torch.linspace(x_domain[0], x_domain[1], Nx)
        self.T, self.X = torch.meshgrid(self.t, self.x, indexing="ij")
        # T and X contain the coordinates the Nt and Nx grid points, spanning the whole t and x domain;
        # they are of shape (Nt, Nx); we can consider these the coordinates of the cell *boundaries*, and
        # the cell centers are the midpoints between these coordinates
        self.T_c = (self.T[:-1, :-1] + self.T[1:, :-1]) / 2
        self.X_c = (self.X[:-1, :-1] + self.X[:-1, 1:]) / 2
        # compute the width of each cell in the t and x directions
        self.dT = self.T[1:, :-1] - self.T[:-1, :-1]
        self.dX = self.X[:-1, 1:] - self.X[:-1, :-1]
        # compute the points in the t and x directions where we will evaluate the functions F_t and F_x
        self.F_t_eval_points = torch.zeros((Nt - 1, Nx - 1, 2 * num_quad_pts_t, 2))
        self.F_x_eval_points = torch.zeros((Nt - 1, Nx - 1, 2 * num_quad_pts_x, 2))
        self.F_t_quad_weights = torch.zeros((Nt - 1, Nx - 1, 2 * num_quad_pts_t))
        self.F_x_quad_weights = torch.zeros((Nt - 1, Nx - 1, 2 * num_quad_pts_x))
        #
        #         (X_c + dX/2, T_c)
        #     +-----------*----------+
        #     |                      |
        #     |                      |
        #   x |       (X_c, T_c)     * (T_c + dT/2, X_c)
        #     |                      |
        #     |                      |
        #     +----------------------+
        #                 t
        #
        for i in range(Nt - 1):
            for j in range(Nx - 1):
                # we will evaluate F_t along the "right" and "left" edges of this cell
                # these are the t values corresponding the "right" edge of the cell; note they are all the same
                F_t_eval_points_t_p = (
                    self.T_c[i, j] + (self.dT[i, j] / 2.0)
                ) * torch.ones(num_quad_pts_x)
                # similarly, these are the t values corresponding to the "left" edge of the cell
                F_t_eval_points_t_m = (
                    self.T_c[i, j] - (self.dT[i, j] / 2.0)
                ) * torch.ones(num_quad_pts_x)
                # these are the x coordinates spanning the "right" edge of the cell
                F_t_eval_points_x_p = (
                    self.X_c[i, j] + (self.dX[i, j] / 2.0) * self.quad_dict["x"][0]
                )
                # these are the x coordinates spanning the "left" edge of the cell
                F_t_eval_points_x_m = (
                    self.X_c[i, j] - (self.dX[i, j] / 2.0) * self.quad_dict["x"][0]
                )
                # concatenate the "right" and "left" edge coordinates
                F_t_eval_points_t = torch.cat(
                    (F_t_eval_points_t_p, F_t_eval_points_t_m)
                )
                F_t_eval_points_x = torch.cat(
                    (F_t_eval_points_x_p, F_t_eval_points_x_m)
                )
                # store the results
                self.F_t_eval_points[i, j, :, 0] = F_t_eval_points_t
                self.F_t_eval_points[i, j, :, 1] = F_t_eval_points_x
                # concatenate the "right" and "left" edge weights; note the "left" edge weights are negative because
                # we are integrating in the opposite direction; also we normalize the weights here
                self.F_t_quad_weights[i, j, :] = (
                    torch.cat((self.quad_dict["x"][1], -self.quad_dict["x"][1])) / 2.0
                )
                # the exact same procedure for the "top" and "bottom" edges of the cell
                F_x_eval_points_x_p = (
                    self.X_c[i, j] + (self.dX[i, j] / 2.0)
                ) * torch.ones(num_quad_pts_t)
                F_x_eval_points_x_m = (
                    self.X_c[i, j] - (self.dX[i, j] / 2.0)
                ) * torch.ones(num_quad_pts_t)
                F_x_eval_points_t_p = (
                    self.T_c[i, j] + (self.dT[i, j] / 2.0) * self.quad_dict["t"][0]
                )
                F_x_eval_points_t_m = (
                    self.T_c[i, j] - (self.dT[i, j] / 2.0) * self.quad_dict["t"][0]
                )
                F_x_eval_points_x = torch.cat(
                    (F_x_eval_points_x_p, F_x_eval_points_x_m)
                )
                F_x_eval_points_t = torch.cat(
                    (F_x_eval_points_t_p, F_x_eval_points_t_m)
                )
                self.F_x_eval_points[i, j, :, 0] = F_x_eval_points_t
                self.F_x_eval_points[i, j, :, 1] = F_x_eval_points_x
                self.F_x_quad_weights[i, j, :] = (
                    torch.cat((self.quad_dict["t"][1], -self.quad_dict["t"][1])) / 2.0
                )
        if requires_grad:
            self.F_t_eval_points.requires_grad_()
            self.F_x_eval_points.requires_grad_()

    def to(self, device="cpu"):
        """
        Moves the mesh to the specified device.

        Args:
            device (str, optional): The device to move the mesh to. Defaults to "cpu".
        """
        self.T = self.T.to(device)
        self.X = self.X.to(device)
        self.T_c = self.T_c.to(device)
        self.X_c = self.X_c.to(device)
        self.dT = self.dT.to(device)
        self.dX = self.dX.to(device)
        self.F_t_eval_points = self.F_t_eval_points.to(device)
        self.F_x_eval_points = self.F_x_eval_points.to(device)
        self.F_t_quad_weights = self.F_t_quad_weights.to(device)
        self.F_x_quad_weights = self.F_x_quad_weights.to(device)

    def get_mesh(self):
        """
        Returns all the relevant mesh variables.

        Returns:
            tuple: A tuple containing T, X, T_c, X_c, dT, dX, F_t_eval_points, F_x_eval_points, F_t_quad_weights, F_x_quad_weights.
        """
        return (
            self.T,
            self.X,
            self.T_c,
            self.X_c,
            self.dT,
            self.dX,
            self.F_t_eval_points,
            self.F_x_eval_points,
            self.F_t_quad_weights,
            self.F_x_quad_weights,
        )

    def get_training_eval_points_and_weights(self):
        """
        Returns the evaluation points and weights for the F_t and F_x.

        Returns:
            tuple: A tuple containing F_t_eval_points, F_x_eval_points, F_t_quad_weights, F_x_quad_weights.
        """
        return (
            self.F_t_eval_points,
            self.F_x_eval_points,
            self.F_t_quad_weights,
            self.F_x_quad_weights,
        )

    def get_eval_points(self, centered=True):
        """
        Get evaluation points.

        Args:
            centered (bool, optional): Whether to return centered evaluation points. Defaults to True.

        Returns:
            t (torch.Tensor): Time values.
            x (torch.Tensor): Spatial values.
            inputs (torch.Tensor): Stacked input values.
        """
        if centered:
            inputs = torch.stack([self.T_c, self.X_c], dim=len(self.T_c.shape))
            t = self.T_c[:, 0]
            x = self.X_c[0, :]
        else:
            inputs = torch.stack([self.T, self.X], dim=len(self.T.shape))
            t = self.T[:, 0]
            x = self.X[0, :]
        return t, x, inputs

    def plot(self):
        """
        Plots the mesh of points T, X.

        This method creates a scatter plot of the cell edges, cell centers, F_t evaluation points, and F_x evaluation points.
        The x-axis represents the T values and the y-axis represents the X values.

        Returns:
            None
        """
        plt.figure()
        plt.scatter(
            self.T.detach().flatten(),
            self.X.detach().flatten(),
            color="blue",
            marker="o",
            label="cell edges",
        )
        # plt.errorbar(
        #     self.T_c.detach().flatten(),
        #     self.X_c.detach().flatten(),
        #     xerr=self.dT.detach().flatten() / 2.0,
        #     yerr=self.dX.detach().flatten() / 2.0,
        #     color="black",
        #     linestyle="",
        #     capsize=3.0,
        # )
        plt.scatter(
            self.T_c.detach().flatten(),
            self.X_c.detach().flatten(),
            color="red",
            marker="o",
            label="cell centers",
        )
        F_t_eval_points_t = self.F_t_eval_points[:, :, :, 0].detach().flatten()
        F_t_eval_points_x = self.F_t_eval_points[:, :, :, 1].detach().flatten()
        F_x_eval_points_t = self.F_x_eval_points[:, :, :, 0].detach().flatten()
        F_x_eval_points_x = self.F_x_eval_points[:, :, :, 1].detach().flatten()
        plt.scatter(
            F_t_eval_points_t,
            F_t_eval_points_x,
            color="green",
            marker="x",
            label="F_t eval points",
        )
        plt.scatter(
            F_x_eval_points_t,
            F_x_eval_points_x,
            color="orange",
            marker="x",
            label="F_x eval points",
        )
        plt.xlabel("T")
        plt.ylabel("X")
        plt.legend(loc="best")
        plt.title("Mesh of Points T, X")
        plt.show()


if __name__ == "__main__":
    t_domain = [0.0, 0.2]
    x_domain = [-1.0, 1.0]
    Nt = 6
    Nx = 11
    cv_mesh = CVMesh(t_domain, x_domain, Nt, Nx, None, quad_pts=(4, 4))
