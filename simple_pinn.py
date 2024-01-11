import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import scipy.io


# MSE loss for complex numbers
def mse_loss(pred, target):
    # return torch.mean(torch.square(torch.abs(pred - target)))
    real_part = torch.real(pred) - torch.real(target)
    imag_part = torch.imag(pred) - torch.imag(target)
    return torch.mean(torch.square(real_part)) + torch.mean(torch.square(imag_part))


def mseLoss(preds, targets):
    pred_u = preds[:, 0]
    pred_v = preds[:, 1]
    target_u = targets[:, 0]
    target_v = targets[:, 1]
    return torch.mean(torch.square(pred_u - target_u)) + torch.mean(
        torch.square(pred_v - target_v)
    )


# calculate the partial derivatives of a complex-valued function f with respect
# to its inputs, assuming its inputs are the real-valued position and time coordinates
def calculateComplexDerivatives(f, inputs):
    df_dinputs_real = torch.autograd.grad(
        torch.real(f),
        inputs,
        grad_outputs=torch.ones_like(torch.real(f)),
        create_graph=True,
    )[0]
    df_dinputs_imag = torch.autograd.grad(
        torch.imag(f),
        inputs,
        grad_outputs=torch.ones_like(torch.imag(f)),
        create_graph=True,
    )[0]
    df_dx = torch.view_as_complex(
        torch.stack((df_dinputs_real[:, 0], df_dinputs_imag[:, 0]), dim=1)
    )
    df_dt = torch.view_as_complex(
        torch.stack((df_dinputs_real[:, 1], df_dinputs_imag[:, 1]), dim=1)
    )
    return df_dx, df_dt


def calculateOperator(outputs, inputs):
    dh_dx, dh_dt = calculateComplexDerivatives(outputs, inputs)
    dh_dxx, _ = calculateComplexDerivatives(dh_dx, inputs)
    # multiply dh_dt by the imaginary unit
    imaginary_unit = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
    f = imaginary_unit * dh_dt + 0.5 * dh_dxx + torch.square(outputs) * outputs
    return f


def calculateOperatorLoss(outputs, inputs):
    u = outputs[:, 0]
    v = outputs[:, 1]
    x = inputs[:, 0]
    t = inputs[:, 1]
    u_t = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        allow_unused=True,
        materialize_grads=True,
    )[0]
    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        allow_unused=True,
        materialize_grads=True,
    )[0]
    v_t = torch.autograd.grad(
        v,
        t,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
        allow_unused=True,
        materialize_grads=True,
    )[0]
    v_x = torch.autograd.grad(
        v,
        x,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
        allow_unused=True,
        materialize_grads=True,
    )[0]
    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        allow_unused=True,
        materialize_grads=True,
    )[0]
    v_xx = torch.autograd.grad(
        v_x,
        x,
        grad_outputs=torch.ones_like(v_x),
        create_graph=True,
        allow_unused=True,
        materialize_grads=True,
    )[0]
    f_u = u_t + 0.5 * v_xx + (torch.square(u) + torch.square(v)) * v
    f_v = v_t - 0.5 * u_xx - (torch.square(u) + torch.square(v)) * u
    loss = torch.mean(torch.square(f_u)) + torch.mean(torch.square(f_v))
    return loss


def calcBoundaryPartials(outputs, inputs):
    u = outputs[:, 0]
    v = outputs[:, 1]
    x = inputs[:, 0]
    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        allow_unused=True,
        materialize_grads=True,
    )[0]
    v_x = torch.autograd.grad(
        v,
        x,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
        allow_unused=True,
        materialize_grads=True,
    )[0]
    return torch.stack((u_x, v_x), dim=1)


def calculateBoundaryLoss(left_inputs, left_outputs, right_inputs, right_outputs):
    bndLoss = mseLoss(left_outputs, right_outputs)
    left_h_x = calcBoundaryPartials(left_outputs, left_inputs)
    right_h_x = calcBoundaryPartials(right_outputs, right_inputs)
    bndDerivLoss = mseLoss(left_h_x, right_h_x)
    loss = bndLoss + bndDerivLoss
    return loss


def evalAndCompare(model, dset):
    model.to("cpu")
    model.eval()
    with torch.no_grad():
        x_grid, t_grid = torch.meshgrid(dset.x, dset.t, indexing="ij")
        # combine x_grid and t_grid into a single N x 2 tensor,
        # where N is the total number of points in the grid
        grid = torch.stack((x_grid.flatten(), t_grid.flatten()), dim=1)
        # evaluate the model at each point in the grid
        output = model(grid)
        # reshape the output to match the shape of the grid
        output = output.reshape(*x_grid.shape, 2)
        output_h = torch.sqrt(
            torch.square(output[:, :, 0]) + torch.square(output[:, :, 1])
        )

    return x_grid, t_grid, output_h, dset.exact_h


class SchrodingersEqDataset(Dataset):
    def __init__(self, file_path, points_to_sample=10_000):
        self.data = scipy.io.loadmat(file_path)
        # 1D tensors containing the x and t coordinates
        self.x = torch.tensor(self.data["x"].flatten(), dtype=torch.float32)
        self.t = torch.tensor(self.data["tt"].flatten(), dtype=torch.float32)
        # 2D tensors (x, t) containing the u, v, and h values of the exact solution
        self.exact_u = torch.tensor(np.real(self.data["uu"]), dtype=torch.float32)
        self.exact_v = torch.tensor(np.imag(self.data["uu"]), dtype=torch.float32)
        self.exact_h = torch.sqrt(
            torch.square(self.exact_u) + torch.square(self.exact_v)
        )
        self.points_to_sample = points_to_sample

    def __len__(self):
        return self.points_to_sample

    def __getitem__(self, idx):
        # this is done via a Latin Hypercube sampling method in the paper,
        # but for now let's just randomly sample the collocation points in x and t
        x = np.random.uniform(self.x[0], self.x[-1])
        t = np.random.uniform(self.t[0], self.t[-1])
        return torch.tensor([x, t], requires_grad=True)

    def getRandomInitSoln(self, points_to_sample):
        # return the initial condition at a random sampling of points in x
        x_indices = torch.randint(0, len(self.x), (points_to_sample,))
        x_sample = self.x[x_indices]
        t_sample = torch.zeros_like(x_sample)
        u_sample = self.exact_u[x_indices, 0]
        v_sample = self.exact_v[x_indices, 0]
        return (
            torch.stack([x_sample, t_sample], dim=1),
            torch.stack([u_sample, v_sample], dim=1),
        )

    def getRandomBoundaryPoints(self, points_to_sample):
        # return the boundary condition at a random sampling of points in t
        t_indices = torch.randint(0, len(self.t), (points_to_sample,))
        t_sample = self.t[t_indices]
        xl_sample = torch.ones_like(t_sample) * self.x[0]
        xl_sample.requires_grad = True
        xr_sample = torch.ones_like(t_sample) * self.x[-1]
        xr_sample.requires_grad = True
        return (
            torch.stack([xl_sample, t_sample], dim=1),
            torch.stack([xr_sample, t_sample], dim=1),
        )


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

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = torch.tanh(x)
        x = self.head(x)
        return x
