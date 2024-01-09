import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import scipy.io


def evalAndCompare(model, dset):
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
        # prepare the input data for sampling
        self.points_to_sample = points_to_sample
        # in principle we want this to be at either t = 0 or x = the boundaries,
        # but for now let's just randomly sample
        x_indices = torch.randint(0, len(self.x), (self.points_to_sample,))
        t_indices = torch.randint(0, len(self.t), (self.points_to_sample,))
        self.x_sample = self.x[x_indices]
        self.t_sample = self.t[t_indices]
        self.u_sample = self.exact_u[x_indices, t_indices]
        self.v_sample = self.exact_v[x_indices, t_indices]
        self.h_sample = self.exact_h[x_indices, t_indices]

    def __len__(self):
        return self.points_to_sample

    def __getitem__(self, idx):
        return (
            torch.tensor([self.x_sample[idx], self.t_sample[idx]], requires_grad=True),
            torch.tensor([self.u_sample[idx], self.v_sample[idx]]),
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
