import sys
import os
import glob
import numpy as np
import torch
import scipy

ATHENA_DIR = "../athena"
sys.path.insert(0, os.path.join(ATHENA_DIR, "vis/python"))
import athena_read


class AthenaReader:
    def __init__(self, data_path, state_vec_components):
        """
        Initialize the AthenaReader object.

        Stores the data from the .athdf files in the specified directory, and sets up interpolators for the state vector components,
        so that the state vector can be evaluated at arbitrary points in time and space (supporting extrapolation).

        Args:
            data_path (str): The path to the directory containing the .athdf files.
            state_vec_components (list): A list of strings representing the components of the state vector, as they are named in the .athdf files.

        Raises:
            ValueError: If no .athdf files are found at the specified data path.
        """
        self.data_path = data_path
        self.state_vec_components = state_vec_components

        files = glob.glob(os.path.join(data_path, "*.athdf"))
        if len(files) == 0:
            raise ValueError(f"No .athdf files found at {data_path}")
        files.sort()

        self.t = np.zeros(len(files))
        for i, file in enumerate(files):
            data = athena_read.athdf(file)
            self.t[i] = data["Time"]
        self.x = athena_read.athdf(files[0])["x1v"]
        self.T, self.X = np.meshgrid(self.t, self.x, indexing="ij")

        for component in state_vec_components:
            setattr(self, component, np.zeros_like(self.T))
            for i, file in enumerate(files):
                data = athena_read.athdf(file)
                getattr(self, component)[i, :] = data[component][0, 0, :]
            interpolator = scipy.interpolate.RegularGridInterpolator(
                (self.t, self.x),
                getattr(self, component),
                bounds_error=False,
                fill_value=None,
            )
            setattr(self, component + "_interpolator", interpolator)

        self.data_eval_points = torch.stack(
            [torch.Tensor(self.T), torch.Tensor(self.X)], dim=-1
        )
        self.data_state_vec = torch.stack(
            [
                torch.Tensor(getattr(self, component))
                for component in state_vec_components
            ],
            dim=-1,
        )

    def get_data(self):
        """
        Returns the data points (in time and space) and the state vectors at those points, from Athena.

        Returns:
            tuple: A tuple containing the evaluation points and state vectors.
        """
        return self.data_eval_points, self.data_state_vec

    def __call__(self, eval_points):
        """
        Interpolates the state vector components at the given evaluation points.

        Args:
            eval_points (torch.Tensor): The evaluation points.

        Returns:
            torch.Tensor: The interpolated state vector components stacked along the last dimension.
        """
        interpolated_components = []
        for component in self.state_vec_components:
            interpolator = getattr(self, component + "_interpolator")
            interpolated_components.append(torch.tensor(interpolator(eval_points)))
        return torch.stack(interpolated_components, dim=-1)

    def set_sample_points(self, sample_points, noise_level=0):
        """
        Sets the sample points for the AthenaReader object when used as "synthetic data."
        The state vector at the sample points is computed and stored, with noise added if desired.

        Args:
            sample_points (torch.Tensor): The sample points in (t, x) at which to evaluate (interpolate) the Athena data.
            noise_level (float, optional): The relative noise level to superimpose on the state vector at the sample points. Defaults to 0.
        """
        self.sample_points = sample_points
        self.sample_state_vec = self(self.sample_points.detach().cpu())
        if noise_level > 0:
            self.sample_state_vec *= torch.ones_like(
                self.sample_state_vec
            ) + noise_level * torch.randn_like(self.sample_state_vec)
        self.sample_state_vec = self.sample_state_vec.to(sample_points.device)

    def get_sample_state_vec(self):
        """
        Returns the sample points and sample state vectors, as set in set_sample_points() above.

        Returns:
            tuple: A tuple containing the sample points and sample state vectors.
        """
        return self.sample_points, self.sample_state_vec
