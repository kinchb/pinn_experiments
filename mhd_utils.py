import torch
import torch.nn as nn
from simple_pinn import SimplePINN, BrioAndWuPINN


def ideal_equation_of_state(mhd_state_variables, gamma=2.0):
    # the ideal gas equation of state is p = (gamma - 1) * (E - rho * v^2 / 2 - B^2 / 2)
    # where p is the pressure, gamma is the ratio of specific heats (adiabatic index),
    # E is the total energy density, rho is the mass density, and v and B are the velocity and magnetic field vectors, respectively
    rho = mhd_state_variables[..., 0]
    v_x = mhd_state_variables[..., 1]
    v_y = mhd_state_variables[..., 2]
    v_z = mhd_state_variables[..., 3]
    B_x = mhd_state_variables[..., 4]
    B_y = mhd_state_variables[..., 5]
    B_z = mhd_state_variables[..., 6]
    E = mhd_state_variables[..., 7]
    return (gamma - 1.0) * (
        E
        - rho * (v_x**2 + v_y**2 + v_z**2) / 2.0
        - (B_x**2 + B_y**2 + B_z**2) / 2.0
    )


def ideal_equation_of_state_for_E(mhd_state_variables, p, gamma=2.0):
    # the ideal gas equation of state is p = (gamma - 1) * (E - rho * v^2 / 2 - B^2 / 2)
    # where p is the pressure, gamma is the ratio of specific heats (adiabatic index),
    # E is the total energy density, rho is the mass density, and v and B are the velocity and magnetic field vectors, respectively
    rho = mhd_state_variables[..., 0]
    v_x = mhd_state_variables[..., 1]
    v_y = mhd_state_variables[..., 2]
    v_z = mhd_state_variables[..., 3]
    B_x = mhd_state_variables[..., 4]
    B_y = mhd_state_variables[..., 5]
    B_z = mhd_state_variables[..., 6]
    return (
        (p / (gamma - 1.0))
        - rho * (v_x**2 + v_y**2 + v_z**2) / 2.0
        - (B_x**2 + B_y**2 + B_z**2) / 2.0
    )


# construct F_t, F_x, F_y, and F_z construct fluxes out of the MHD state variables and a provided equation of state function;
# right now only constructs F_t and F_x for the 1D (in space) problem;
# these fluxes are taken from Gardiner and Stone (2005), https://arxiv.org/pdf/astro-ph/0501557.pdf, where F_t, F_x, and F_y are equations 7 and 8;
# you can also just expand out the ideal MHD equations in conservative form (equations 1--4)
def construct_fluxes(mhd_state_variables, equation_of_state):
    # extract the state variables from the flattened input
    rho = mhd_state_variables[..., 0]
    v_x = mhd_state_variables[..., 1]
    v_y = mhd_state_variables[..., 2]
    v_z = mhd_state_variables[..., 3]
    B_x = mhd_state_variables[..., 4]
    B_y = mhd_state_variables[..., 5]
    B_z = mhd_state_variables[..., 6]
    E = mhd_state_variables[..., 7]
    # arrange these into F_t;
    # the selection operations above reduced the rank of the tensors by 1, we want to stack along this lost dimension,
    # so that the flux is a tensor of shape (..., 8)
    F_t = torch.stack(
        [rho, rho * v_x, rho * v_y, rho * v_z, B_x, B_y, B_z, E], dim=len(rho.shape)
    )
    # for the spatial components of the flux, we need to compute the total pressure P_star,
    # which is the sum of the gas pressure P and the magnetic pressure B^2 / 2;
    # the gas pressure is a function of the MHD state variables and the provided equation of state
    P = equation_of_state(mhd_state_variables)
    P_star = P + (B_x**2 + B_y**2 + B_z**2) / 2.0
    # now construct F_x (restoring the lost dimension as in F_t)
    F_x = torch.stack(
        [
            rho * v_x,
            rho * v_x**2 + P_star - B_x * B_x,
            rho * v_x * v_y - B_x * B_y,
            rho * v_x * v_z - B_x * B_z,
            torch.zeros_like(rho),
            v_x * B_y - B_x * v_y,
            v_x * B_z - B_x * v_z,
            (E + P_star) * v_x - B_x * (B_x * v_x + B_y * v_y + B_z * v_z),
        ],
        dim=len(rho.shape),
    )
    # at some point we could construct F_y and F_z here as well
    return F_t, F_x


# compute the collocation point-based PDE residual loss function, i.e., equation 6 of Gardiner and Stone (2005),
# that is, just the continuity equation that the ideal MHD equations form in their conservative form
# d/dt F_t + d/dx F_x + d/dy F_y + d/dz F_z = 0;
# note this is just for the 1D (in space) problem for now
def collocation_based_PDE_residual_loss(inputs, F_t, F_x):
    dF_t_dt = torch.zeros_like(F_t)
    dF_x_dx = torch.zeros_like(F_x)
    for k in range(F_t.shape[-1]):
        F_t_kth_component = F_t[..., k]
        F_x_kth_component = F_x[..., k]
        dF_t_dt[..., k] = torch.autograd.grad(
            F_t_kth_component,
            inputs,
            torch.ones_like(F_t_kth_component),
            retain_graph=True,
            create_graph=True,
        )[0][..., 0]
        dF_x_dx[..., k] = torch.autograd.grad(
            F_x_kth_component,
            inputs,
            torch.ones_like(F_x_kth_component),
            retain_graph=True,
            create_graph=True,
        )[0][..., 1]
    loss_structure = torch.square(dF_t_dt + dF_x_dx)
    breakpoint()
    loss = torch.mean(loss_structure)
    loss_structure = loss_structure.clone().detach()
    return loss, loss_structure


def monopole_loss(inputs, mhd_state_variables):
    B_x = mhd_state_variables[..., 4]
    B_y = mhd_state_variables[..., 5]
    B_z = mhd_state_variables[..., 6]
    # get the divergence of the magnetic field everywhere
    dB_x_dx = torch.autograd.grad(
        B_x,
        inputs,
        torch.ones_like(B_x),
        retain_graph=True,
        create_graph=True,
    )[0][..., 1]
    # dB_y_dy and dB_z_dz are zero because this is a 1D spatial problem
    monopole_loss = torch.mean(torch.square(dB_x_dx))
    return monopole_loss


# compute the collocation point-based initial condition and boundary loss functions
# for the Brio and Wu shock tube problem specifically, Brio and Wu (1988)
def collocation_based_brio_and_wu_IC_BC_residual_loss(
    model, eos, Nx=101, Nt=101, device="cpu"
):
    x = torch.linspace(-1.0, 1.0, Nx).to(device)
    t = torch.zeros_like(x).to(device)
    inputs = torch.stack([t, x], dim=1)
    mhd_state_variables = model(inputs)
    # we construct a comparison tensor that describes the IC of the Brio and Wu shock tube problem
    rho_IC = torch.where(x < 0.0, 1.0, 0.125)
    v_x_IC = torch.zeros_like(x)
    v_y_IC = torch.zeros_like(x)
    v_z_IC = torch.zeros_like(x)
    B_x_IC = 0.75 * torch.ones_like(x)
    B_y_IC = torch.where(x < 0.0, 1.0, -1.0)
    B_z_IC = torch.zeros_like(x)
    # note that Brio and Wu specify an initial pressure, not an initial total energy...
    p_IC = torch.where(x < 0.0, 1.0, 0.1)
    IC_mhd_state_variables = torch.stack(
        [rho_IC, v_x_IC, v_y_IC, v_z_IC, B_x_IC, B_y_IC, B_z_IC, p_IC],
        dim=1,
    )
    # ... therefore, before constructing the loss, we need to translate the initial energy density of our model output to pressure
    # Note: we could in principle do this the other way, right? translate the Brio and Wu initial pressure to an initial energy density.
    p = eos(mhd_state_variables)
    mhd_state_variables_with_p = torch.cat(
        [mhd_state_variables[..., :-1], p.unsqueeze(1)], dim=1
    )
    # now construct the loss
    ic_loss = torch.mean(
        torch.square(mhd_state_variables_with_p - IC_mhd_state_variables)
    )

    # compute the collocation point-based boundary condition residual loss function
    IC_mhd_state_variables_left = IC_mhd_state_variables[0, :]
    IC_mhd_state_variables_right = IC_mhd_state_variables[-1, :]

    t = torch.linspace(0.0, 0.2, Nt).to(device)
    x_left = -torch.ones_like(t)
    x_right = torch.ones_like(t)
    inputs_left = torch.stack([t, x_left], dim=1)
    inputs_right = torch.stack([t, x_right], dim=1)
    mhd_state_variables_left = model(inputs_left)
    mhd_state_variables_right = model(inputs_right)
    p_left = eos(mhd_state_variables_left)
    p_right = eos(mhd_state_variables_right)
    mhd_state_variables_with_p_left = torch.cat(
        [mhd_state_variables_left[..., :-1], p_left.unsqueeze(1)], dim=1
    )
    mhd_state_variables_with_p_right = torch.cat(
        [mhd_state_variables_right[..., :-1], p_right.unsqueeze(1)], dim=1
    )
    loss_left = torch.mean(
        torch.square(mhd_state_variables_with_p_left - IC_mhd_state_variables_left)
    )
    loss_right = torch.mean(
        torch.square(mhd_state_variables_with_p_right - IC_mhd_state_variables_right)
    )
    bc_loss = 0.5 * (loss_left + loss_right)

    return ic_loss, bc_loss, IC_mhd_state_variables


# compute the collocation point-based initial condition and boundary loss functions
# for the Brio and Wu shock tube problem specifically, Brio and Wu (1988)
def collocation_based_brio_and_wu_IC_BC_residual_loss_comparing_Es(
    model, eos, Nx=101, Nt=101, device="cpu"
):
    x = torch.linspace(-1.0, 1.0, Nx).to(device)
    t = torch.zeros_like(x).to(device)
    inputs = torch.stack([t, x], dim=1)
    mhd_state_variables = model(inputs)
    # we construct a comparison tensor that describes the IC of the Brio and Wu shock tube problem
    rho_IC = torch.where(x < 0.0, 1.0, 0.125)
    v_x_IC = torch.zeros_like(x)
    v_y_IC = torch.zeros_like(x)
    v_z_IC = torch.zeros_like(x)
    B_x_IC = 0.75 * torch.ones_like(x)
    B_y_IC = torch.where(x < 0.0, 1.0, -1.0)
    B_z_IC = torch.zeros_like(x)
    # note that Brio and Wu specify an initial pressure, not an initial total energy...
    p_IC = torch.where(x < 0.0, 1.0, 0.1)
    IC_mhd_state_variables = torch.stack(
        [rho_IC, v_x_IC, v_y_IC, v_z_IC, B_x_IC, B_y_IC, B_z_IC, p_IC],
        dim=1,
    )
    E = ideal_equation_of_state_for_E(IC_mhd_state_variables, p_IC)
    IC_mhd_state_variables = torch.cat(
        [IC_mhd_state_variables[..., :-1], E.unsqueeze(1)], dim=1
    )
    # ... therefore, before constructing the loss, we need to translate the initial energy density of our model output to pressure
    # Note: we could in principle do this the other way, right? translate the Brio and Wu initial pressure to an initial energy density.
    # now construct the loss
    ic_loss = torch.mean(torch.square(mhd_state_variables - IC_mhd_state_variables))

    # compute the collocation point-based boundary condition residual loss function
    IC_mhd_state_variables_left = IC_mhd_state_variables[0, :]
    IC_mhd_state_variables_right = IC_mhd_state_variables[-1, :]

    t = torch.linspace(0.0, 0.2, Nt).to(device)
    x_left = -torch.ones_like(t)
    x_right = torch.ones_like(t)
    inputs_left = torch.stack([t, x_left], dim=1)
    inputs_right = torch.stack([t, x_right], dim=1)
    mhd_state_variables_left = model(inputs_left)
    mhd_state_variables_right = model(inputs_right)
    p_left = eos(mhd_state_variables_left)
    p_right = eos(mhd_state_variables_right)
    E_left = ideal_equation_of_state_for_E(mhd_state_variables_left, p_left)
    E_right = ideal_equation_of_state_for_E(mhd_state_variables_right, p_right)
    mhd_state_variables_left = torch.cat(
        [mhd_state_variables_left[..., :-1], E_left.unsqueeze(1)], dim=1
    )
    mhd_state_variables_right = torch.cat(
        [mhd_state_variables_right[..., :-1], E_right.unsqueeze(1)], dim=1
    )
    loss_left = torch.mean(
        torch.square(mhd_state_variables_left - IC_mhd_state_variables_left)
    )
    loss_right = torch.mean(
        torch.square(mhd_state_variables_right - IC_mhd_state_variables_right)
    )
    bc_loss = 0.5 * (loss_left + loss_right)

    return ic_loss, bc_loss


if __name__ == "__main__":
    mhd_state_variables_nn = BrioAndWuPINN(
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

    outputs = mhd_state_variables_nn(inputs)
    eos = ideal_equation_of_state
    F_t, F_x = construct_fluxes(outputs, eos)
    pde_loss, _ = collocation_based_PDE_residual_loss(inputs, F_t, F_x)
    mnpl_loss = monopole_loss(inputs, outputs)
    ic_loss, bc_loss = collocation_based_brio_and_wu_IC_BC_residual_loss(
        mhd_state_variables_nn, ideal_equation_of_state, Nx=Nx, Nt=Nt
    )
    ic_loss, bc_loss = collocation_based_brio_and_wu_IC_BC_residual_loss_comparing_Es(
        mhd_state_variables_nn, ideal_equation_of_state, Nx=Nx, Nt=Nt
    )
