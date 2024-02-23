import torch
import torch.nn as nn
from simple_pinn import SimplePINN, BrioAndWuPINN
import cv_mesh


# TEST: Sod shock tube gamma = 1.4
def ideal_equation_of_state(mhd_state_variables, gamma=1.4):
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
        E - rho * (v_x**2 + v_y**2 + v_z**2) / 2.0 - (B_x**2 + B_y**2 + B_z**2) / 2.0
    )


def ideal_equation_of_state_for_E(mhd_state_variables, p, gamma=1.4):
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
        + rho * (v_x**2 + v_y**2 + v_z**2) / 2.0
        + (B_x**2 + B_y**2 + B_z**2) / 2.0
    )


# construct F_t, F_x, F_y, and F_z construct fluxes out of the MHD state variables and a provided equation of state function;
# right now only constructs F_t and F_x for the 1D (in space) problem;
# these fluxes are taken from Gardiner and Stone (2005), https://arxiv.org/pdf/astro-ph/0501557.pdf, where F_t, F_x, and F_y are equations 7 and 8;
# you can also just expand out the ideal MHD equations in conservative form (equations 1--4)
def construct_fluxes(
    mhd_state_variables, equation_of_state, return_entropy_fluxes=False
):
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
    if not return_entropy_fluxes:
        return F_t, F_x

    # now construct the entropy fluxes
    E = mhd_state_variables[..., 7]
    e = (
        E - rho * (v_x**2 + v_y**2 + v_z**2) / 2.0 - (B_x**2 + B_y**2 + B_z**2) / 2.0
    ) / rho
    eps = torch.tensor(1.0e-9)
    gamma = 1.4
    s = torch.log(torch.max(e, eps) ** (1.0 / (gamma - 1.0)) / torch.max(rho, eps))
    eta = -rho * s
    q_x = -rho * v_x * s
    q_y = -rho * v_y * s
    q_z = -rho * v_z * s
    F_t_entropy = eta
    F_x_entropy = q_x

    return F_t, F_x, F_t_entropy, F_x_entropy


def replace_fluxes_with_ICs_and_BCs(
    F_t, F_x, eos, eval_points, t_domain, x_domain, flux_str
):
    # construct the initial condition MHD state variables corresponding to the Brio and Wu problem
    # at all points supplied
    rho_IC = torch.where(eval_points[..., 1] < 0.0, 1.0, 0.125)
    v_x_IC = torch.zeros_like(eval_points[..., 1])
    v_y_IC = torch.zeros_like(eval_points[..., 1])
    v_z_IC = torch.zeros_like(eval_points[..., 1])
    B_x_IC = 0.75 * torch.ones_like(eval_points[..., 1])
    B_y_IC = torch.where(eval_points[..., 1] < 0.0, 1.0, -1.0)
    B_z_IC = torch.zeros_like(eval_points[..., 1])

    # note that Brio and Wu specify an initial pressure, not an initial total energy...
    p_IC = torch.where(eval_points[..., 1] < 0.0, 1.0, 0.1)
    IC_mhd_state_variables = torch.stack(
        [rho_IC, v_x_IC, v_y_IC, v_z_IC, B_x_IC, B_y_IC, B_z_IC, p_IC],
        dim=len(rho_IC.shape),
    )
    # E = ideal_equation_of_state_for_E(IC_mhd_state_variables, p_IC)
    # TEST: gamma = 1.4 for Sod shock tube
    E = ideal_equation_of_state_for_E(IC_mhd_state_variables, p_IC, gamma=1.4)
    IC_mhd_state_variables = torch.cat(
        [IC_mhd_state_variables[..., :-1], E.unsqueeze(-1)], dim=-1
    )
    # turn these into fluxes
    F_t_IC, F_x_IC = construct_fluxes(IC_mhd_state_variables, eos)

    # do the same for the (Dirichlet) BCs
    IC_mhd_state_variables_left = IC_mhd_state_variables[:, 0, ...].unsqueeze(1)
    IC_mhd_state_variables_right = IC_mhd_state_variables[:, -1, ...].unsqueeze(1)
    IC_mhd_state_variables_left = IC_mhd_state_variables_left.expand(
        IC_mhd_state_variables.shape
    )
    IC_mhd_state_variables_right = IC_mhd_state_variables_right.expand(
        IC_mhd_state_variables.shape
    )

    F_t_BC_left, F_x_BC_left = construct_fluxes(IC_mhd_state_variables_left, eos)
    F_t_BC_right, F_x_BC_right = construct_fluxes(IC_mhd_state_variables_right, eos)

    if flux_str == "F_t":
        F_t = torch.where(
            torch.isclose(eval_points[..., 0], torch.tensor(t_domain[0])).unsqueeze(-1),
            F_t_IC,
            F_t,
        )
        F_t = torch.where(
            torch.isclose(eval_points[..., 1], torch.tensor(x_domain[0])).unsqueeze(-1),
            F_t_BC_left,
            F_t,
        )
        F_t = torch.where(
            torch.isclose(eval_points[..., 1], torch.tensor(x_domain[1])).unsqueeze(-1),
            F_t_BC_right,
            F_t,
        )
        return F_t
    elif flux_str == "F_x":
        F_x = torch.where(
            torch.isclose(eval_points[..., 0], torch.tensor(t_domain[0])).unsqueeze(-1),
            F_x_IC,
            F_x,
        )
        F_x = torch.where(
            torch.isclose(eval_points[..., 1], torch.tensor(x_domain[0])).unsqueeze(-1),
            F_x_BC_left,
            F_x,
        )
        F_x = torch.where(
            torch.isclose(eval_points[..., 1], torch.tensor(x_domain[1])).unsqueeze(-1),
            F_x_BC_right,
            F_x,
        )
        return F_x
    else:
        raise ValueError("flux_str must be either 'F_t' or 'F_x'")


def sod_ICs(eval_points):
    # construct the initial condition MHD state variables corresponding to the Sod shock tube problem
    # at all points supplied
    rho_IC = torch.where(eval_points[..., 1] < 0.0, 1.0, 0.125)
    v_x_IC = torch.zeros_like(eval_points[..., 1])
    v_y_IC = torch.zeros_like(eval_points[..., 1])
    v_z_IC = torch.zeros_like(eval_points[..., 1])
    B_x_IC = torch.zeros_like(eval_points[..., 1])
    B_y_IC = torch.zeros_like(eval_points[..., 1])
    B_z_IC = torch.zeros_like(eval_points[..., 1])

    # note that Brio and Wu specify an initial pressure, not an initial total energy...
    p_IC = torch.where(eval_points[..., 1] < 0.0, 1.0, 0.1)
    IC_mhd_state_variables = torch.stack(
        [rho_IC, v_x_IC, v_y_IC, v_z_IC, B_x_IC, B_y_IC, B_z_IC, p_IC],
        dim=len(rho_IC.shape),
    )
    # E = ideal_equation_of_state_for_E(IC_mhd_state_variables, p_IC)
    # TEST: gamma = 1.4 for Sod shock tube
    E = ideal_equation_of_state_for_E(IC_mhd_state_variables, p_IC, gamma=1.4)
    IC_mhd_state_variables = torch.cat(
        [IC_mhd_state_variables[..., :-1], E.unsqueeze(-1)], dim=-1
    )
    return IC_mhd_state_variables


def replace_fluxes_with_ICs_and_BCs_for_Sod(
    F_t, F_x, eos, eval_points, t_domain, x_domain, flux_str
):
    # construct the initial conditions for the Sod shock tube problem
    rho_IC = torch.where(eval_points[..., 1] < 0.0, 1.0, 0.125)
    v_x_IC = torch.zeros_like(eval_points[..., 1])
    v_y_IC = torch.zeros_like(eval_points[..., 1])
    v_z_IC = torch.zeros_like(eval_points[..., 1])
    B_x_IC = torch.zeros_like(eval_points[..., 1])
    B_y_IC = torch.zeros_like(eval_points[..., 1])
    B_z_IC = torch.zeros_like(eval_points[..., 1])

    def eos_sod(mhd_state_variables):
        return ideal_equation_of_state(mhd_state_variables, gamma=1.4)

    # note that Brio and Wu specify an initial pressure, not an initial total energy...
    p_IC = torch.where(eval_points[..., 1] < 0.0, 1.0, 0.1)
    IC_mhd_state_variables = torch.stack(
        [rho_IC, v_x_IC, v_y_IC, v_z_IC, B_x_IC, B_y_IC, B_z_IC, p_IC],
        dim=len(rho_IC.shape),
    )
    # gamma = 1.4 for Sod shock tube
    E = ideal_equation_of_state_for_E(IC_mhd_state_variables, p_IC, gamma=1.4)
    IC_mhd_state_variables = torch.cat(
        [IC_mhd_state_variables[..., :-1], E.unsqueeze(-1)], dim=-1
    )
    breakpoint()
    # turn these into fluxes
    F_t_IC, F_x_IC = construct_fluxes(IC_mhd_state_variables, eos_sod)

    # do the same for the (Dirichlet) BCs
    IC_mhd_state_variables_left = IC_mhd_state_variables[:, 0, ...].unsqueeze(1)
    IC_mhd_state_variables_right = IC_mhd_state_variables[:, -1, ...].unsqueeze(1)
    IC_mhd_state_variables_left = IC_mhd_state_variables_left.expand(
        IC_mhd_state_variables.shape
    )
    IC_mhd_state_variables_right = IC_mhd_state_variables_right.expand(
        IC_mhd_state_variables.shape
    )

    F_t_BC_left, F_x_BC_left = construct_fluxes(IC_mhd_state_variables_left, eos_sod)
    F_t_BC_right, F_x_BC_right = construct_fluxes(IC_mhd_state_variables_right, eos_sod)

    if flux_str == "F_t":
        F_t = torch.where(
            torch.isclose(eval_points[..., 0], torch.tensor(t_domain[0])).unsqueeze(-1),
            F_t_IC,
            F_t,
        )
        F_t = torch.where(
            torch.isclose(eval_points[..., 1], torch.tensor(x_domain[0])).unsqueeze(-1),
            F_t_BC_left,
            F_t,
        )
        F_t = torch.where(
            torch.isclose(eval_points[..., 1], torch.tensor(x_domain[1])).unsqueeze(-1),
            F_t_BC_right,
            F_t,
        )
        return F_t
    elif flux_str == "F_x":
        F_x = torch.where(
            torch.isclose(eval_points[..., 0], torch.tensor(t_domain[0])).unsqueeze(-1),
            F_x_IC,
            F_x,
        )
        F_x = torch.where(
            torch.isclose(eval_points[..., 1], torch.tensor(x_domain[0])).unsqueeze(-1),
            F_x_BC_left,
            F_x,
        )
        F_x = torch.where(
            torch.isclose(eval_points[..., 1], torch.tensor(x_domain[1])).unsqueeze(-1),
            F_x_BC_right,
            F_x,
        )
        return F_x
    else:
        raise ValueError("flux_str must be either 'F_t' or 'F_x'")


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
    model, eos, Nt=101, Nx=101, t_p=None, x_p=None, device="cpu"
):
    if x_p is None:
        x = torch.linspace(-1.0, 1.0, Nx).to(device)
    else:
        x = x_p
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
    ic_loss = torch.sum(
        torch.square(mhd_state_variables_with_p - IC_mhd_state_variables)
    )

    # compute the collocation point-based boundary condition residual loss function
    IC_mhd_state_variables_left = IC_mhd_state_variables[0, :]
    IC_mhd_state_variables_right = IC_mhd_state_variables[-1, :]

    if t_p is None:
        t = torch.linspace(0.0, 0.2, Nt).to(device)
    else:
        t = t_p
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
    loss_left = torch.sum(
        torch.square(mhd_state_variables_with_p_left - IC_mhd_state_variables_left)
    )
    loss_right = torch.sum(
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


def cv_based_PDE_residual_loss(model, eos, mesh):
    (
        F_t_eval_points,
        F_x_eval_points,
        F_t_quad_weights,
        F_x_quad_weights,
    ) = mesh.get_training_eval_points_and_weights()

    # F_t_eval_points and F_x_eval_points are structured like
    # (time cell i, space cell j, quadrature point k, time or space coordinate 0 or 1),
    # thus we can supply them both to the neural network to get the mhd_state_variables at all quadrature points
    F_t_eval_points_mhd_state_variables = model(F_t_eval_points)
    F_x_eval_points_mhd_state_variables = model(F_x_eval_points)

    # get the fluxes corresponding to the state variables at these points
    # F_t_eval_points_fluxes, _ = construct_fluxes(
    #     F_t_eval_points_mhd_state_variables, eos
    # )
    # _, F_x_eval_points_fluxes = construct_fluxes(
    #     F_x_eval_points_mhd_state_variables, eos
    # )

    F_t_eval_points_fluxes, _, F_t_entropy, _ = construct_fluxes(
        F_t_eval_points_mhd_state_variables,
        eos,
        return_entropy_fluxes=True,
    )
    _, F_x_eval_points_fluxes, _, F_x_entropy = construct_fluxes(
        F_x_eval_points_mhd_state_variables,
        eos,
        return_entropy_fluxes=True,
    )

    # replace the "edges" of the evaluated fluxes with the initial condition and boundary conditions
    replaced_F_t_eval_points_fluxes = replace_fluxes_with_ICs_and_BCs_for_Sod(
        F_t_eval_points_fluxes,
        F_x_eval_points_fluxes,
        eos,
        F_t_eval_points,
        mesh.t_domain,
        mesh.x_domain,
        "F_t",
    )
    # print("Are the input and output fluxes the same?")
    # print(torch.allclose(F_t_eval_points_fluxes, replaced_F_t_eval_points_fluxes))
    F_t_eval_points_fluxes = replaced_F_t_eval_points_fluxes
    replaced_F_x_eval_points_fluxes = replace_fluxes_with_ICs_and_BCs_for_Sod(
        F_t_eval_points_fluxes,
        F_x_eval_points_fluxes,
        eos,
        F_x_eval_points,
        mesh.t_domain,
        mesh.x_domain,
        "F_x",
    )
    # print(torch.allclose(F_x_eval_points_fluxes, replaced_F_x_eval_points_fluxes))
    F_x_eval_points_fluxes = replaced_F_x_eval_points_fluxes

    # the quad weights are already organized on a per-cell basis and normalized to one, i.e., they are structured like
    # (time cell i, space cell j, quadrature points for the "right" or "top" face of the cell + quadrature points for the "left" or "bottom" face of the cell)
    # so for each cell, we need to compute the dot product of the fluxes with the quad weights, that is
    # F_t_integrated_{ijkl} = \sum_{k} F_t_eval_points_fluxes{ijkl} * w_{ijk}, which we can write directly in Einstein notation
    F_t_integrated = torch.einsum(
        "ijkl,ijk->ijl", F_t_eval_points_fluxes, F_t_quad_weights
    )
    F_x_integrated = torch.einsum(
        "ijkl,ijk->ijl", F_x_eval_points_fluxes, F_x_quad_weights
    )

    F_t_entropy_integrated = torch.einsum("ijk,ijk->ij", F_t_entropy, F_t_quad_weights)
    F_x_entropy_integrated = torch.einsum("ijk,ijk->ij", F_x_entropy, F_x_quad_weights)

    # now we can compute the PDE residual loss in the control volume formalism
    pde_loss_structure = torch.square((F_t_integrated + F_x_integrated))
    entropy_loss_structure = torch.square(
        torch.max((F_t_entropy_integrated + F_x_entropy_integrated), torch.tensor(0))
    )
    pde_loss = torch.sum(pde_loss_structure)
    entropy_loss = torch.sum(entropy_loss_structure)

    return pde_loss, pde_loss_structure, entropy_loss, entropy_loss_structure


def cv_based_entropy_residual_loss(model, eos, mesh):
    (
        F_t_eval_points,
        F_x_eval_points,
        F_t_quad_weights,
        F_x_quad_weights,
    ) = mesh.get_training_eval_points_and_weights()

    # F_t_eval_points and F_x_eval_points are structured like
    # (time cell i, space cell j, quadrature point k, time or space coordinate 0 or 1),
    # thus we can supply them both to the neural network to get the mhd_state_variables at all quadrature points
    F_t_eval_points_mhd_state_variables = model(F_t_eval_points)
    F_x_eval_points_mhd_state_variables = model(F_x_eval_points)

    # get the fluxes corresponding to the state variables at these points
    F_t_eval_points_fluxes, _ = construct_fluxes(
        F_t_eval_points_mhd_state_variables, eos
    )
    _, F_x_eval_points_fluxes = construct_fluxes(
        F_x_eval_points_mhd_state_variables, eos
    )

    # replace the "edges" of the evaluated fluxes with the initial condition and boundary conditions
    F_t_eval_points_fluxes = replace_fluxes_with_ICs_and_BCs_for_Sod(
        F_t_eval_points_fluxes,
        F_x_eval_points_fluxes,
        eos,
        F_t_eval_points,
        mesh.t_domain,
        mesh.x_domain,
        "F_t",
    )
    F_x_eval_points_fluxes = replace_fluxes_with_ICs_and_BCs_for_Sod(
        F_t_eval_points_fluxes,
        F_x_eval_points_fluxes,
        eos,
        F_x_eval_points,
        mesh.t_domain,
        mesh.x_domain,
        "F_x",
    )

    # the quad weights are already organized on a per-cell basis and normalized to one, i.e., they are structured like
    # (time cell i, space cell j, quadrature points for the "right" or "top" face of the cell + quadrature points for the "left" or "bottom" face of the cell)
    # so for each cell, we need to compute the dot product of the fluxes with the quad weights, that is
    # F_t_integrated_{ijkl} = \sum_{k} F_t_eval_points_fluxes{ijkl} * w_{ijk}, which we can write directly in Einstein notation
    F_t_integrated = torch.einsum(
        "ijkl,ijk->ijl", F_t_eval_points_fluxes, F_t_quad_weights
    )
    F_x_integrated = torch.einsum(
        "ijkl,ijk->ijl", F_x_eval_points_fluxes, F_x_quad_weights
    )

    # now we can compute the PDE residual loss in the control volume formalism
    loss_structure = torch.square((F_t_integrated + F_x_integrated))
    loss = torch.sum(loss_structure)

    return loss, loss_structure


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

    t_domain = [0.0, 0.2]
    x_domain = [-1.0, 1.0]
    mesh = cv_mesh.CVMesh(t_domain, x_domain, Nt, Nx, None, quad_pts=(3, 3))

    outputs = mhd_state_variables_nn(inputs)
    eos = ideal_equation_of_state

    cv_based_PDE_residual_loss(mhd_state_variables_nn, eos, mesh)

    F_t, F_x = construct_fluxes(outputs, eos)

    pde_loss, _ = collocation_based_PDE_residual_loss(inputs, F_t, F_x)
    mnpl_loss = monopole_loss(inputs, outputs)
    ic_loss, bc_loss = collocation_based_brio_and_wu_IC_BC_residual_loss(
        mhd_state_variables_nn, ideal_equation_of_state, Nx=Nx, Nt=Nt
    )
    ic_loss, bc_loss = collocation_based_brio_and_wu_IC_BC_residual_loss_comparing_Es(
        mhd_state_variables_nn, ideal_equation_of_state, Nx=Nx, Nt=Nt
    )
