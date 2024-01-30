import torch
import numpy as np
import scipy


def analytic_sod_soln(eval_points, Ul, Ur, gamma):
    rhol, jl, El = Ul
    rhor, jr, Er = Ur

    ul = jl / rhol
    ur = jr / rhor
    pl = (gamma - 1.0) * (El - 0.5 * rhol * ul**2)
    pr = (gamma - 1.0) * (Er - 0.5 * rhor * ur**2)

    cl = np.sqrt(gamma * pl / rhol)
    cr = np.sqrt(gamma * pr / rhor)

    def phil(p):
        if p <= pl:
            u = ul + 2 * cl / (gamma - 1) * (
                1.0 - (p / pl) ** ((gamma - 1) / (2 * gamma))
            )
        else:
            u = ul + (
                2
                * cl
                / np.sqrt(2 * gamma * (gamma - 1))
                * (1.0 - p / pl)
                / np.sqrt(1 + p / pl * (gamma + 1) / (gamma - 1))
            )
        return u

    def phir(p):
        if p <= pr:
            u = ur - 2 * cr / (gamma - 1) * (
                1.0 - (p / pr) ** ((gamma - 1) / (2 * gamma))
            )
        else:
            u = ur - (
                2
                * cr
                / np.sqrt(2 * gamma * (gamma - 1))
                * (1.0 - p / pr)
                / np.sqrt(1 + p / pr * (gamma + 1) / (gamma - 1))
            )
        return u

    def phi(p):
        return phir(p) - phil(p)

    ps = scipy.optimize.root_scalar(phi, x0=pr, x1=pl).root
    us = phil(ps)
    rhosr = (
        (1 + (gamma + 1) / (gamma - 1) * ps / pr)
        / (ps / pr + (gamma + 1) / (gamma - 1))
        * rhor
    )
    rhosl = (ps / pl) ** (1.0 / gamma) * rhol

    s = (rhosr * us - rhor * ur) / (rhosr - rhor)

    csl = np.sqrt(gamma * ps / rhosl)
    laml = ul - cl
    lamsl = us - csl

    xtr = torch.reshape(eval_points, (-1, 2))

    u = torch.zeros(xtr.shape[0], 3)

    for i, xti in enumerate(xtr):
        if xti[0] < 1e-12:
            if xti[1] < 0:
                rho = rhol
                v = ul
                p = pl
            else:
                rho = rhor
                v = ur
                p = pr
        else:
            xi = xti[1] / xti[0]
            if xi <= laml:
                rho = rhol
                v = ul
                p = pl
            elif xi > laml and xi <= lamsl:
                v = ((gamma - 1) * ul + 2 * (cl + xi)) / (gamma + 1)
                rho = (rhol**gamma * (v - xi) ** 2 / (gamma * pl)) ** (
                    1.0 / (gamma - 1)
                )
                p = (pl / rhol**gamma) * rho**gamma
            elif xi > lamsl and xi <= us:
                rho = rhosl
                v = us
                p = ps
            elif xi > us and xi <= s:
                rho = rhosr
                v = us
                p = ps
            else:
                rho = rhor
                v = ur
                p = pr

        u[i, 0] = rho
        u[i, 1] = rho * v
        u[i, 2] = p / (gamma - 1) + 0.5 * rho * v**2

    u = u.reshape(eval_points.shape[:-1] + (3,))

    return u
