import numpy as np

def euler_step(rho, u, p, gamma, dx, dt):

    N = len(rho)

    rho_new = rho.copy()
    u_new = u.copy()
    p_new = p.copy()

    for i in range(1, N-1):

        if rho[i] <= 0:
            rho[i] = 1e-6

        drho = -(rho[i]*u[i] - rho[i-1]*u[i-1]) / dx

        du = -(u[i]**2 + p[i]/rho[i] - (u[i-1]**2 + p[i-1]/rho[i-1])) / dx

        dp = -(gamma*p[i]*u[i] - gamma*p[i-1]*u[i-1]) / dx

        rho_new[i] = max(rho[i] + drho*dt, 1e-6)
        u_new[i] = u[i] + du*dt
        p_new[i] = max(p[i] + dp*dt, 1e3)

    return rho_new, u_new, p_new