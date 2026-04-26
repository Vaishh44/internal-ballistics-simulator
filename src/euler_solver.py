import numpy as np

def euler_step(rho, u, p, gamma, dx, dt):

    N = len(rho)

    rho_new = rho.copy()
    u_new = u.copy()
    p_new = p.copy()

    for i in range(1, N-1):

        rho_i = max(rho[i],1e-6)
        rho_ip = max(rho[i+1],1e-6)
        rho_im = max(rho[i-1],1e-6)

        u_i = u[i]
        u_ip = u[i+1]
        u_im = u[i-1]

        p_i = max(p[i],1e3)
        p_ip = max(p[i+1],1e3)
        p_im = max(p[i-1],1e3)

        # Continuity
        drho = -(rho_ip*u_ip - rho_im*u_im)/(2*dx)

        # Momentum
        du = -( (u_ip**2 + p_ip/rho_ip) - (u_im**2 + p_im/rho_im) )/(2*dx)

        # Energy (pressure form)
        dp = -(gamma*p_ip*u_ip - gamma*p_im*u_im)/(2*dx)

        rho_new[i] = max(rho_i + drho*dt,1e-6)
        u_new[i]   = np.clip(u_i + du*dt,-3000,3000)
        p_new[i]   = max(p_i + dp*dt,1e3)

    return rho_new, u_new, p_new