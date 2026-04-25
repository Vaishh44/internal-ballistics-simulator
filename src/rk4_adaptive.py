import numpy as np

def rk4_step(f, y, t, dt):

    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt*k1/2)
    k3 = f(t + dt/2, y + dt*k2/2)
    k4 = f(t + dt, y + dt*k3)

    return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6


def adaptive_step(f, y, t, dt):

    y1 = rk4_step(f, y, t, dt)

    y_half = rk4_step(f, y, t, dt/2)
    y2 = rk4_step(f, y_half, t + dt/2, dt/2)

    error = np.linalg.norm(y2 - y1)

    tol = 1e-6

    if error > tol:
        dt *= 0.5
    else:
        dt *= 1.2

    return y2, dt