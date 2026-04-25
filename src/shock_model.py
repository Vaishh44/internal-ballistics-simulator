import numpy as np

def shock_jump(P1, rho1, gamma, Mach):

    # Clamp Mach number for numerical stability
    Mach = max(1.0, min(Mach, 20.0))

    P2 = P1 * (1 + (2 * gamma / (gamma + 1)) * (Mach**2 - 1))

    rho2 = rho1 * ((gamma + 1) * Mach**2) / ((gamma - 1) * Mach**2 + 2)

    return P2, rho2