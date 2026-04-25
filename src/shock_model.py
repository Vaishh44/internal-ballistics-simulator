import numpy as np

def shock_jump(P1, rho1, gamma, Mach):

    P2 = P1 * (1 + 2*gamma/(gamma+1)*(Mach**2 - 1))

    rho2 = rho1 * ((gamma+1)*Mach**2) / ((gamma-1)*Mach**2 + 2)

    return P2, rho2