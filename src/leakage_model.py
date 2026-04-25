import numpy as np

def piston_leakage(P_high, P_low, A_leak, Cd, rho):

    dP = max(P_high - P_low, 0)

    m_dot = Cd * A_leak * np.sqrt(2*rho*dP)

    return m_dot