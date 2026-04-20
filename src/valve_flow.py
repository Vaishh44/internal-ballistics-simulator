import numpy as np

def mass_flow_rate(P_up, P_down, T, gamma, R, A, Cd):

    pressure_ratio = P_down / P_up

    critical_ratio = (2/(gamma+1))**(gamma/(gamma-1))

    # CHOKED FLOW
    if pressure_ratio <= critical_ratio:

        m_dot = Cd * A * P_up * np.sqrt(gamma/(R*T)) * \
            (2/(gamma+1))**((gamma+1)/(2*(gamma-1)))

    else:

        m_dot = Cd * A * P_up * np.sqrt(
            (2*gamma)/(R*T*(gamma-1)) *
            (pressure_ratio**(2/gamma) -
             pressure_ratio**((gamma+1)/gamma))
        )

    return m_dot