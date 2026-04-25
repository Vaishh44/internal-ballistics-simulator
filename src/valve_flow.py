import numpy as np


def mass_flow_rate(P_up, P_down, T, gamma, R, A, Cd):

    # -----------------------------
    # SAFETY CHECKS
    # -----------------------------

    if P_up <= 0:
        return 0.0

    if T <= 0:
        return 0.0

    # If downstream pressure higher, no flow
    if P_down >= P_up:
        return 0.0

    pressure_ratio = P_down / P_up

    critical_ratio = (2/(gamma+1))**(gamma/(gamma-1))


    # -----------------------------
    # CHOKED FLOW
    # -----------------------------

    if pressure_ratio <= critical_ratio:

        m_dot = Cd * A * P_up * np.sqrt(gamma/(R*T)) * \
                (2/(gamma+1))**((gamma+1)/(2*(gamma-1)))


    # -----------------------------
    # SUBSONIC FLOW
    # -----------------------------

    else:

        term = pressure_ratio**(2/gamma) - \
               pressure_ratio**((gamma+1)/gamma)

        # Prevent sqrt negative
        if term < 0:
            term = 0

        m_dot = Cd * A * P_up * np.sqrt(
            (2*gamma)/(R*T*(gamma-1)) * term
        )

    return m_dot