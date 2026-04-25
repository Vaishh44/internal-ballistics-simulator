import numpy as np

class GasModel:

    def __init__(self, gas_type, gamma, cp, cv, molar_mass, covolume=1e-3):

        self.gas_type = gas_type
        self.gamma = gamma
        self.cp = cp
        self.cv = cv
        self.molar_mass = molar_mass
        self.b = covolume

        self.R_universal = 8.31446261815324
        self.R_specific = self.R_universal / molar_mass

    def pressure(self, rho, T):

        denom = 1 - self.b * rho
        if denom <= 0:
            denom = 1e-6

        return rho * self.R_specific * T / denom

    def temperature(self, U, m):

        if m <= 0:
            return 300.0

        return U / (m * self.cv)

    def density(self, m, V):

        if V <= 0:
            V = 1e-8

        return m / V