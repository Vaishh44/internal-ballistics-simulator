import numpy as np

class PowderCombustion:
    def __init__(self, length=0.440, diameter=0.1646, mass_powder=1.6, f=1e6, alpha=0.8, beta=0.008, viva=2500.0):
        self.length = length
        self.diameter = diameter
        self.mass_powder = mass_powder
        self.f = f
        self.alpha = alpha
        self.beta = beta
        self.viva = viva

        # Chamber volume (m^3)
        self.volume = np.pi * (self.diameter / 2) ** 2 * self.length

        self.dt = 1e-6
        self.z = 0.0
        self.P_init = 101325.0 # 1 atm in Pa
        self.P = self.P_init
        self.t = 0.0

    def step(self):
        # dz/dt = VIVA * ψ(z) * β * P^α
        # ψ(z) = (1 - z)
        if self.z < 1.0:
            psi_z = 1.0 - self.z
            dz_dt = self.viva * psi_z * self.beta * (self.P ** self.alpha)
            self.z += dz_dt * self.dt
            
            if self.z > 1.0:
                self.z = 1.0
        
        # P = (f * z * m_powder) / V
        P_gas = (self.f * self.z * self.mass_powder) / self.volume
        
        # We clamp to P_init so that P^alpha isn't 0 at the start.
        self.P = max(self.P_init, P_gas)
        
        self.t += self.dt

    def run_simulation(self, max_time=1.0):
        time_arr = []
        p_arr = []
        z_arr = []
        
        # Run while powder is still burning and we haven't reached max_time
        while self.z < 1.0 and self.t < max_time:
            time_arr.append(self.t)
            p_arr.append(self.P)
            z_arr.append(self.z)
            self.step()
            
        plateau_time = self.t + 0.001
        while self.t < plateau_time and self.t < max_time:
            time_arr.append(self.t)
            p_arr.append(self.P)
            z_arr.append(self.z)
            self.step()
            
        return np.array(time_arr), np.array(p_arr), np.array(z_arr)
