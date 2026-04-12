import numpy as np

class PistonDynamics:
    def __init__(self, mass=4.5, diameter=0.105, length=12.245, P0=5e5, gamma=1.4):
        self.mass = mass
        self.diameter = diameter
        self.length = length
        self.P0 = P0
        self.gamma = gamma
        
        self.A = np.pi * (self.diameter**2) / 4.0
        self.V0 = self.A * self.length
        
        self.x = 0.0
        self.v = 0.0
        self.P_gas = self.P0
        self.t = 0.0
        self.dt = 1e-6
        self.burst_pressure = 470e5 # 470 bar
        self.burst_time = -1.0
        
    def step(self, P_powder):
        # Force on piston.
        # User defined: F = P_powder * A.
        # But physically, the gas back-pressure opposes it, so net force should be: F = (P_powder - P_gas) * A
        F = (P_powder - self.P_gas) * self.A
        
        a = F / self.mass
        self.v += a * self.dt
        self.x += self.v * self.dt
        
        # Gas compression
        V = self.V0 - self.A * self.x
        
        # Ensure V never becomes zero or negative
        if V <= 1e-10:
            V = 1e-10
            
        self.P_gas = self.P0 * (self.V0 / V) ** self.gamma
        self.t += self.dt
        
        if self.P_gas >= self.burst_pressure and self.burst_time < 0:
            self.burst_time = self.t
            return True # Burst!
            
        return False
