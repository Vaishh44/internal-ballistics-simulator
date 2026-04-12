import numpy as np

class ProjectileDynamics:
    def __init__(self, mass=0.0279, diameter=0.036, length=8.1, P0=470e5, V0=1.0, gamma=1.4):
        self.mass = mass
        self.diameter = diameter
        self.length = length
        self.P0 = P0
        self.V0 = V0
        self.gamma = gamma
        
        self.A = np.pi * (self.diameter**2) / 4.0
        
        self.x = 0.0
        self.v = 0.0
        self.a = 0.0
        self.P_gas = self.P0
        self.t = 0.0
        self.dt = 1e-6
        self.exit_time = -1.0
        
    def step(self):
        # Force on projectile
        F = self.P_gas * self.A
        
        self.a = F / self.mass
        self.v += self.a * self.dt
        self.x += self.v * self.dt
        
        # Gas expansion
        V = self.V0 + self.A * self.x
        
        self.P_gas = self.P0 * (self.V0 / V) ** self.gamma
        self.t += self.dt
        
        # Stop condition
        if self.x >= self.length and self.exit_time < 0:
            self.exit_time = self.t
            return True # Exited!
            
        return False
