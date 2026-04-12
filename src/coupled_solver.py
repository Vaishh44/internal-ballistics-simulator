import numpy as np

class CoupledSolver:
    def __init__(self, m_p, D_p, L_p, m_proj, D_proj, L_proj, P0_gas, gamma, burst_pressure=470e5):
        self.m_p = m_p
        self.A_p = np.pi * (D_p**2) / 4.0
        self.L_p = L_p
        
        self.m_proj = m_proj
        self.A_proj = np.pi * (D_proj**2) / 4.0
        self.L_proj = L_proj
        
        self.P0_gas = P0_gas
        self.gamma = gamma
        self.burst_pressure = burst_pressure
        
        self.V0 = self.A_p * self.L_p
        
        self.x_p = 0.0
        self.v_p = 0.0
        self.a_p = 0.0
        
        self.x_proj = 0.0
        self.v_proj = 0.0
        self.a_proj = 0.0
        
        self.P_gas = self.P0_gas
        self.burst_time = -1.0
        self.exit_time = -1.0
        
        self.t = 0.0
        self.dt = 1e-6
        
    def step(self, P_powder):
        # 1. Piston forces and motion
        F_piston = (P_powder - self.P_gas) * self.A_p
        self.a_p = F_piston / self.m_p
        self.v_p += self.a_p * self.dt
        
        # No negative drift
        self.v_p = max(0.0, self.v_p)
        self.x_p += self.v_p * self.dt
        
        # 2. Projectile forces and motion (only if valve burst)
        if self.burst_time >= 0.0:
            F_proj = self.P_gas * self.A_proj
            self.a_proj = F_proj / self.m_proj
            self.v_proj += self.a_proj * self.dt
            # No negative drift
            self.v_proj = max(0.0, self.v_proj)
            self.x_proj += self.v_proj * self.dt
            
        # 3. Update total volume
        V = self.V0 - self.A_p * self.x_p + self.A_proj * self.x_proj
        if V <= 1e-10: 
            V = 1e-10
            
        # 4. Update gas pressure
        self.P_gas = self.P0_gas * (self.V0 / V) ** self.gamma
        
        # 5. Check valve burst
        if self.P_gas >= self.burst_pressure and self.burst_time < 0.0:
            self.burst_time = self.t
            
        # 6. Check exit condition
        if self.x_proj >= self.L_proj and self.exit_time < 0.0:
            self.exit_time = self.t
            return True # Target reached, exit
            
        self.t += self.dt
        return False
