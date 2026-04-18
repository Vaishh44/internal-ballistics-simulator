import math
import numpy as np

class MultiZoneSolver:
    def __init__(self, L1, D1, L2, D2, L3, D3, powder, gas, shot):
        self.dt = 5e-7  # dt <= 1e-6 as strictly requested
        self.t = 0.0
        self.powder = powder
        self.gas = gas
        self.shot = shot
        
        self.L1, self.D1 = L1, D1
        self.L2, self.D2 = L2, D2
        self.L3, self.D3 = L3, D3
        
        self.A_ch = math.pi * (D1**2) / 4.0
        self.A_pump = math.pi * (D2**2) / 4.0
        self.A_launch = math.pi * (D3**2) / 4.0
        
        # Physics tunable params
        self.eta = 0.6
        self.h_loss = 100.0
        self.k_leak = 1e-7
        self.c_damp = 2.0
        self.tau_proj = 2e-4
        
        # 1. Zone Chamber (Powder Gas)
        self.V_ch = self.A_ch * self.L1
        self.m_ch = 1e-6
        self.U_ch = 101325.0 * self.V_ch / (self.powder.polytropic_ratio - 1.0)
        self.P_ch = 101325.0
        self.cv_powder = 1500.0 
        
        # 2. Behind Piston
        self.x_p = 0.0
        self.v_p = 0.0
        self.a_p = 0.0
        
        # 3. Pump Tube Gas
        self.V_pump_0 = self.A_pump * self.L2
        self.V_pump = self.V_pump_0
        self.P_pump = self.shot.initial_gas_pressure
        self.T_pump = self.shot.initial_temp
        self.m_pump = self.P_pump * self.V_pump / (self.gas.R_specific * self.T_pump)
        self.U_pump = self.m_pump * self.gas.cv * self.T_pump
        
        # 4. Valve Interface
        self.burst_time = -1.0
        self.valve_open = False
        
        # 5. Launch Tube Gas
        dist = max(self.shot.distance_valve_to_proj, 0.001)
        self.V_launch = self.A_launch * dist
        self.T_launch = self.shot.residual_gas_temp
        self.P_launch = self.shot.residual_gas_pressure
        self.R_air = 8.314 / 0.02896
        self.cv_air = 718.0
        self.m_launch = self.P_launch * self.V_launch / (self.R_air * self.T_launch)
        self.U_launch = self.m_launch * self.cv_air * self.T_launch
        
        # 6. Projectile
        self.x_proj = 0.0
        self.v_proj = 0.0
        self.a_proj = 0.0
        self.exit_time = -1.0
        
        # Tracking states
        self.t_open = 0.0
        self.e_powder_total = 0.0
        self.e_heat_loss = 0.0
        self.e_leak_loss = 0.0
        
    def step(self):
        # 1. Gas Generation from Powder
        if self.powder.z < 1.0:
            dz_dt = self.powder.viva * self.powder.compute_shape_function(self.powder.z) * (self.P_ch ** self.powder.alpha)
            dz = dz_dt * self.dt
            if self.powder.z + dz > 1.0:
                 dz = 1.0 - self.powder.z
            self.powder.z += dz
            self.powder.dz_dt = dz / self.dt
            
            dm_pow = self.powder.powder_mass * dz
            dU_pow = self.eta * self.powder.specific_force * self.powder.powder_mass * dz
            
            self.m_ch += dm_pow
            self.U_ch += dU_pow
            self.e_powder_total += dU_pow
        else:
            self.powder.dz_dt = 0.0
            
        # 2. Chamber Thermodynamics & Heat Loss
        T_ch = self.U_ch / (self.m_ch * self.cv_powder) if self.m_ch > 0 else 300.0
        T_ch = max(200.0, T_ch)
        
        A_wall = 2 * math.pi * (self.D1/2)**2 + math.pi * self.D1 * self.L1 + math.pi * self.D2 * self.x_p
        dQ_loss = self.h_loss * A_wall * (T_ch - 300.0) * self.dt
        if dQ_loss < 0: dQ_loss = 0
        
        self.U_ch -= dQ_loss
        self.e_heat_loss += dQ_loss
        
        dm_leak = self.k_leak * self.P_ch * self.dt
        dU_leak = self.cv_powder * T_ch * dm_leak
        
        self.m_ch = max(1e-6, self.m_ch - dm_leak)
        self.U_ch = max(1e-6, self.U_ch - dU_leak)
        self.e_leak_loss += dU_leak
        
        self.P_ch = max(1000.0, (self.powder.polytropic_ratio - 1.0) * self.U_ch / self.V_ch)
        
        # 3. Piston Motion with Damping Friction
        fric_piston = self.shot.piston_friction_coeff * (self.P_ch * self.A_pump) + self.c_damp * self.v_p
        fric_piston = math.copysign(fric_piston, self.v_p) if self.v_p != 0 else fric_piston
        
        force_piston = max((self.P_ch - self.P_pump) * self.A_pump - fric_piston, 0.0)
        
        self.a_p = force_piston / self.shot.piston_mass
        self.v_p += self.a_p * self.dt
        
        if self.v_p < 0 or (self.x_p <= 0 and self.v_p < 0): 
            self.v_p = 0.0
            self.a_p = 0.0
        
        dx_p = self.v_p * self.dt
        self.x_p += dx_p
        
        dV_pump = -self.A_pump * dx_p
        self.V_ch += self.A_pump * dx_p
        
        self.V_pump = max(1e-6, self.V_pump + dV_pump)
        
        # 4. Pump Thermodynamics
        self.U_ch -= self.P_ch * (self.A_pump * dx_p)
        self.U_pump -= self.P_pump * dV_pump
        self.U_ch = max(1e-6, self.U_ch)
        self.U_pump = max(1e-6, self.U_pump)
        
        self.T_pump = self.U_pump / (self.m_pump * self.gas.cv) if self.m_pump > 0 else 300.0
        self.T_pump = max(50.0, self.T_pump)
        self.P_pump = max(1000.0, (self.gas.gamma - 1.0) * self.U_pump / self.V_pump)
        
        # 5. Valve Burst & Flow
        burst_threshold = self.shot.valve_burst_pressure * 1e5 if self.shot.valve_burst_pressure < 1e6 else self.shot.valve_burst_pressure
        if self.P_pump >= burst_threshold and self.burst_time < 0:
            self.burst_time = self.t
            
        if self.burst_time >= 0:
            self.t_open = self.t - self.burst_time
            if self.t_open >= self.shot.valve_opening_delay:
                self.valve_open = True
                
        dm_flow = 0.0
        dU_flow = 0.0
        
        if self.valve_open and self.P_pump > self.P_launch:
            # Valve smoothing opening
            A_val = self.shot.a_valve * (1.0 - math.exp(-self.t_open / max(self.shot.valve_opening_delay, 1e-6)))
            
            # Simple Bernoulli Flow as dictated by user
            rho_pump = self.m_pump / self.V_pump if self.V_pump > 0 else 1.0
            dP = max(self.P_pump - self.P_launch, 0.0)
            
            m_dot = self.shot.cd_valve * A_val * math.sqrt(2.0 * rho_pump * dP)
            dm_flow = m_dot * self.dt
            if dm_flow > self.m_pump * 0.9: dm_flow = self.m_pump * 0.9
            
            dU_flow = dm_flow * self.gas.cp * self.T_pump
            
            self.m_pump -= dm_flow
            self.m_launch += dm_flow
            self.U_pump -= dU_flow
            self.U_launch += dU_flow
            
            self.U_pump = max(1e-6, self.U_pump)
            
        # 6. Launch Thermodynamics
        self.T_launch = self.U_launch / (self.m_launch * self.gas.cv) if self.m_launch > 0 else 300.0
        self.T_launch = max(50.0, self.T_launch)
        self.P_launch = max(1000.0, (self.gas.gamma - 1.0) * self.U_launch / self.V_launch)
        
        # 7. Projectile Motion with Delay & Friction
        if self.valve_open:
            P_eff = self.P_launch * (1.0 - math.exp(-self.t_open / self.tau_proj))
            
            fric_proj = self.shot.projectile_friction_coeff * (P_eff * self.A_launch) + self.c_damp * self.v_proj
            fric_proj = math.copysign(fric_proj, self.v_proj) if self.v_proj != 0 else fric_proj
            
            force_proj = (P_eff - self.shot.residual_gas_pressure) * self.A_launch - fric_proj
            
            if force_proj > 0 or self.v_proj > 0:
                self.a_proj = force_proj / self.shot.projectile_mass
                self.v_proj += self.a_proj * self.dt
                if self.v_proj < 0: self.v_proj = 0
                
                dx_proj = self.v_proj * self.dt
                self.x_proj += dx_proj
                self.V_launch += self.A_launch * dx_proj
                self.U_launch -= self.P_launch * (self.A_launch * dx_proj)
                self.U_launch = max(1e-6, self.U_launch)
                
        # 8. Time Advance & Checks
        self.t += self.dt
        
        if math.isnan(self.P_ch) or math.isnan(self.P_pump):
            raise ValueError("NaN detected in thermodynamic states.")
            
        if self.x_proj >= self.L3 and self.exit_time < 0:
            self.exit_time = self.t
            return True
            
        if self.v_proj > 15000.0 or self.P_ch > 10000e5: 
            self.exit_time = self.t
            return True
            
        return False
        
    def run(self, progress_callback=None):
        t_arr, p1_arr, p2_arr, p3_arr = [], [], [], []
        x_p_arr, v_p_arr = [], []
        x_proj_arr, v_proj_arr, a_proj_arr = [], [], []
        ep_arr, eg_arr, kep_arr, keproj_arr, el_arr = [], [], [], [], []
        
        max_steps = int(0.1 / self.dt) # 100 ms limit
        steps = 0
        
        while steps < max_steps:
            exited = self.step()
            
            if steps % 100 == 0 or exited:
                t_arr.append(self.t)
                p1_arr.append(self.P_ch)
                p2_arr.append(self.P_pump)
                p3_arr.append(self.P_launch)
                x_p_arr.append(self.x_p)
                v_p_arr.append(self.v_p)
                x_proj_arr.append(self.x_proj)
                v_proj_arr.append(self.v_proj)
                a_proj_arr.append(self.a_proj)
                
                # Energies
                ep_arr.append(self.e_powder_total)
                eg_arr.append(self.U_ch + self.U_pump + self.U_launch)
                kep_arr.append(0.5 * self.shot.piston_mass * self.v_p**2)
                keproj_arr.append(0.5 * self.shot.projectile_mass * self.v_proj**2)
                el_arr.append(self.e_heat_loss + self.e_leak_loss)
                
            if steps % 10000 == 0 and progress_callback:
                progress_callback((self.t / 0.05) if self.t < 0.05 else 1.0)
                
            if exited:
                break
            steps += 1
            
        return {
            't': np.array(t_arr),
            'p1': np.array(p1_arr),
            'p2': np.array(p2_arr),
            'p3': np.array(p3_arr),
            'x_p': np.array(x_p_arr),
            'v_p': np.array(v_p_arr),
            'x_proj': np.array(x_proj_arr),
            'v_proj': np.array(v_proj_arr),
            'a_proj': np.array(a_proj_arr),
            'e_powder': np.array(ep_arr),
            'e_gas': np.array(eg_arr),
            'e_ke_piston': np.array(kep_arr),
            'e_ke_proj': np.array(keproj_arr),
            'e_loss': np.array(el_arr),
            'burst_time': self.burst_time,
            'exit_time': self.exit_time
        }
