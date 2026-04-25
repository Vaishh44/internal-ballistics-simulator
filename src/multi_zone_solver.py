import math
import numpy as np
from src.valve_flow import mass_flow_rate


class MultiZoneSolver:

    def __init__(self, L1, D1, L2, D2, L3, D3, powder, gas, shot):

        self.dt = 5e-7
        self.t = 0.0

        self.powder = powder
        self.gas = gas
        self.shot = shot

        self.L1, self.D1 = L1, D1
        self.L2, self.D2 = L2, D2
        self.L3, self.D3 = L3, D3

        self.A_ch = math.pi * (D1**2)/4
        self.A_pump = math.pi * (D2**2)/4
        self.A_launch = math.pi * (D3**2)/4

        # physics constants
        self.eta = 0.6
        self.h_loss = 120.0
        self.c_damp = 2.0
        self.k_leak = 1e-8
        self.tau_proj = 2e-4

        # Noble-Abel covolume
        self.b = 1e-3

        # ---------------- CHAMBER ----------------

        self.V_ch = self.A_ch * L1
        self.m_ch = 1e-6
        self.cv_powder = 1500

        self.U_ch = 101325 * self.V_ch / (powder.polytropic_ratio - 1)
        self.P_ch = 101325

        # ---------------- PISTON ----------------

        self.x_p = 0
        self.v_p = 0
        self.a_p = 0

        # ---------------- PUMP ----------------

        self.V_pump0 = self.A_pump * L2
        self.V_pump = self.V_pump0

        self.P_pump = shot.initial_gas_pressure
        self.T_pump = shot.initial_temp

        self.m_pump = self.P_pump * self.V_pump / (gas.R_specific * self.T_pump)
        self.U_pump = self.m_pump * gas.cv * self.T_pump

        # ---------------- BURST DISK ----------------

        self.valve_open = False
        self.burst_time = -1
        self.t_open = 0

        # ---------------- LAUNCH TUBE ----------------

        dist = max(shot.distance_valve_to_proj, 0.001)

        self.V_launch = self.A_launch * dist
        self.T_launch = shot.residual_gas_temp
        self.P_launch = shot.residual_gas_pressure

        self.R_air = 287
        self.cv_air = 718

        self.m_launch = self.P_launch * self.V_launch / (self.R_air * self.T_launch)
        self.U_launch = self.m_launch * self.cv_air * self.T_launch

        # ---------------- PROJECTILE ----------------

        self.x_proj = 0
        self.v_proj = 0
        self.a_proj = 0
        self.exit_time = -1

        # energy tracking
        self.e_powder_total = 0
        self.e_heat_loss = 0
        self.e_leak_loss = 0

    # -------------------------------------------------

    def noble_abel_pressure(self, rho, T):

        denom = 1 - self.b * rho
        if denom <= 0:
            denom = 1e-6

        return rho * self.gas.R_specific * T / denom

    # -------------------------------------------------

    def step(self):

        # ================= POWDER BURN =================

        if self.powder.z < 1:

            psi = self.powder.compute_shape_function(self.powder.z)

            dz_dt = self.powder.viva * psi * (self.P_ch ** self.powder.alpha)

            dz = dz_dt * self.dt

            if self.powder.z + dz > 1:
                dz = 1 - self.powder.z

            self.powder.z += dz

            dm = self.powder.powder_mass * dz
            dU = self.eta * self.powder.specific_force * dm

            self.m_ch += dm
            self.U_ch += dU
            self.e_powder_total += dU

        # ================= CHAMBER =================

        T_ch = self.U_ch / (self.m_ch * self.cv_powder)
        rho_ch = self.m_ch / self.V_ch

        self.P_ch = self.noble_abel_pressure(rho_ch, T_ch)

        # heat loss
        A_wall = 2*math.pi*(self.D1/2)**2 + math.pi*self.D1*self.L1

        dQ = self.h_loss * A_wall * (T_ch - 300) * self.dt
        dQ = max(dQ,0)

        self.U_ch -= dQ
        self.e_heat_loss += dQ

        # ================= PISTON =================

        F = (self.P_ch - self.P_pump) * self.A_pump

        fric = self.shot.piston_friction_coeff * abs(F) + self.c_damp * self.v_p

        if self.v_p != 0:
            fric *= math.copysign(1,self.v_p)

        F -= fric

        self.a_p = F / self.shot.piston_mass

        self.v_p += self.a_p * self.dt
        self.x_p += self.v_p * self.dt

        if self.x_p < 0:
            self.x_p = 0
            self.v_p = 0

        dx = self.v_p * self.dt

        self.V_ch += self.A_pump * dx
        self.V_pump -= self.A_pump * dx
        self.V_pump = max(self.V_pump,1e-6)

        # ================= PUMP =================

        self.U_pump -= self.P_pump * (-self.A_pump*dx)

        T_pump = self.U_pump / (self.m_pump * self.gas.cv)
        rho_pump = self.m_pump / self.V_pump

        self.P_pump = self.noble_abel_pressure(rho_pump, T_pump)

        # piston seal leakage
        m_leak = self.k_leak * (self.P_ch - self.P_pump)

        m_leak = max(m_leak,0)

        self.m_ch -= m_leak*self.dt
        self.m_pump += m_leak*self.dt

        # ================= BURST DISK =================

        if self.P_pump >= self.shot.valve_burst_pressure and self.burst_time < 0:
            self.burst_time = self.t

        if self.burst_time >= 0:
            self.t_open = self.t - self.burst_time

            if self.t_open >= self.shot.valve_opening_delay:
                self.valve_open = True

        # ================= VALVE FLOW =================

        if self.valve_open and self.P_pump > self.P_launch:

            A_val = self.shot.a_valve * (1 - math.exp(-self.t_open/5e-5))

            m_dot = mass_flow_rate(
                self.P_pump,
                self.P_launch,
                T_pump,
                self.gas.gamma,
                self.gas.R_specific,
                A_val,
                self.shot.cd_valve
            )

            dm = m_dot * self.dt
            dm = min(dm, self.m_pump*0.9)

            rho = self.m_pump/self.V_pump
            h = self.gas.cv*T_pump + self.P_pump/rho

            dU = dm*h

            self.m_pump -= dm
            self.m_launch += dm

            self.U_pump -= dU
            self.U_launch += dU

        # ================= LAUNCH GAS =================

        T_launch = self.U_launch/(self.m_launch*self.gas.cv)
        rho_launch = self.m_launch/self.V_launch

        self.P_launch = self.noble_abel_pressure(rho_launch,T_launch)

        # ================= PROJECTILE =================

        if self.valve_open:

            a_sound = math.sqrt(self.gas.gamma*self.gas.R_specific*T_launch)

            Mach = self.v_proj/max(a_sound,1e-6)

            P_eff = self.P_launch*(1 - (self.gas.gamma-1)/2*Mach**2)

            P_eff *= (1-math.exp(-self.t_open/self.tau_proj))

            Fp = (P_eff - self.shot.residual_gas_pressure)*self.A_launch

            self.a_proj = Fp/self.shot.projectile_mass

            self.v_proj += self.a_proj*self.dt
            self.x_proj += self.v_proj*self.dt

            dV = self.A_launch*self.v_proj*self.dt

            self.V_launch += dV
            self.U_launch -= self.P_launch*dV

        # ================= TIME =================

        self.t += self.dt

        if self.x_proj >= self.L3:
            self.exit_time = self.t
            return True

        return False

    # -------------------------------------------------

    def run(self):

        t_arr=[]
        p1_arr=[]
        p2_arr=[]
        p3_arr=[]

        xp_arr=[]
        vp_arr=[]

        xproj_arr=[]
        vproj_arr=[]
        aproj_arr=[]

        ep_arr=[]
        eg_arr=[]
        kep_arr=[]
        keproj_arr=[]
        el_arr=[]

        max_steps=int(0.02/self.dt)

        for step in range(max_steps):

            exited=self.step()

            if step%100==0 or exited:

                t_arr.append(self.t)

                p1_arr.append(self.P_ch)
                p2_arr.append(self.P_pump)
                p3_arr.append(self.P_launch)

                xp_arr.append(self.x_p)
                vp_arr.append(self.v_p)

                xproj_arr.append(self.x_proj)
                vproj_arr.append(self.v_proj)
                aproj_arr.append(self.a_proj)

                ep_arr.append(self.e_powder_total)

                eg_arr.append(self.U_ch+self.U_pump+self.U_launch)

                kep_arr.append(0.5*self.shot.piston_mass*self.v_p**2)

                keproj_arr.append(0.5*self.shot.projectile_mass*self.v_proj**2)

                el_arr.append(self.e_heat_loss+self.e_leak_loss)

            if exited:
                break

        return {
            't':np.array(t_arr),
            'p1':np.array(p1_arr),
            'p2':np.array(p2_arr),
            'p3':np.array(p3_arr),
            'x_p':np.array(xp_arr),
            'v_p':np.array(vp_arr),
            'x_proj':np.array(xproj_arr),
            'v_proj':np.array(vproj_arr),
            'a_proj':np.array(aproj_arr),
            'e_powder':np.array(ep_arr),
            'e_gas':np.array(eg_arr),
            'e_ke_piston':np.array(kep_arr),
            'e_ke_proj':np.array(keproj_arr),
            'e_loss':np.array(el_arr),
            'burst_time':self.burst_time,
            'exit_time':self.exit_time
        }