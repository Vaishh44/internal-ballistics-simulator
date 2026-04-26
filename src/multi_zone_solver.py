import math
import numpy as np

from src.valve_flow import mass_flow_rate
from src.shock_model import shock_jump
from src.euler_solver import euler_step


class MultiZoneSolver:

    def __init__(
        self,
        L1, D1,
        L2, D2,
        L3, D3,
        L_hpv, D_hpv,
        powder,
        gas,
        shot
    ):

        self.dt = 5e-7
        self.t = 0.0

        self.powder = powder
        self.gas = gas
        self.shot = shot

        self.L3 = L3

        # Areas
        self.A_ch = math.pi*(D1**2)/4
        self.A_pump = math.pi*(D2**2)/4
        self.A_launch = math.pi*(D3**2)/4
        self.A_hpv = math.pi*(D_hpv**2)/4

        # ---------------- VOLUMES ----------------

        self.V_ch = self.A_ch * L1
        self.V_pump = self.A_pump * L2
        self.V_hpv = self.A_hpv * L_hpv
        self.V_launch = self.A_launch * 0.001

        # ---------------- PRESSURES ----------------

        self.P_ch = 1e5
        self.P_pump = shot.initial_gas_pressure
        self.P_hpv = shot.initial_gas_pressure
        self.P_launch = shot.residual_gas_pressure

        self.T_launch = shot.residual_gas_temp
        
        # ---------------- MASSES ----------------

        self.m_ch = 1e-6
        self.m_pump = self.P_pump * self.V_pump / (self.gas.R_specific * 300)
        self.m_hpv = self.P_hpv * self.V_hpv / (self.gas.R_specific * 300)
        self.m_launch = self.P_launch * self.V_launch / (self.gas.R_specific * self.T_launch)

        # ---------------- ENERGIES ----------------

        self.U_ch = 1e3
        T_init = 300
        self.U_pump = self.m_pump * self.gas.cv * T_init
        T_init = 300
        self.U_hpv = self.m_hpv * self.gas.cv * T_init
        self.U_launch = self.m_launch * self.gas.cv * self.T_launch

        

        # ---------------- PISTON ----------------

        self.x_p = 0
        self.v_p = 0
        self.a_p = 0

        # ---------------- PROJECTILE ----------------

        self.x_proj = 0
        self.v_proj = 0
        self.a_proj = 0

        self.exit_time = -1

        # ---------------- VALVE ----------------

        self.valve_open = False
        self.burst_time = -1
        self.t_open = 0

        # ---------------- MULTI CELL PUMP ----------------

        self.N_cells = 20

        rho_init = self.m_pump / self.V_pump
        self.rho_cells = np.ones(self.N_cells) * rho_init
        self.u_cells = np.zeros(self.N_cells)
        self.P_cells = np.ones(self.N_cells) * self.P_pump

        # ---------------- ENERGY TRACK ----------------

        self.e_powder_total = 0
        self.e_heat_loss = 0
        self.e_leak_loss = 0

        self.cv_powder = 1500

    # --------------------------------------------------

    def noble_abel(self, rho, T):

        b = 1e-3

        denom = 1 - b*rho
        if denom <= 0:
            denom = 1e-6

        return rho * self.gas.R_specific * T / denom
    # --------------------------------------------------

    def virial_eos(self, rho, T):

        B = -1e-4
        C = 1e-7

        return rho * self.gas.R_specific * T * (1 + B*rho + C*rho**2)
    # --------------------------------------------------

    def step(self):
        # ---------- ADAPTIVE TIMESTEP ----------

        a_sound = math.sqrt(self.gas.gamma * self.gas.R_specific * 300)

        dx = max(self.V_pump / self.A_pump / self.N_cells, 1e-6)

        self.dt = min(
            5e-7,
            0.5 * dx / (abs(self.v_p) + a_sound)
        )
        # ---------- POWDER BURN ----------

        if self.powder.z < 1:

            psi = self.powder.compute_shape_function(self.powder.z)

            dz_dt = self.powder.viva * psi * (self.P_ch**self.powder.alpha)

            dz = dz_dt * self.dt

            if self.powder.z + dz > 1:
                dz = 1 - self.powder.z

            self.powder.z += dz

            dm = self.powder.powder_mass * dz
            dU = self.powder.specific_force * dm * 0.6

            self.m_ch += dm
            self.U_ch += dU
            self.e_powder_total += dU

        # ---------- CHAMBER ----------

        T_ch = self.U_ch / (self.m_ch * self.cv_powder)
        rho_ch = self.m_ch / self.V_ch

        

        # ---------- HEAT TRANSFER ----------

        h = 200
        A_wall = self.A_ch * 4
        T_wall = 300

        Qdot = h * A_wall * (T_ch - T_wall)
        dQ = Qdot * self.dt

        self.U_ch -= dQ
        self.e_heat_loss += dQ

        T_ch = self.U_ch / (self.m_ch * self.cv_powder)

        self.P_ch = self.virial_eos(rho_ch, T_ch)
        # ---------- PISTON ----------

        F = max(0, self.P_ch - self.P_pump) * self.A_pump

        self.a_p = F / self.shot.piston_mass

        self.v_p += self.a_p * self.dt
        self.v_p *= 0.999
        self.v_p = max(min(self.v_p, 2000), -2000)
        self.x_p += self.v_p * self.dt

        dx = self.v_p * self.dt

        V_old = self.V_pump

        self.V_pump = max(self.V_pump - self.A_pump * dx, 1e-8)
        self.V_ch = max(self.V_ch + self.A_pump * dx, 1e-8)

        # --- isentropic compression heating ---
        compression_ratio = V_old / self.V_pump

        self.P_pump *= compression_ratio ** self.gas.gamma
        self.U_pump *= compression_ratio ** (self.gas.gamma - 1)
        # update first pump cell due to piston compression
        self.P_cells[0] = self.P_pump
        rho_new = self.m_pump / max(self.V_pump, 1e-6)
        self.rho_cells[0] = rho_new
                        # ---------- PISTON SEAL LEAKAGE ----------

        if self.P_ch > self.P_cells[0]:

            C_l = 0.01
            A_seal = self.A_pump * 0.001

            rho = self.m_ch / max(self.V_ch,1e-6)

            m_leak = C_l * A_seal * math.sqrt(2 * rho * (self.P_ch - self.P_cells[0]))

            dm = m_leak * self.dt
            dm = min(dm, self.m_ch)

            self.m_ch -= dm
            self.m_pump += dm

            self.e_leak_loss += dm * self.gas.cv * 300
        # ---------- SHOCK ----------

        a = math.sqrt(self.gas.gamma * self.gas.R_specific * T_ch)

        Mach = abs(self.v_p) / max(a,1e-6)

        if Mach > 1:

            P2, rho2 = shock_jump(
                self.P_cells[0],
                self.rho_cells[0],
                self.gas.gamma,
                Mach
            )

            self.P_cells[0] = P2
            self.rho_cells[0] = rho2

        # ---------- 1D EULER GAS DYNAMICS ----------

        dx_cells = max(self.V_pump / self.A_pump / self.N_cells, 1e-6)

        self.rho_cells, self.u_cells, self.P_cells = euler_step(
    self.rho_cells,
    self.u_cells,
    self.P_cells,
    self.gas.gamma,
    dx=dx_cells,
    dt=self.dt
)

        self.P_pump = self.P_cells[0]

        # ---------- HPV ----------

        rho_hpv = self.m_hpv / self.V_hpv
        T_hpv = self.U_hpv / max(self.m_hpv * self.gas.cv, 1e-6)
        self.P_hpv = self.noble_abel(rho_hpv, T_hpv)

        if self.P_pump > self.P_hpv * 1.01:

            m_dot = mass_flow_rate(
                self.P_pump,
                self.P_hpv,
                T_hpv,
                self.gas.gamma,
                self.gas.R_specific,
                1e-4,
                0.8
            )

            dm = m_dot * self.dt

            
            dm = min(dm, self.m_pump)
            self.m_pump -= dm
            self.m_hpv += dm

        # ---------- BURST DISK ----------

        if self.P_hpv >= self.shot.valve_burst_pressure and self.burst_time < 0:
            self.burst_time = self.t

        if self.burst_time >= 0:

            self.t_open = self.t - self.burst_time

            if self.t_open >= self.shot.valve_opening_delay:
                self.valve_open = True

        # ---------- HPV → LAUNCH ----------

        if self.valve_open and self.P_hpv > self.P_launch:

            m_dot = mass_flow_rate(
                self.P_hpv,
                self.P_launch,
                self.T_launch,
                self.gas.gamma,
                self.gas.R_specific,
                self.shot.a_valve,
                self.shot.cd_valve
            )

            dm = m_dot * self.dt
            dm = min(dm, self.m_hpv * 0.1)

            self.m_hpv -= dm
            self.m_launch += dm

        # ---------- LAUNCH GAS ----------

        rho_launch = self.m_launch / max(self.V_launch, 1e-8)
        self.P_launch = self.noble_abel(
            rho_launch,
            self.T_launch
        )

        # ---------- PROJECTILE ----------

        if self.valve_open:

            a_sound = math.sqrt(
                self.gas.gamma *
                self.gas.R_specific *
                self.T_launch
            )

            Mach = self.v_proj / max(a_sound,1e-6)

            P_eff = self.P_launch * max(0.1, 1 - (self.gas.gamma-1)/2 * Mach**2)

            rho_launch = self.m_launch / max(self.V_launch,1e-6)

            Cd_proj = 0.3

            F_drag = 0.5 * Cd_proj * rho_launch * abs(self.v_proj) * self.v_proj * self.A_launch

            F = P_eff * self.A_launch - F_drag

            self.a_proj = F / self.shot.projectile_mass

            self.v_proj += self.a_proj * self.dt
            self.x_proj += self.v_proj * self.dt

            dV = self.A_launch * self.v_proj * self.dt

            self.V_launch += dV

        # ---------- TIME ----------

        self.t += self.dt

        if self.x_proj >= self.L3:

            self.exit_time = self.t
            return True

        return False

    # --------------------------------------------------

    def run(self):

        t_arr=[]
        p1=[]
        p2=[]
        p3=[]
        xp=[]
        vp=[]
        xproj=[]
        vproj=[]
        aproj=[]
        ep=[]
        eg=[]
        kep=[]
        keproj=[]
        el=[]

        max_steps=int(0.01/self.dt)

        for i in range(max_steps):
            
            stop = self.step()

            # ---- solver stability check ----
            if (
                np.isnan(self.P_ch) or
                np.isnan(self.P_pump) or
                np.isnan(self.P_launch) or
                np.isinf(self.P_ch) or
                np.isinf(self.P_pump) or
                np.isinf(self.P_launch)
            ):
                print("Solver unstable — stopping early")
                break

            if i%100==0 or stop:

                t_arr.append(self.t)

                p1.append(self.P_ch)
                p2.append(self.P_pump)
                p3.append(self.P_launch)

                xp.append(self.x_p)
                vp.append(self.v_p)

                xproj.append(self.x_proj)
                vproj.append(self.v_proj)
                aproj.append(self.a_proj)

                ep.append(self.e_powder_total)

                eg.append(self.U_ch+self.U_pump+self.U_hpv+self.U_launch)

                kep.append(0.5*self.shot.piston_mass*self.v_p**2)

                keproj.append(0.5*self.shot.projectile_mass*self.v_proj**2)

                el.append(self.e_heat_loss+self.e_leak_loss)

            if stop:
                break

        return {
            "t":np.array(t_arr),
            "p1":np.array(p1),
            "p2":np.array(p2),
            "p3":np.array(p3),
            "x_p":np.array(xp),
            "v_p":np.array(vp),
            "x_proj":np.array(xproj),
            "v_proj":np.array(vproj),
            "a_proj":np.array(aproj),
            "e_powder":np.array(ep),
            "e_gas":np.array(eg),
            "e_ke_piston":np.array(kep),
            "e_ke_proj":np.array(keproj),
            "e_loss":np.array(el),
            "burst_time":self.burst_time,
            "exit_time":self.exit_time
        }