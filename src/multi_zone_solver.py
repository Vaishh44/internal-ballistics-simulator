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

        # ---------------- MASSES ----------------

        self.m_ch = 1e-6
        self.m_pump = 1e-6
        self.m_hpv = 1e-6
        self.m_launch = 1e-6

        # ---------------- ENERGIES ----------------

        self.U_ch = 1e3
        self.U_pump = 1e3
        self.U_hpv = 1e3
        self.U_launch = 1e3

        # ---------------- PRESSURES ----------------

        self.P_ch = 1e5
        self.P_pump = shot.initial_gas_pressure
        self.P_hpv = shot.initial_gas_pressure
        self.P_launch = shot.residual_gas_pressure

        self.T_launch = shot.residual_gas_temp

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

        self.rho_cells = np.ones(self.N_cells) * 1.0
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

    def step(self):

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

        self.P_ch = self.noble_abel(rho_ch, T_ch)

        # ---------- PISTON ----------

        F = (self.P_ch - self.P_cells[0]) * self.A_pump

        self.a_p = F / self.shot.piston_mass

        self.v_p += self.a_p * self.dt
        self.v_p = max(min(self.v_p, 2000), -2000)
        self.x_p += self.v_p * self.dt

        dx = self.v_p * self.dt

        self.V_pump = max(self.V_pump - self.A_pump * dx, 1e-8)
        self.V_ch = max(self.V_ch + self.A_pump * dx, 1e-8)

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

        self.rho_cells, self.u_cells, self.P_cells = euler_step(
            self.rho_cells,
            self.u_cells,
            self.P_cells,
            self.gas.gamma,
            dx=0.2,
            dt=self.dt
        )

        self.P_pump = self.P_cells[-1]

        # ---------- HPV ----------

        rho_hpv = self.m_hpv / self.V_hpv
        T_hpv = self.U_hpv / max(self.m_hpv * self.gas.cv, 1e-6)
        self.P_hpv = self.noble_abel(rho_hpv, T_hpv)

        if self.P_pump > self.P_hpv:

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

            F = P_eff * self.A_launch

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

        max_steps=int(0.02/self.dt)

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