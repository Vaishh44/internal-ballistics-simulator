import sys
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QFormLayout, QGroupBox, 
                             QMessageBox, QTabWidget, QComboBox, QApplication, QGridLayout,
                             QProgressBar)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QTimer
import matplotlib.patches as patches

from src.combustion import PowderCombustion
from src.coupled_solver import CoupledSolver
from src.gas_comparison import compare_all_gases
from src.parameter_sweep import run_sweep

DARK_THEME_QSS = """
QMainWindow {
    background-color: #121212;
}
QWidget {
    background-color: #121212;
    color: #ffffff;
    font-size: 14px;
}
QTabWidget::pane {
    border: 2px solid #1E1E2E;
    background-color: #1E1E2E;
    border-radius: 6px;
}
QTabBar::tab {
    background: #1E1E2E;
    color: #aaaaaa;
    padding: 10px 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #1E1E2E;
    color: #00ADB5;
    font-weight: bold;
    border-bottom: 2px solid #00ADB5;
}
QGroupBox {
    border: 1px solid #333344;
    border-radius: 8px;
    margin-top: 15px;
    background-color: #1E1E2E;
    padding: 20px 15px 15px 15px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    color: #00ADB5;
    font-weight: bold;
    background-color: transparent;
}
QLineEdit, QComboBox {
    background-color: #121212;
    border: 1px solid #333344;
    padding: 8px;
    border-radius: 4px;
    color: white;
}
QLineEdit:focus, QComboBox:focus {
    border: 1px solid #00ADB5;
}
QPushButton {
    background-color: #00ADB5;
    border-radius: 6px;
    padding: 14px 30px;
    font-weight: bold;
    color: white;
    font-size: 16px;
}
QPushButton:hover {
    background-color: #008C9E;
}
QPushButton:disabled {
    background-color: #333344;
    color: #888888;
}
QLabel {
    background-color: transparent;
    font-weight: bold;
}
QProgressBar {
    border: 1px solid #333344;
    border-radius: 4px;
    text-align: center;
    background-color: #121212;
    color: white;
    height: 25px;
}
QProgressBar::chunk {
    background-color: #4CAF50;
    width: 20px;
}
"""

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, num_subplots=1):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.patch.set_facecolor('#1E1E2E')
        self.axes = []
        for i in range(num_subplots):
            ax = fig.add_subplot(num_subplots, 1, i+1)
            ax.set_facecolor('#121212')
            ax.tick_params(colors='white', labelsize=10)
            ax.xaxis.label.set_color('white')
            ax.xaxis.label.set_fontsize(12)
            ax.yaxis.label.set_color('white')
            ax.yaxis.label.set_fontsize(12)
            ax.title.set_color('#00ADB5')
            ax.title.set_fontsize(14)
            for spine in ax.spines.values():
                spine.set_color('#333344')
            self.axes.append(ax)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        fig.tight_layout()

class CompareCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.patch.set_facecolor('#1E1E2E')
        
        self.ax_v = fig.add_subplot(2, 2, 1)
        self.ax_p = fig.add_subplot(2, 2, 2)
        self.ax_t = fig.add_subplot(2, 1, 2)
        
        for ax in [self.ax_v, self.ax_p, self.ax_t]:
            ax.set_facecolor('#121212')
            ax.tick_params(colors='white', labelsize=10)
            ax.title.set_color('#00ADB5')
            ax.title.set_fontsize(14)
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#333344')
                
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        fig.tight_layout(pad=2.0)

class SweepCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.patch.set_facecolor('#1E1E2E')
        
        self.ax = fig.add_subplot(1, 1, 1)
        self.ax.set_facecolor('#121212')
        self.ax.tick_params(colors='white', labelsize=10)
        self.ax.title.set_color('#00ADB5')
        self.ax.title.set_fontsize(14)
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        for spine in self.ax.spines.values():
            spine.set_color('#333344')
                
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        fig.tight_layout()

class AnimationCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.patch.set_facecolor('#1E1E2E')
        
        self.ax = fig.add_subplot(1, 1, 1)
        self.ax.set_facecolor('#121212')
        self.ax.tick_params(colors='white', labelsize=10)
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        for spine in self.ax.spines.values():
            spine.set_color('#333344')
            
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        fig.tight_layout()

        self.piston_patch = None
        self.proj_patch = None
        self.gas_patch = None
        self.barrier_line = None
        self.barrier_open = False
        self.piston_len = 0.0
        
        self.trail_line = None
        self.trail_x = []
        self.trail_y = []
        
        self.lbl_piston = None
        self.lbl_valve = None
        self.lbl_proj = None
        self.txt_telemetry = None
        self.txt_event = None

    def setup_scene(self, L2, L3, D2, D3):
        self.ax.clear()
        self.ax.axis('off')
        
        total_L = L2 + L3
        max_D = max(D2, D3)
        self.ax.set_ylim(-max_D*4, max_D*4)
        
        # Piston Geometry
        p_len = max(0.4, min(L2 * 0.08, 1.5))
        self.piston_len = p_len
        self.ax.set_xlim(-p_len - 0.5, total_L + 0.5)
        
        # Tubes (Thickened)
        s2_top = D2 / 2
        s2_bot = -D2 / 2
        self.ax.plot([0, L2], [s2_top, s2_top], color='white', lw=4)
        self.ax.plot([0, L2], [s2_bot, s2_bot], color='white', lw=4)
        
        s3_top = D3 / 2
        s3_bot = -D3 / 2
        self.ax.plot([L2, L2 + L3], [s3_top, s3_top], color='gray', lw=4)
        self.ax.plot([L2, L2 + L3], [s3_bot, s3_bot], color='gray', lw=4)
        
        # Gas Region (Light Cyan, semi-transparent)
        self.gas_patch = patches.Rectangle((0, s2_bot), L2, D2, color='#00FFFF', alpha=0.3, zorder=2)
        self.ax.add_patch(self.gas_patch)
        
        # Valve barrier (Yellow = Closed, Green = Open)
        self.barrier_line = self.ax.vlines(L2, s2_bot*1.2, s2_top*1.2, colors='#FFEB3B', linestyles='solid', lw=5, zorder=4)
        self.barrier_open = False
        self.lbl_valve = self.ax.text(L2, s2_top*1.5, "Valve\n[Closed]", color='#FFEB3B', ha='center', va='bottom', fontsize=11, weight='bold')
        
        # Piston (Blue, taller than tube)
        self.piston_patch = patches.Rectangle((-p_len, s2_bot * 1.5), p_len, D2 * 1.5, color='#2196F3', zorder=5)
        self.ax.add_patch(self.piston_patch)
        self.lbl_piston = self.ax.text(-p_len/2, s2_top*1.8, "Piston", color='#2196F3', ha='center', va='bottom', fontsize=11, weight='bold')
        
        # Projectile Trail
        self.trail_x = []
        self.trail_y = []
        self.trail_line, = self.ax.plot([], [], color='#FF5252', lw=2, linestyle=':', zorder=3)
        
        # Projectile (Red, larger)
        proj_rad = max(D3, 0.05)
        self.proj_patch = patches.Circle((L2, 0), proj_rad, color='#FF5252', zorder=6)
        self.ax.add_patch(self.proj_patch)
        self.lbl_proj = self.ax.text(L2, s3_top*1.5 + proj_rad, "Projectile", color='#FF5252', ha='center', va='bottom', fontsize=11, weight='bold')
        
        # Texts overlays
        self.txt_event = self.ax.text(total_L/2, max_D*3, "Stage 1: Combustion", color='#00ADB5', ha='center', va='center', fontsize=16, weight='bold')
        
        self.txt_telemetry = self.ax.text(total_L, max_D*3.5, "Time: 0.00 ms\nPressure: 0.0 bar\nVelocity: 0.0 m/s", 
                                          color='white', ha='right', va='top', fontsize=12,
                                          bbox=dict(facecolor='#121212', alpha=0.8, edgecolor='#333344'))

        self.draw_idle()

    def update_positions(self, x_p, x_proj, L2, L3, time, current_pressure, velocity_proj):
        # Update Valve Logic (Burst >= 470 bar)
        if not self.barrier_open and current_pressure >= 470.0:
            self.barrier_open = True
            self.barrier_line.set_color('#4CAF50')
            self.barrier_line.set_linestyle('--')
            self.lbl_valve.set_text("Valve\n[Open]")
            self.lbl_valve.set_color('#4CAF50')
            self.txt_event.set_text("Stage 3: Projectile Launch")
            self.txt_event.set_color('#FFF')
            
        elif not self.barrier_open and current_pressure < 470.0 and x_p > 1e-4:
            self.txt_event.set_text("Stage 2: Gas Compression")
            self.txt_event.set_color('#FF9800')
            
        if self.barrier_open and x_proj >= L3 * 0.95:
            self.txt_event.set_text("Projectile Exit!")
            self.txt_event.set_color('#FF5252')

        # Update Piston
        px = x_p - self.piston_len
        self.piston_patch.set_x(px)
        self.lbl_piston.set_position((px + self.piston_len/2, self.lbl_piston.get_position()[1]))
        
        # Update Projectile & Trail
        curr_proj_x = L2 + x_proj
        self.proj_patch.center = (curr_proj_x, 0)
        self.lbl_proj.set_position((curr_proj_x, self.lbl_proj.get_position()[1]))
        
        if self.barrier_open and x_proj > 0:
            self.trail_x.append(curr_proj_x)
            self.trail_y.append(0)
            self.trail_line.set_data(self.trail_x, self.trail_y)

        # Update Gas Region dynamically
        self.gas_patch.set_x(x_p)
        if self.barrier_open:
            self.gas_patch.set_width(curr_proj_x - x_p)
        else:
            self.gas_patch.set_width(L2 - x_p)
            
        # Telemetry
        self.txt_telemetry.set_text(f"Time: {time:.2f} ms\nPressure: {current_pressure:.1f} bar\nVelocity: {velocity_proj:.1f} m/s")

        self.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Two-Stage Light Gas Gun Simulator")
        self.setGeometry(50, 50, 1400, 900)
        self.setStyleSheet(DARK_THEME_QSS)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        self.main_tabs = QTabWidget()
        main_layout.addWidget(self.main_tabs)
        
        # --- TAB 1: INPUT PARAMETERS ---
        self.tab_inputs = QWidget()
        self.build_tab_inputs()
        self.main_tabs.addTab(self.tab_inputs, "INPUT PARAMETERS")
        
        # --- TAB 2: SIMULATION ---
        self.tab_simulation = QWidget()
        self.build_tab_simulation()
        self.main_tabs.addTab(self.tab_simulation, "SIMULATION")
        
        # --- TAB 3: RESULTS ---
        self.tab_results = QWidget()
        self.build_tab_results()
        self.main_tabs.addTab(self.tab_results, "RESULTS")

        # --- TAB 4: GAS COMPARISON ---
        self.tab_compare = QWidget()
        self.build_tab_comparison()
        self.main_tabs.addTab(self.tab_compare, "GAS COMPARISON")

        # --- TAB 5: PARAMETER SWEEP ---
        self.tab_sweep = QWidget()
        self.build_tab_sweep()
        self.main_tabs.addTab(self.tab_sweep, "PARAMETER SWEEP")

        # --- TAB 6: ANIMATION ---
        self.tab_animation = QWidget()
        self.build_tab_animation()
        self.main_tabs.addTab(self.tab_animation, "ANIMATION")
        
        # Animation states
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.anim_step)
        self.anim_index = 0
        self.anim_data_xp = []
        self.anim_data_xproj = []
        self.anim_data_t = []
        self.anim_data_p = []
        self.anim_data_v = []
        self.anim_L2 = 0.0
        self.anim_L3 = 0.0

    def build_tab_inputs(self):
        layout = QVBoxLayout(self.tab_inputs)
        
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(20)
        
        # [ Geometry ]
        geom_group = QGroupBox("Geometry")
        geom_form = QFormLayout()
        geom_form.setVerticalSpacing(10)
        self.inp_L1 = QLineEdit("0.440")
        self.inp_D1 = QLineEdit("0.1646")
        self.inp_L2 = QLineEdit("12.245")
        self.inp_D2 = QLineEdit("0.105")
        self.inp_L3 = QLineEdit("8.1")
        self.inp_D3 = QLineEdit("0.036")
        geom_form.addRow("Powder Chamber Length (m):", self.inp_L1)
        geom_form.addRow("Powder Chamber Diameter (m):", self.inp_D1)
        geom_form.addRow("Pump Tube Length (m):", self.inp_L2)
        geom_form.addRow("Pump Tube Diameter (m):", self.inp_D2)
        geom_form.addRow("Launch Tube Bore Length (m):", self.inp_L3)
        geom_form.addRow("Launch Tube Bore Diameter (m):", self.inp_D3)
        geom_group.setLayout(geom_form)
        
        # [ Gas & Material ]
        gas_group = QGroupBox("Gas & Material")
        gas_form = QFormLayout()
        gas_form.setVerticalSpacing(10)
        self.cmb_gas = QComboBox()
        self.cmb_gas.addItems(["Hydrogen (γ=1.4)", "Helium (γ=1.66)", "Nitrogen (γ=1.4)"])
        self.cmb_gas.currentTextChanged.connect(self.update_gamma)
        self.lbl_gamma = QLabel("Gamma: 1.4")
        self.lbl_gamma.setStyleSheet("color: #4CAF50;")
        gas_form.addRow("Gas Type:", self.cmb_gas)
        gas_form.addRow("Specific Heat Ratio:", self.lbl_gamma)
        gas_group.setLayout(gas_form)
        
        # [ Combustion Parameters ]
        comb_group = QGroupBox("Combustion Parameters")
        comb_form = QFormLayout()
        comb_form.setVerticalSpacing(10)
        self.inp_viva = QLineEdit("2500")
        self.inp_beta = QLineEdit("0.008")
        self.inp_alpha = QLineEdit("0.8")
        self.inp_f = QLineEdit("1e6")
        self.inp_m1 = QLineEdit("1.6")
        comb_form.addRow("VIVA [1000-5000]:", self.inp_viva)
        comb_form.addRow("Beta (β) [0.001-0.02]:", self.inp_beta)
        comb_form.addRow("Alpha (α) [0.7-0.9]:", self.inp_alpha)
        comb_form.addRow("Specific Force f (J/kg):", self.inp_f)
        comb_form.addRow("Powder Mass (kg):", self.inp_m1)
        comb_group.setLayout(comb_form)
        
        # [ Projectile & Initial Conditions ]
        proj_group = QGroupBox("Projectile & Initial Conditions")
        proj_form = QFormLayout()
        proj_form.setVerticalSpacing(10)
        self.inp_mproj = QLineEdit("0.0279")
        self.inp_mp = QLineEdit("4.5")
        self.inp_P0 = QLineEdit("5")
        self.inp_exp_vel = QLineEdit("")
        self.inp_exp_vel.setPlaceholderText("e.g. 8512 (Optional)")
        proj_form.addRow("Projectile Mass (kg):", self.inp_mproj)
        proj_form.addRow("Piston Mass (kg):", self.inp_mp)
        proj_form.addRow("Initial Gas Pressure (bar):", self.inp_P0)
        proj_form.addRow("Experimental Velocity (m/s) (Optional):", self.inp_exp_vel)
        proj_group.setLayout(proj_form)
        
        grid_layout.addWidget(geom_group, 0, 0)
        grid_layout.addWidget(gas_group, 1, 0)
        grid_layout.addWidget(comb_group, 0, 1)
        grid_layout.addWidget(proj_group, 1, 1)
        
        layout.addLayout(grid_layout)
        layout.addStretch()

    def update_gamma(self, text):
        if "Helium" in text:
            self.lbl_gamma.setText("Gamma: 1.66")
        else:
            self.lbl_gamma.setText("Gamma: 1.4")

    def build_tab_simulation(self):
        layout = QVBoxLayout(self.tab_simulation)
        layout.addStretch()
        
        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignCenter)
        
        self.lbl_sim_status = QLabel("Status: Ready")
        self.lbl_sim_status.setStyleSheet("font-size: 24px; color: #00ADB5;")
        self.lbl_sim_status.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(self.lbl_sim_status)
        
        # Some spacing
        center_layout.addSpacing(20)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(600)
        
        prog_layout = QHBoxLayout()
        prog_layout.setAlignment(Qt.AlignCenter)
        prog_layout.addWidget(self.progress_bar)
        center_layout.addLayout(prog_layout)
        
        center_layout.addSpacing(40)
        
        self.btn_run = QPushButton("RUN SIMULATION")
        self.btn_run.setFixedWidth(300)
        self.btn_run.clicked.connect(self.run_simulation)
        
        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignCenter)
        btn_layout.addWidget(self.btn_run)
        
        center_layout.addLayout(btn_layout)
        
        layout.addLayout(center_layout)
        layout.addStretch()

    def build_tab_results(self):
        layout = QHBoxLayout(self.tab_results)
        
        # LEFT PANEL
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(320)
        left_panel.setStyleSheet("QWidget { background-color: #1E1E2E; border-radius: 8px; }")
        
        title_lbl = QLabel("SIMULATION SUMMARY")
        title_lbl.setStyleSheet("font-size: 18px; color: #00ADB5; padding-bottom: 10px; border-bottom: 1px solid #333344;")
        left_layout.addWidget(title_lbl)
        
        left_layout.addSpacing(10)
        
        self.res_muzzle_v = QLabel("Muzzle Velocity:\n--")
        self.res_peak_p = QLabel("Peak Pressure:\n--")
        self.res_burst_t = QLabel("Valve Burst Time:\n--")
        self.res_firing_t = QLabel("Firing Time:\n--")
        self.res_exp_comp = QLabel("Error vs Exp:\n--")
        
        for lbl in [self.res_muzzle_v, self.res_peak_p, self.res_burst_t, self.res_firing_t, self.res_exp_comp]:
            lbl.setStyleSheet("font-size: 15px; padding: 10px; background-color: #121212; border-radius: 5px;")
            left_layout.addWidget(lbl)
            left_layout.addSpacing(5)
            
        left_layout.addStretch()
        
        # RIGHT PANEL
        right_panel = QTabWidget()
        
        self.tab_graph_pressure = QWidget()
        l_p = QVBoxLayout(self.tab_graph_pressure)
        self.canvas_pressure = MatplotlibCanvas(self, width=8, height=6, dpi=100, num_subplots=1)
        l_p.addWidget(self.canvas_pressure)
        right_panel.addTab(self.tab_graph_pressure, "Pressure vs Time")
        
        self.tab_graph_velocity = QWidget()
        l_v = QVBoxLayout(self.tab_graph_velocity)
        self.canvas_velocity = MatplotlibCanvas(self, width=8, height=6, dpi=100, num_subplots=1)
        l_v.addWidget(self.canvas_velocity)
        right_panel.addTab(self.tab_graph_velocity, "Velocity vs Time")
        
        self.tab_graph_accel = QWidget()
        l_a = QVBoxLayout(self.tab_graph_accel)
        self.canvas_accel = MatplotlibCanvas(self, width=8, height=6, dpi=100, num_subplots=1)
        l_a.addWidget(self.canvas_accel)
        right_panel.addTab(self.tab_graph_accel, "Acceleration vs Time")
        
        layout.addWidget(left_panel)
        layout.addWidget(right_panel)

    def run_simulation(self):
        try:
            # Stage 1
            L1 = float(self.inp_L1.text())
            D1 = float(self.inp_D1.text())
            m1 = float(self.inp_m1.text())
            f = float(self.inp_f.text())
            alpha = float(self.inp_alpha.text())
            beta = float(self.inp_beta.text())
            viva = float(self.inp_viva.text())
            
            # Stage 2
            mp = float(self.inp_mp.text())
            D2 = float(self.inp_D2.text())
            L2 = float(self.inp_L2.text())
            P0 = float(self.inp_P0.text()) * 1e5
            
            gas_choice = self.cmb_gas.currentText()
            gamma = 1.66 if "Helium" in gas_choice else 1.4
            
            # Stage 3
            mproj = float(self.inp_mproj.text())
            L3 = float(self.inp_L3.text())
            D3 = float(self.inp_D3.text())
            
            # Exp value
            exp_val_text = self.inp_exp_vel.text().strip()
            exp_velocity = float(exp_val_text) if exp_val_text else None
            
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please ensure all numerical inputs are valid.")
            return

        self.btn_run.setEnabled(False)
        self.btn_run.setText("COMPUTING...")
        self.lbl_sim_status.setText("Status: Running...")
        self.lbl_sim_status.setStyleSheet("font-size: 24px; color: #FF5252;")
        self.progress_bar.setValue(0)
        self.main_tabs.setCurrentIndex(1) # Keep on SIMULATION tab while running
        QApplication.processEvents()
        
        try:
            comb = PowderCombustion(length=L1, diameter=D1, mass_powder=m1, f=f, alpha=alpha, beta=beta, viva=viva)
            solver = CoupledSolver(m_p=mp, D_p=D2, L_p=L2, m_proj=mproj, D_proj=D3, L_proj=L3, P0_gas=P0, gamma=gamma)
            
            t_arr = []; p1_arr = []
            x_s2_arr = []; v_s2 = []; p2_arr = []
            x_s3_arr = []; v_s3 = []; a_s3 = []
            
            max_time = 1.0 
            dt = comb.dt
            t_current = 0.0
            
            plot_step = 0
            exited = False
            
            while t_current < max_time and not exited:
                comb.step()
                exited = solver.step(comb.P)
                
                if plot_step % 200 == 0 or exited:
                    t_arr.append(t_current)
                    p1_arr.append(comb.P)
                    
                    x_s2_arr.append(solver.x_p)
                    v_s2.append(solver.v_p)
                    p2_arr.append(solver.P_gas)
                    
                    x_s3_arr.append(solver.x_proj)
                    v_s3.append(solver.v_proj)
                    a_s3.append(solver.a_proj)
                    
                if plot_step % 5000 == 0:
                    prog = min(int((t_current / max_time) * 100), 99)
                    self.progress_bar.setValue(prog)
                    QApplication.processEvents()
                    
                t_current += dt
                plot_step += 1
            
            self.progress_bar.setValue(100)
            self.lbl_sim_status.setText("Status: Completed")
            self.lbl_sim_status.setStyleSheet("font-size: 24px; color: #4CAF50;")
            
            t_arr = np.array(t_arr) * 1000 # ms
            p1_arr = np.array(p1_arr) / 1e5 # bar
            
            x_s2_arr = np.array(x_s2_arr)
            v_s2 = np.array(v_s2)
            p2_arr = np.array(p2_arr) / 1e5
            
            x_s3_arr = np.array(x_s3_arr)
            v_s3 = np.array(v_s3)
            a_s3 = np.array(a_s3)
            
            # Store for animation
            self.anim_data_xp = list(x_s2_arr)
            self.anim_data_xproj = list(x_s3_arr)
            self.anim_data_t = list(t_arr)
            self.anim_data_p = list(p2_arr)
            self.anim_data_v = list(v_s3)
            self.anim_L2 = L2
            self.anim_L3 = L3
            self.anim_index = 0
            self.canvas_anim.setup_scene(L2, L3, D2, D3)
            self.lbl_anim_status.setText("Status: Ready to Play")
            self.lbl_anim_status.setStyleSheet("color: #4CAF50;")
            
            # --- UPDATE UI --- #
            peak_p1 = np.max(p1_arr)
            self.res_peak_p.setText(f"Peak Pressure:\n{peak_p1:.2f} bar")
            
            if solver.burst_time >= 0.0:
                self.res_burst_t.setText(f"Valve Burst Time:\n{solver.burst_time * 1000:.3f} ms")
            else:
                self.res_burst_t.setText("Valve Burst Time:\nDid not burst")
                
            if exited:
                final_mv = v_s3[-1]
                self.res_muzzle_v.setText(f"Muzzle Velocity:\n{final_mv:.2f} m/s")
                self.res_firing_t.setText(f"Firing Time:\n{solver.exit_time * 1000:.3f} ms")
                if exp_velocity is not None:
                    error = abs(exp_velocity - final_mv) / exp_velocity * 100.0
                    self.res_exp_comp.setText(f"Error vs Exp ({exp_velocity} m/s):\n{error:.2f}%")
                    if error < 5.0:
                        self.res_exp_comp.setStyleSheet("font-size: 15px; padding: 10px; background-color: #121212; border-radius: 5px; color: #4CAF50;")
                    else:
                        self.res_exp_comp.setStyleSheet("font-size: 15px; padding: 10px; background-color: #121212; border-radius: 5px; color: #FF5252;")
                else:
                    self.res_exp_comp.setText("Error vs Exp:\nN/A")
                    self.res_exp_comp.setStyleSheet("font-size: 15px; padding: 10px; background-color: #121212; border-radius: 5px; color: white;")
            else:
                self.res_muzzle_v.setText("Muzzle Velocity:\nN/A")
                self.res_firing_t.setText("Firing Time:\nN/A")
                self.res_exp_comp.setText("Error vs Exp:\nN/A")
                self.res_exp_comp.setStyleSheet("font-size: 15px; padding: 10px; background-color: #121212; border-radius: 5px; color: white;")
            
            # Plottings
            # Plot 1: Pressure
            ax_p = self.canvas_pressure.axes[0]
            ax_p.clear()
            ax_p.plot(t_arr, p1_arr, label="Chamber Pressure", color='#FF5252', linewidth=2.5)
            ax_p.plot(t_arr, p2_arr, label="Light Gas Pressure", color='#00ADB5', linewidth=2.5)
            ax_p.axhline(y=470, color='#FF5252', linestyle=':', label='Burst Threshold (470 bar)', linewidth=2)
            ax_p.set_title("Pressure vs Time")
            ax_p.set_xlabel("Time (ms)")
            ax_p.set_ylabel("Pressure (bar)")
            ax_p.legend(facecolor='#1E1E2E', edgecolor='#333344', labelcolor='white')
            ax_p.grid(color='#333344', linestyle='--')
            self.canvas_pressure.draw()
            
            # Plot 2: Velocity
            ax_v = self.canvas_velocity.axes[0]
            ax_v.clear()
            ax_v.plot(t_arr, v_s2, label="Piston Velocity", color='#e040fb', linewidth=2.5)
            ax_v.plot(t_arr, v_s3, label="Projectile Velocity", color='#4CAF50', linewidth=2.5)
            ax_v.set_title("Velocity vs Time")
            ax_v.set_xlabel("Time (ms)")
            ax_v.set_ylabel("Velocity (m/s)")
            ax_v.legend(facecolor='#1E1E2E', edgecolor='#333344', labelcolor='white')
            ax_v.grid(color='#333344', linestyle='--')
            self.canvas_velocity.draw()
            
            # Plot 3: Acceleration
            ax_a = self.canvas_accel.axes[0]
            ax_a.clear()
            ax_a.plot(t_arr, a_s3, label="Projectile Acceleration", color='#ffd740', linewidth=2.5)
            ax_a.set_title("Acceleration vs Time")
            ax_a.set_xlabel("Time (ms)")
            ax_a.set_ylabel("Acceleration (m/s²)")
            ax_a.legend(facecolor='#1E1E2E', edgecolor='#333344', labelcolor='white')
            ax_a.grid(color='#333344', linestyle='--')
            self.canvas_accel.draw()

            # Switch to RESULTS tab
            self.main_tabs.setCurrentIndex(2)

        except Exception as e:
            QMessageBox.critical(self, "Simulation Error", str(e))
            self.lbl_sim_status.setText("Status: Error")
            self.lbl_sim_status.setStyleSheet("font-size: 24px; color: #FF5252;")
        finally:
            self.btn_run.setEnabled(True)
            self.btn_run.setText("RUN SIMULATION")

    def build_tab_comparison(self):
        layout = QVBoxLayout(self.tab_compare)
        
        # TOP SECTION
        top_layout = QHBoxLayout()
        title = QLabel("Multi-Gas Performance Comparison")
        title.setStyleSheet("font-size: 20px; color: #00ADB5; font-weight: bold;")
        
        self.lbl_compare_status = QLabel("Ready")
        self.lbl_compare_status.setStyleSheet("font-size: 16px; color: #aaaaaa;")
        
        self.btn_compare = QPushButton("RUN COMPARISON")
        self.btn_compare.setFixedWidth(200)
        self.btn_compare.clicked.connect(self.run_gas_comparison)
        
        top_layout.addWidget(title)
        top_layout.addStretch()
        top_layout.addWidget(self.lbl_compare_status)
        top_layout.addSpacing(20)
        top_layout.addWidget(self.btn_compare)
        layout.addLayout(top_layout)
        
        # MIDDLE SECTION (Cards)
        cards_layout = QHBoxLayout()
        self.compare_cards = {}
        for gas_name, color in [("Hydrogen", "#00ADB5"), ("Helium", "#FFD369"), ("Nitrogen", "#9D4EDD")]:
            card = QGroupBox(gas_name)
            card.setStyleSheet(f"""
                QGroupBox {{ border: 2px solid {color}; border-radius: 8px; background-color: #1E1E2E; margin-top: 15px; margin-bottom: 5px; }}
                QGroupBox::title {{ color: {color}; font-weight: bold; background-color: transparent; }}
            """)
            card_layout = QVBoxLayout(card)
            lbl_v = QLabel("Velocity:\n--")
            lbl_p = QLabel("Peak Pressure (P2):\n--")
            lbl_bt = QLabel("Burst Time:\n--")
            lbl_ft = QLabel("Firing Time:\n--")
            for lbl in [lbl_v, lbl_p, lbl_bt, lbl_ft]:
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet("font-size: 14px; padding: 2px;")
                card_layout.addWidget(lbl)
                
            self.compare_cards[gas_name] = { "v": lbl_v, "p": lbl_p, "bt": lbl_bt, "ft": lbl_ft }
            cards_layout.addWidget(card)
            
        layout.addLayout(cards_layout)
        
        # BOTTOM SECTION (Graphs)
        self.canvas_compare = CompareCanvas(self, width=10, height=8, dpi=100)
        layout.addWidget(self.canvas_compare)

    def run_gas_comparison(self):
        try:
            params = {
                'L1': float(self.inp_L1.text()),
                'D1': float(self.inp_D1.text()),
                'm1': float(self.inp_m1.text()),
                'f': float(self.inp_f.text()),
                'alpha': float(self.inp_alpha.text()),
                'beta': float(self.inp_beta.text()),
                'viva': float(self.inp_viva.text()),
                'mp': float(self.inp_mp.text()),
                'D2': float(self.inp_D2.text()),
                'L2': float(self.inp_L2.text()),
                'P0': float(self.inp_P0.text()) * 1e5,
                'mproj': float(self.inp_mproj.text()),
                'L3': float(self.inp_L3.text()),
                'D3': float(self.inp_D3.text())
            }
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please ensure all numerical inputs are valid.")
            return

        self.btn_compare.setEnabled(False)
        self.btn_compare.setText("COMPARING...")
        
        def update_status(gas_name):
            self.lbl_compare_status.setText(f"Running {gas_name}...")
            self.lbl_compare_status.setStyleSheet("font-size: 16px; color: #FFD369;")
            QApplication.processEvents()
            
        try:
            results = compare_all_gases(params, progress_callback=update_status)
            
            self.lbl_compare_status.setText("Completed")
            self.lbl_compare_status.setStyleSheet("font-size: 16px; color: #4CAF50;")
            
            # Update UI Cards
            for gas_name, res in results.items():
                c = self.compare_cards[gas_name]
                c['v'].setText(f"Velocity:\n{res['velocity']:.2f} m/s")
                c['p'].setText(f"Peak Pressure (P2):\n{res['peak_pressure']:.2f} bar")
                c['bt'].setText(f"Burst Time:\n{res['burst_time']:.3f} ms" if res['burst_time'] > 0 else "Burst Time:\nDid not burst")
                c['ft'].setText(f"Firing Time:\n{res['firing_time']:.3f} ms" if res['firing_time'] > 0 else "Firing Time:\nN/A")
                
            # Plot Graphs
            ax_v, ax_p, ax_t = self.canvas_compare.ax_v, self.canvas_compare.ax_p, self.canvas_compare.ax_t
            ax_v.clear(); ax_p.clear(); ax_t.clear()
            
            gases = ["Hydrogen", "Helium", "Nitrogen"]
            colors = ["#00ADB5", "#FFD369", "#9D4EDD"]
            
            vels = [results[g]['velocity'] for g in gases]
            pressures = [results[g]['peak_pressure'] for g in gases]
            
            bars_v = ax_v.bar(gases, vels, color=colors)
            ax_v.set_title("Velocity Comparison")
            ax_v.set_ylabel("Muzzle Velocity (m/s)")
            ax_v.bar_label(bars_v, fmt='%.0f', color='white')
            
            bars_p = ax_p.bar(gases, pressures, color=colors)
            ax_p.set_title("Peak Pressure Comparison")
            ax_p.set_ylabel("Peak P2 (bar)")
            ax_p.bar_label(bars_p, fmt='%.1f', color='white')
            
            for i, gas in enumerate(gases):
                t_arr = results[gas]['t_arr']
                p2_arr = results[gas]['p2_arr']
                ax_t.plot(t_arr, p2_arr, label=gas, color=colors[i], linewidth=2.5)
                
            ax_t.set_title("Light Gas Pressure vs Time")
            ax_t.set_xlabel("Time (ms)")
            ax_t.set_ylabel("Pressure P2 (bar)")
            ax_t.legend(facecolor='#1E1E2E', edgecolor='#333344', labelcolor='white')
            ax_t.grid(color='#333344', linestyle='--')
            
            # Apply grid to bar charts too
            for ax in [ax_v, ax_p]:
                ax.grid(color='#333344', linestyle='--', axis='y')
                
            self.canvas_compare.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Comparison Error", str(e))
            self.lbl_compare_status.setText("Error")
            self.lbl_compare_status.setStyleSheet("font-size: 16px; color: #FF5252;")
        finally:
            self.btn_compare.setEnabled(True)
            self.btn_compare.setText("RUN COMPARISON")

    def build_tab_sweep(self):
        layout = QVBoxLayout(self.tab_sweep)
        
        # TOP ROW (Inputs)
        top_layout = QHBoxLayout()
        title = QLabel("Parameter Sweep Analysis")
        title.setStyleSheet("font-size: 20px; color: #00ADB5; font-weight: bold;")
        top_layout.addWidget(title)
        top_layout.addSpacing(30)
        
        self.cmb_sweep_param = QComboBox()
        self.cmb_sweep_param.addItem("VIVA")
        
        self.inp_sweep_start = QLineEdit("1000")
        self.inp_sweep_end = QLineEdit("4000")
        self.inp_sweep_steps = QLineEdit("10")
        
        for w in [self.inp_sweep_start, self.inp_sweep_end, self.inp_sweep_steps]:
            w.setFixedWidth(80)
            
        form_layout = QHBoxLayout()
        form_layout.addWidget(QLabel("Parameter:"))
        form_layout.addWidget(self.cmb_sweep_param)
        form_layout.addSpacing(10)
        form_layout.addWidget(QLabel("Start:"))
        form_layout.addWidget(self.inp_sweep_start)
        form_layout.addSpacing(10)
        form_layout.addWidget(QLabel("End:"))
        form_layout.addWidget(self.inp_sweep_end)
        form_layout.addSpacing(10)
        form_layout.addWidget(QLabel("Steps:"))
        form_layout.addWidget(self.inp_sweep_steps)
        
        top_layout.addLayout(form_layout)
        top_layout.addStretch()
        
        self.btn_run_sweep = QPushButton("RUN SWEEP")
        self.btn_run_sweep.clicked.connect(self.run_parameter_sweep)
        top_layout.addWidget(self.btn_run_sweep)
        layout.addLayout(top_layout)
        
        # MIDDLE ROW
        mid_layout = QHBoxLayout()
        self.lbl_sweep_status = QLabel("Status: Ready")
        self.lbl_sweep_status.setStyleSheet("color: #aaaaaa;")
        mid_layout.addWidget(self.lbl_sweep_status)
        mid_layout.addStretch()
        layout.addLayout(mid_layout)
        
        # BOTTOM ROW
        self.canvas_sweep = SweepCanvas(self, width=10, height=6, dpi=100)
        layout.addWidget(self.canvas_sweep)
        
        # FOOTER
        foot_layout = QHBoxLayout()
        self.lbl_sweep_max = QLabel("Max Muzzle Velocity: --")
        self.lbl_sweep_min = QLabel("Min Muzzle Velocity: --")
        self.lbl_sweep_max.setStyleSheet("font-size: 16px; color: #4CAF50; padding: 10px; background-color: #1E1E2E; border-radius: 6px;")
        self.lbl_sweep_min.setStyleSheet("font-size: 16px; color: #FF5252; padding: 10px; background-color: #1E1E2E; border-radius: 6px;")
        
        foot_layout.addWidget(self.lbl_sweep_max)
        foot_layout.addSpacing(20)
        foot_layout.addWidget(self.lbl_sweep_min)
        foot_layout.addStretch()
        layout.addLayout(foot_layout)

    def run_parameter_sweep(self):
        try:
            start_val = float(self.inp_sweep_start.text())
            end_val = float(self.inp_sweep_end.text())
            steps = int(self.inp_sweep_steps.text())
            
            params = {
                'L1': float(self.inp_L1.text()),
                'D1': float(self.inp_D1.text()),
                'm1': float(self.inp_m1.text()),
                'f': float(self.inp_f.text()),
                'alpha': float(self.inp_alpha.text()),
                'beta': float(self.inp_beta.text()),
                'viva': float(self.inp_viva.text()),
                'mp': float(self.inp_mp.text()),
                'D2': float(self.inp_D2.text()),
                'L2': float(self.inp_L2.text()),
                'P0': float(self.inp_P0.text()) * 1e5,
                'mproj': float(self.inp_mproj.text()),
                'L3': float(self.inp_L3.text()),
                'D3': float(self.inp_D3.text())
            }
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Ensure all numerical inputs are valid.")
            return

        gas_choice = self.cmb_gas.currentText()
        gamma = 1.66 if "Helium" in gas_choice else 1.4
            
        self.btn_run_sweep.setEnabled(False)
        self.btn_run_sweep.setText("RUNNING...")
        self.lbl_sweep_max.setText("Max Muzzle Velocity: --")
        self.lbl_sweep_min.setText("Min Muzzle Velocity: --")
        
        def update_progress(current, total, viva_val):
            pct = int((current / total) * 100)
            self.lbl_sweep_status.setText(f"Status: Running... ({pct}%) -> VIVA={viva_val:.1f}")
            self.lbl_sweep_status.setStyleSheet("color: #FFD369;")
            QApplication.processEvents()

        try:
            results = run_sweep(params, gamma, start_val, end_val, steps, update_progress)
            
            self.lbl_sweep_status.setText("Status: Completed")
            self.lbl_sweep_status.setStyleSheet("color: #4CAF50;")
            
            vivas = [r['viva'] for r in results]
            vels = [r['velocity'] for r in results]
            
            # Find max/min
            valid_results = [r for r in results if r['velocity'] > 0]
            if len(valid_results) > 0:
                max_res = max(valid_results, key=lambda x: x['velocity'])
                min_res = min(valid_results, key=lambda x: x['velocity'])
                
                self.lbl_sweep_max.setText(f"Max Muzzle Velocity: {max_res['velocity']:.2f} m/s @ VIVA {max_res['viva']:.1f}")
                self.lbl_sweep_min.setText(f"Min Muzzle Velocity: {min_res['velocity']:.2f} m/s @ VIVA {min_res['viva']:.1f}")
            else:
                self.lbl_sweep_max.setText("Max Muzzle Velocity: No valid firings")
                self.lbl_sweep_min.setText("Min Muzzle Velocity: No valid firings")
                max_res = None
                min_res = None
            
            ax = self.canvas_sweep.ax
            ax.clear()
            ax.plot(vivas, vels, color='#00ADB5', linewidth=2.5, marker='o', label='Muzzle Velocity')
            
            if len(valid_results) > 0:
                ax.scatter([max_res['viva']], [max_res['velocity']], color='#4CAF50', s=150, zorder=5, label='Maximum')
                ax.scatter([min_res['viva']], [min_res['velocity']], color='#FF5252', s=150, zorder=5, label='Minimum')
                
            ax.set_title("VIVA vs Muzzle Velocity")
            ax.set_xlabel("VIVA")
            ax.set_ylabel("Velocity (m/s)")
            ax.grid(color='#333344', linestyle='--')
            ax.legend(facecolor='#1E1E2E', edgecolor='#333344', labelcolor='white')
            
            self.canvas_sweep.draw()

        except Exception as e:
            QMessageBox.critical(self, "Sweep Error", str(e))
            self.lbl_sweep_status.setText("Status: Error")
            self.lbl_sweep_status.setStyleSheet("color: #FF5252;")
        finally:
            self.btn_run_sweep.setEnabled(True)
            self.btn_run_sweep.setText("RUN SWEEP")

    def build_tab_animation(self):
        layout = QVBoxLayout(self.tab_animation)
        
        # TOP ROW
        top_layout = QHBoxLayout()
        title = QLabel("2D Real-Time Simulation Panel")
        title.setStyleSheet("font-size: 20px; color: #00ADB5; font-weight: bold;")
        top_layout.addWidget(title)
        
        self.lbl_anim_status = QLabel("Status: Awaiting Simulation Run...")
        self.lbl_anim_status.setStyleSheet("color: #aaaaaa; font-size: 16px;")
        top_layout.addStretch()
        top_layout.addWidget(self.lbl_anim_status)
        layout.addLayout(top_layout)
        
        # MIDDLE ROW (Controls)
        ctrl_layout = QHBoxLayout()
        self.btn_anim_play = QPushButton("▶ PLAY")
        self.btn_anim_pause = QPushButton("⏸ PAUSE")
        self.btn_anim_reset = QPushButton("🔄 RESET")
        
        self.cmb_anim_speed = QComboBox()
        self.cmb_anim_speed.addItems(["0.5x", "1x", "2x", "5x"])
        self.cmb_anim_speed.setCurrentText("1x")
        
        for btn in [self.btn_anim_play, self.btn_anim_pause, self.btn_anim_reset]:
            btn.setFixedWidth(120)
            ctrl_layout.addWidget(btn)
            
        self.btn_anim_play.clicked.connect(self.anim_play)
        self.btn_anim_pause.clicked.connect(self.anim_pause)
        self.btn_anim_reset.clicked.connect(self.anim_reset)
        
        ctrl_layout.addSpacing(20)
        lbl_speed = QLabel("Speed:")
        lbl_speed.setStyleSheet("color: white;")
        ctrl_layout.addWidget(lbl_speed)
        self.cmb_anim_speed.setFixedWidth(60)
        ctrl_layout.addWidget(self.cmb_anim_speed)
        
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)
        
        # BOTTOM ROW (Canvas)
        self.canvas_anim = AnimationCanvas(self, width=10, height=4, dpi=100)
        layout.addWidget(self.canvas_anim)
        
        layout.addStretch()

    def anim_play(self):
        if not self.anim_data_xp:
            QMessageBox.warning(self, "No Data", "Please run a simulation first!")
            return
        if self.anim_index >= len(self.anim_data_xp):
            self.anim_index = 0
        self.anim_timer.start(30)
        self.lbl_anim_status.setText("Status: Playing...")
        self.lbl_anim_status.setStyleSheet("color: #FFD369;")

    def anim_pause(self):
        self.anim_timer.stop()
        self.lbl_anim_status.setText("Status: Paused")
        self.lbl_anim_status.setStyleSheet("color: #FF5252;")

    def anim_reset(self):
        self.anim_timer.stop()
        self.anim_index = 0
        if self.anim_data_xp:
            self.canvas_anim.update_positions(
                self.anim_data_xp[0], self.anim_data_xproj[0], self.anim_L2, self.anim_L3,
                self.anim_data_t[0], self.anim_data_p[0], self.anim_data_v[0]
            )
            self.lbl_anim_status.setText("Status: Ready to Play")
            self.lbl_anim_status.setStyleSheet("color: #4CAF50;")

    def anim_step(self):
        if self.anim_index < len(self.anim_data_xp):
            xp = self.anim_data_xp[self.anim_index]
            xproj = self.anim_data_xproj[self.anim_index]
            t = self.anim_data_t[self.anim_index]
            p = self.anim_data_p[self.anim_index]
            v = self.anim_data_v[self.anim_index]
            
            self.canvas_anim.update_positions(xp, xproj, self.anim_L2, self.anim_L3, t, p, v)
            
            speed_str = self.cmb_anim_speed.currentText().replace("x", "")
            speed = float(speed_str)
            base_step = max(1, len(self.anim_data_xp) // 300) 
            self.anim_index += max(1, int(base_step * speed))
        else:
            self.anim_timer.stop()
            self.lbl_anim_status.setText("Status: Simulation End Reached")
            self.lbl_anim_status.setStyleSheet("color: #00ADB5;")

