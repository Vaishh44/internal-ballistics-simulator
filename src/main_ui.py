import sys
import math
import numpy as np
import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QGridLayout, QGroupBox, 
                             QMessageBox, QTabWidget, QComboBox, QApplication,
                             QScrollArea, QFrame, QTextEdit, QSizePolicy)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PyQt5.QtCore import Qt, QTimer
import matplotlib.patches as patches

from src.powder_model import PowderModel
from src.gas_model import GasModel
from src.shot_parameters import ShotParameters
from src.multi_zone_solver import MultiZoneSolver

DARK_THEME_QSS = """
QMainWindow { background-color: #0d1117; }
QWidget { background-color: #0d1117; font-family: 'Segoe UI', Arial, sans-serif; }

QTabWidget::pane { border: 1px solid #30363d; background-color: #161b22; border-radius: 4px; }
QTabBar::tab { background: #161b22; color: #8b949e; padding: 12px 28px; border: 1px solid #30363d; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 4px; font-size: 16px; font-weight: bold; }
QTabBar::tab:selected { background: #0d1117; color: #58a6ff; border-top: 3px solid #58a6ff; }

QLabel { color: #B0BEC5; font-size: 13px; }
QLabel#UnitLabel { color: #78909C; font-size: 11px; }
QLabel#CardTitle { color: #58a6ff; font-size: 18px; font-weight: bold; padding-bottom: 5px; }

QLineEdit, QComboBox { 
    background-color: #0d1117; 
    border: 1px solid #30363d; 
    padding: 8px; 
    border-radius: 4px; 
    color: #FFFFFF; 
    font-family: Consolas, monospace; 
    font-size: 14px; 
    font-weight: 600; 
}
QLineEdit:focus, QComboBox:focus { border: 1px solid #58a6ff; }

QFrame#SetupCard, QFrame#ReportCard {
    background-color: #161b22; 
    border: 1px solid #30363d; 
    border-radius: 8px; 
    padding: 10px;
}

QPushButton { background-color: #238636; border: 1px solid #2ea043; border-radius: 6px; padding: 12px 24px; font-weight: bold; color: #ffffff; font-size: 16px; }
QPushButton:hover { background-color: #2ea043; }
QPushButton:disabled { background-color: #21262d; border: 1px solid #30363d; color: #8b949e; }

QPushButton#btnReset { background-color: #21262d; border: 1px solid #30363d; color: #B0BEC5; padding: 8px 15px; font-size: 14px;}
QPushButton#btnReset:hover { background-color: #30363d; border: 1px solid #8b949e; color: white;}

QScrollArea { border: none; background-color: transparent; }
"""

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100, num_subplots=1, sparkline=False):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.patch.set_facecolor('#161b22')
        self.axes = []
        for i in range(num_subplots):
            ax = fig.add_subplot(num_subplots, 1, i+1)
            ax.set_facecolor('#0d1117')
            if not sparkline:
                ax.tick_params(colors='#B0BEC5', labelsize=12)
                ax.xaxis.label.set_color('#B0BEC5')
                ax.yaxis.label.set_color('#B0BEC5')
                ax.xaxis.label.set_fontsize(14)
                ax.yaxis.label.set_fontsize(14)
                ax.title.set_color('#58a6ff')
                ax.title.set_fontsize(16)
                ax.title.set_weight('bold')
                ax.grid(color='#30363d', linestyle='-', linewidth=0.5, alpha=0.5)
                for spine in ax.spines.values():
                    spine.set_color('#30363d')
            else:
                ax.axis('off')
                fig.patch.set_facecolor('#161b22')
                ax.set_facecolor('#161b22')
            self.axes.append(ax)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        if sparkline:
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        else:
            fig.tight_layout(pad=3.0)

class InteractiveCanvas(MatplotlibCanvas):
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        super().__init__(parent, width, height, dpi, num_subplots=1, sparkline=False)
        self.annotations = []
        for ax in self.axes:
            annot = ax.annotate("", xy=(0,0), xytext=(15, 15), textcoords="offset points",
                                bbox=dict(boxstyle="round4", fc="#21262d", ec="#58a6ff", alpha=0.9),
                                color="#FFFFFF", fontsize=11, zorder=20)
            annot.set_visible(False)
            self.annotations.append(annot)
            
        self.mpl_connect("motion_notify_event", self.on_hover)

    def on_hover(self, event):
        vis_changed = False
        for i, ax in enumerate(self.axes):
            annot = self.annotations[i]
            if event.inaxes == ax:
                closest_line = None
                closest_point = None
                min_dist = float('inf')
                
                for line in ax.get_lines():
                    xdata = line.get_xdata()
                    ydata = line.get_ydata()
                    if len(xdata) == 0: continue
                    
                    xy_data = np.column_stack((xdata, ydata))
                    xy_disp = ax.transData.transform(xy_data)
                    event_disp = np.array([event.x, event.y])
                    
                    dists = np.linalg.norm(xy_disp - event_disp, axis=1)
                    min_idx = np.argmin(dists)
                    min_d = dists[min_idx]
                    
                    if min_d < 20 and min_d < min_dist:  
                        min_dist = min_d
                        closest_line = line
                        closest_point = (xdata[min_idx], ydata[min_idx])
                
                if closest_line:
                    annot.xy = closest_point
                    lbl = closest_line.get_label()
                    if not lbl.startswith('_'):
                        annot.set_text(f"[{lbl}]\nValue: {closest_point[1]:.1f}\nTime: {closest_point[0]:.2f} ms")
                        if not annot.get_visible():
                            annot.set_visible(True)
                            vis_changed = True
                        else:
                            vis_changed = True  
                else:
                    if annot.get_visible():
                        annot.set_visible(False)
                        vis_changed = True
            else:
                if annot.get_visible():
                    annot.set_visible(False)
                    vis_changed = True

        if vis_changed:
            self.draw_idle()

class AnimationCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.patch.set_facecolor('#161b22')
        self.ax = fig.add_subplot(1, 1, 1)
        self.ax.set_facecolor('#0d1117')
        self.ax.tick_params(colors='#8b949e', labelsize=12)
        for spine in self.ax.spines.values():
            spine.set_color('#30363d')
            
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        fig.tight_layout()

        self.piston_patch = None
        self.proj_patch = None
        self.ch_patch = None
        self.pump_patch = None
        self.launch_patch = None
        self.barrier_line = None
        
        self.cmap = cm.get_cmap('coolwarm')
        self.norm = mcolors.Normalize(vmin=1.0, vmax=4500.0)

    def setup_scene(self, L1, L2, L3, D1, D2, D3):
        self.ax.clear()
        
        total_L = L1 + L2 + L3
        max_D = max(D1, max(D2, D3))
        self.ax.set_ylim(-max_D*4.5, max_D*4.5)
        self.ax.set_xlim(-L1 - 0.5, total_L - L1 + 0.5)
        
        self.ax.spines['bottom'].set_position(('data', -max_D*3))
        self.ax.spines['top'].set_color('none')
        self.ax.spines['left'].set_color('none')
        self.ax.spines['right'].set_color('none')
        self.ax.set_yticks([])
        
        ch_top, ch_bot = D1/2, -D1/2
        self.ax.plot([-L1, 0], [ch_top, ch_top], color='#B0BEC5', lw=3)
        self.ax.plot([-L1, 0], [ch_bot, ch_bot], color='#B0BEC5', lw=3)
        self.ch_patch = patches.Rectangle((-L1, ch_bot), L1, D1, color=self.cmap(self.norm(1.0)), alpha=0.7, zorder=2)
        self.ax.add_patch(self.ch_patch)
        self.ax.text(-L1/2, ch_top*1.2, "Chamber", color='#B0BEC5', ha='center', va='bottom', fontsize=12, weight='bold')
        
        p_top, p_bot = D2/2, -D2/2
        self.ax.plot([0, L2], [p_top, p_top], color='#8b949e', lw=3)
        self.ax.plot([0, L2], [p_bot, p_bot], color='#8b949e', lw=3)
        self.pump_patch = patches.Rectangle((0, p_bot), L2, D2, color=self.cmap(self.norm(1.0)), alpha=0.7, zorder=2)
        self.ax.add_patch(self.pump_patch)
        self.ax.text(L2/2, p_top*1.2, "Pump Tube", color='#B0BEC5', ha='center', va='bottom', fontsize=12, weight='bold')
        
        l_top, l_bot = D3/2, -D3/2
        self.ax.plot([L2, L2 + L3], [l_top, l_top], color='#484f58', lw=3)
        self.ax.plot([L2, L2 + L3], [l_bot, l_bot], color='#484f58', lw=3)
        self.launch_patch = patches.Rectangle((L2, l_bot), L3, D3, color=self.cmap(self.norm(1.0)), alpha=0.7, zorder=2)
        self.ax.add_patch(self.launch_patch)
        self.ax.text(L2 + L3/2, l_top*1.2, "Launch Tube", color='#B0BEC5', ha='center', va='bottom', fontsize=12, weight='bold')
        
        p_len = max(0.2, min(L2 * 0.08, 1.0))
        self.piston_len = p_len
        self.piston_patch = patches.Rectangle((-p_len, p_bot*1.2), p_len, D2*1.2, color='#1f6feb', zorder=5)
        self.ax.add_patch(self.piston_patch)
        
        proj_rad = max(D3, 0.05)
        self.proj_patch = patches.Rectangle((L2, l_bot*1.2), proj_rad*2, D3*1.2, color='#2ea043', zorder=6)
        self.ax.add_patch(self.proj_patch)
        
        self.barrier_line = self.ax.vlines(L2, p_bot*1.5, p_top*1.5, colors='#d29922', linestyles='solid', lw=4, zorder=4)

        self.draw_idle()

    def update_positions(self, x_p, x_proj, L1, L2, L3, p1, p2, p3):
        self.ch_patch.set_color(self.cmap(self.norm(p1)))
        self.pump_patch.set_color(self.cmap(self.norm(p2)))
        self.launch_patch.set_color(self.cmap(self.norm(p3)))
        
        self.piston_patch.set_x(x_p - self.piston_len)
        self.ch_patch.set_width(L1 + x_p)
        self.pump_patch.set_x(x_p)
        
        curr_proj_x = L2 + x_proj
        self.proj_patch.set_x(curr_proj_x)
        self.pump_patch.set_width(curr_proj_x - x_p)
        
        self.launch_patch.set_x(curr_proj_x)
        self.launch_patch.set_width(max(0, L2 + L3 - curr_proj_x))
        
        if p2 >= 500 or x_proj > 0:
            self.barrier_line.set_color('#2ea043')
            self.barrier_line.set_linestyle('--')
            
        self.draw_idle()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Two-Stage Light Gas Gun Simulator - Professional Edition")
        self.setGeometry(50, 50, 1680, 1000)
        self.setStyleSheet(DARK_THEME_QSS)
        
        self.res_labels = {}
        self.res_sparks = {}
        
        self.raw_report_text = ""
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        self.main_tabs = QTabWidget()
        main_layout.addWidget(self.main_tabs)
        
        self.tab_setup = QWidget(); self.build_tab_setup(); self.main_tabs.addTab(self.tab_setup, "SETUP")
        self.tab_sim = QWidget(); self.build_tab_simulation(); self.main_tabs.addTab(self.tab_sim, "SIMULATION")
        self.tab_res = QWidget(); self.build_tab_results(); self.main_tabs.addTab(self.tab_res, "RESULTS")
        self.tab_ana = QWidget(); self.build_tab_analysis(); self.main_tabs.addTab(self.tab_ana, "ANALYSIS")
        self.tab_comp = QWidget(); self.build_tab_comparison(); self.main_tabs.addTab(self.tab_comp, "COMPARISON")
        
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.anim_step)
        self.anim_index = 0
        self.anim_data = {}

    def create_setup_card(self, title):
        card = QFrame()
        card.setObjectName("SetupCard")
        l = QVBoxLayout(card)
        l.setContentsMargins(20, 20, 20, 20)
        
        t_lbl = QLabel(title)
        t_lbl.setObjectName("CardTitle")
        l.addWidget(t_lbl)
        
        grid_w = QWidget()
        grid = QGridLayout(grid_w)
        grid.setSpacing(15)
        grid.setColumnMinimumWidth(0, 180)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(2, 60)
        l.addWidget(grid_w)
        return card, grid
        
    def add_input_row(self, layout, row_idx, label, unit, default_val, readonly=False):
        lbl = QLabel(label)
        u_lbl = QLabel(f"{unit}" if unit else "")
        u_lbl.setObjectName("UnitLabel")
        u_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        if default_val == "dropdown":
            inp = QComboBox()
            inp.addItems(["Hydrogen", "Helium", "Nitrogen"])
            inp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            layout.addWidget(lbl, row_idx, 0)
            layout.addWidget(inp, row_idx, 1)
            layout.addWidget(u_lbl, row_idx, 2)
            return inp
        else:
            inp = QLineEdit(default_val)
            inp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            inp.setAlignment(Qt.AlignLeft)
            
            if readonly:
                inp.setReadOnly(True)
                inp.setStyleSheet("background-color: #21262d; color: #8b949e; border: 1px solid #30363d; padding: 8px; border-radius: 4px; font-family: Consolas, monospace; font-size: 14px; font-weight: 600;")
                inp.setToolTip("This parameter is automatically set based on physics model")
                
            layout.addWidget(lbl, row_idx, 0)
            layout.addWidget(inp, row_idx, 1)
            layout.addWidget(u_lbl, row_idx, 2)
            return inp

    def compute_bounds(self):
        try:
            D3 = float(self.inp_D3.text())
            a_val = math.pi * (D3 / 2)**2
            self.inp_a_valve.setText(f"{a_val:.7f}")
        except:
            pass
            
    def update_gas_properties(self):
        gas = self.cmb_gas.currentText()
        if "Hydrogen" in gas:
            self.inp_gamma.setText("1.4")
            self.inp_cp.setText("14300")
            self.inp_cv.setText("10200")
        elif "Helium" in gas:
            self.inp_gamma.setText("1.66")
            self.inp_cp.setText("5192")
            self.inp_cv.setText("3115")
        else:
            self.inp_gamma.setText("1.4")
            self.inp_cp.setText("1040")
            self.inp_cv.setText("743")

    def build_tab_setup(self):
        layout = QVBoxLayout(self.tab_setup)
        layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        view_container = QWidget()
        view_layout = QVBoxLayout(view_container)
        view_layout.setContentsMargins(40, 40, 40, 40)
        view_layout.setSpacing(25)
        
        banner_h = QHBoxLayout()
        lbl_in = QLabel("SIMULATION PARAMETERS")
        lbl_in.setStyleSheet("font-size: 22px; color: #FFFFFF; font-weight: bold; padding-bottom: 10px;")
        banner_h.addWidget(lbl_in)
        banner_h.addStretch()
        
        btn_reset = QPushButton("Reset to Default Safe Values")
        btn_reset.setObjectName("btnReset")
        btn_reset.clicked.connect(self.reset_defaults)
        banner_h.addWidget(btn_reset)
        view_layout.addLayout(banner_h)
        
        card1, l1 = self.create_setup_card("🔷 1. Geometry Specs")
        r=0
        self.inp_L1 = self.add_input_row(l1, r, "Chamber Length", "m", "0.44"); r+=1
        self.inp_D1 = self.add_input_row(l1, r, "Chamber Diameter", "m", "0.1646"); r+=1
        self.inp_L2 = self.add_input_row(l1, r, "Pump Tube Length", "m", "12.0"); r+=1
        self.inp_D2 = self.add_input_row(l1, r, "Pump Tube Diameter", "m", "0.10"); r+=1
        self.inp_L3 = self.add_input_row(l1, r, "Launch Tube Length", "m", "8.0"); r+=1
        self.inp_D3 = self.add_input_row(l1, r, "Launch Tube Diameter", "m", "0.036"); r+=1
        self.inp_D3.textChanged.connect(self.compute_bounds)
        view_layout.addWidget(card1)
        
        card2, l2 = self.create_setup_card("🔷 2. Powder Thermodynamics")
        r=0
        self.inp_num_perfs = self.add_input_row(l2, r, "Perforations", "-", "7"); r+=1
        self.inp_outer_rad = self.add_input_row(l2, r, "Outer Radius", "m", "0.004"); r+=1
        self.inp_inner_rad = self.add_input_row(l2, r, "Inner Radius", "m", "0.0003"); r+=1
        self.inp_grain_len = self.add_input_row(l2, r, "Grain Length", "m", "0.012"); r+=1
        self.inp_web_thick = self.add_input_row(l2, r, "Web Thickness", "m", "0.0015"); r+=1
        self.inp_density = self.add_input_row(l2, r, "Density", "kg/m³", "1600"); r+=1
        self.inp_flame_temp = self.add_input_row(l2, r, "Flame Temp", "K", "2800"); r+=1
        self.inp_energy = self.add_input_row(l2, r, "Energy", "J/kg", "9.0e5"); r+=1
        self.inp_poly_ratio = self.add_input_row(l2, r, "Polytropic Ratio", "-", "1.25"); r+=1
        self.inp_covolume = self.add_input_row(l2, r, "Co-volume", "-", "0.001"); r+=1
        self.inp_shape_coeffs = self.add_input_row(l2, r, "Coeffs (A-F)", "-", "0.1, 0.5, 0.2, 0.1, 0.05, 0.05"); r+=1
        self.inp_viva = self.add_input_row(l2, r, "VIVA", "-", "2000"); r+=1
        self.inp_alpha = self.add_input_row(l2, r, "Alpha", "-", "0.8"); r+=1
        self.inp_f = self.add_input_row(l2, r, "Specific Force", "J/kg", "9e5"); r+=1
        self.inp_m1 = self.add_input_row(l2, r, "Powder Mass", "kg", "1.2"); r+=1
        view_layout.addWidget(card2)
        
        card3, l3 = self.create_setup_card("🔷 3. Working Mechanics")
        r=0
        self.cmb_gas = self.add_input_row(l3, r, "Working Gas Type", "-", "dropdown"); r+=1
        self.cmb_gas.currentIndexChanged.connect(self.update_gas_properties)
        self.inp_gamma = self.add_input_row(l3, r, "Gamma (γ)", "-", "1.4", readonly=True); r+=1
        self.inp_cp = self.add_input_row(l3, r, "Isobaric Heat Cap (Cp)", "J/kgK", "14300", readonly=True); r+=1
        self.inp_cv = self.add_input_row(l3, r, "Isochoric Heat Cap (Cv)", "J/kgK", "10200", readonly=True); r+=1
        self.inp_mproj = self.add_input_row(l3, r, "Projectile Mass", "kg", "0.03"); r+=1
        self.inp_mp = self.add_input_row(l3, r, "Piston Mass", "kg", "4.0"); r+=1
        self.inp_P0 = self.add_input_row(l3, r, "Initial Gas Press", "bar", "20"); r+=1
        view_layout.addWidget(card3)
        
        card4, l4 = self.create_setup_card("🔷 4. Advance Flags & Friction")
        r=0
        self.inp_eta = self.add_input_row(l4, r, "Efficiency Factor", "-", "0.6"); r+=1
        self.inp_coeff_p = self.add_input_row(l4, r, "Piston Friction", "-", "0.02"); r+=1
        self.inp_coeff_proj = self.add_input_row(l4, r, "Proj Friction", "-", "0.01"); r+=1
        self.inp_p_residual = self.add_input_row(l4, r, "Resid Gas Pressure", "bar", "1.0"); r+=1
        self.inp_p_burst = self.add_input_row(l4, r, "Valve Burst Press", "bar", "500"); r+=1
        self.inp_valve_delay = self.add_input_row(l4, r, "Valve Delay", "µs", "50"); r+=1
        self.inp_cd = self.add_input_row(l4, r, "Valve Fluid Cd", "-", "0.8"); r+=1
        self.inp_a_valve = self.add_input_row(l4, r, "Valve Surface Area", "m²", "0.0010178", readonly=True); r+=1
        view_layout.addWidget(card4)
        
        view_layout.addStretch()
        scroll.setWidget(view_container)
        layout.addWidget(scroll)

    def reset_defaults(self):
        self.inp_L1.setText("0.44")
        self.inp_D1.setText("0.1646")
        self.inp_L2.setText("12.0")
        self.inp_D2.setText("0.10")
        self.inp_L3.setText("8.0")
        self.inp_D3.setText("0.036")
        
        self.inp_num_perfs.setText("7")
        self.inp_outer_rad.setText("0.004")
        self.inp_inner_rad.setText("0.0003")
        self.inp_grain_len.setText("0.012")
        self.inp_web_thick.setText("0.0015")
        self.inp_density.setText("1600")
        self.inp_flame_temp.setText("2800")
        self.inp_energy.setText("9.0e5")
        self.inp_poly_ratio.setText("1.25")
        self.inp_covolume.setText("0.001")
        self.inp_shape_coeffs.setText("0.1, 0.5, 0.2, 0.1, 0.05, 0.05")
        self.inp_viva.setText("2000")
        self.inp_alpha.setText("0.8")
        self.inp_f.setText("9e5")
        self.inp_m1.setText("1.2")
        
        self.cmb_gas.setCurrentIndex(0)
        
        self.inp_mproj.setText("0.03")
        self.inp_mp.setText("4.0")
        self.inp_P0.setText("20")
        
        self.inp_eta.setText("0.6")
        self.inp_coeff_p.setText("0.02")
        self.inp_coeff_proj.setText("0.01")
        self.inp_p_residual.setText("1.0")
        self.inp_p_burst.setText("500")
        self.inp_valve_delay.setText("50")
        self.inp_cd.setText("0.8")
        self.compute_bounds()
        self.update_gas_properties()

    def build_tab_simulation(self):
        layout = QVBoxLayout(self.tab_sim)
        layout.setContentsMargins(40, 20, 40, 20)
        
        top_layout = QHBoxLayout()
        self.btn_run = QPushButton("RUN SIMULATION")
        self.btn_run.clicked.connect(self.run_simulation)
        self.btn_reset = QPushButton("RESET ANIMATION")
        self.btn_reset.setObjectName("btnReset")
        self.btn_reset.clicked.connect(lambda: self.lbl_sim_status.setText("Cleared. Ready for execution..."))
        
        self.lbl_sim_status = QLabel("Awaiting deployment commands...")
        self.lbl_sim_status.setStyleSheet("color: #8b949e; font-size: 16px; font-style: italic; padding-left: 20px;")
        
        top_layout.addWidget(self.btn_run)
        top_layout.addWidget(self.btn_reset)
        top_layout.addWidget(self.lbl_sim_status)
        top_layout.addStretch()
        layout.addLayout(top_layout)
        
        anim_frame = QFrame()
        anim_frame.setStyleSheet("background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; margin-top: 20px;")
        anim_layout = QVBoxLayout(anim_frame)
        self.canvas_anim = AnimationCanvas(self, width=12, height=6)
        anim_layout.addWidget(self.canvas_anim)
        layout.addWidget(anim_frame, stretch=1)
        
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(0, 20, 0, 0)
        self.live_lbls = []
        labels_str = ["Time: --", "P_ch: --", "P_pump: --", "P_launch: --", "Vel Piston: --", "Vel Proj: --"]
        for txt in labels_str:
            lbl = QLabel(txt)
            lbl.setStyleSheet("color: #FFFFFF; font-family: Consolas, monospace; font-size: 16px; font-weight: 600; padding: 12px; background: #161b22; border: 1px solid #30363d; border-radius: 4px;")
            lbl.setAlignment(Qt.AlignCenter)
            self.live_lbls.append(lbl)
            footer_layout.addWidget(lbl)
            
        layout.addLayout(footer_layout)

    def create_stat_block(self, dict_key, label_text):
        w = QWidget(); l = QVBoxLayout(w); l.setAlignment(Qt.AlignCenter); l.setSpacing(5)
        val = QLabel("--")
        val.setStyleSheet("font-size: 28px; font-weight: bold; color: #FFFFFF; font-family: Consolas, monospace; border: none; background: transparent;")
        val.setAlignment(Qt.AlignCenter)
        
        lbl = QLabel(label_text)
        lbl.setStyleSheet("font-size: 13px; color: #8b949e; font-weight: normal; border: none; background: transparent;")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setWordWrap(True)
        
        self.res_labels[dict_key] = val
        l.addWidget(val); l.addWidget(lbl)
        return w

    def create_report_card(self, title, items, spark_key):
        card = QFrame()
        card.setObjectName("ReportCard")
        l = QVBoxLayout(card); l.setSpacing(15); l.setContentsMargins(20, 20, 20, 20)
        
        t_lbl = QLabel(title)
        t_lbl.setObjectName("CardTitle")
        l.addWidget(t_lbl)
        
        metrics_h = QHBoxLayout()
        for k, txt in items:
            metrics_h.addWidget(self.create_stat_block(k, txt))
        l.addLayout(metrics_h)
        
        spark = MatplotlibCanvas(self, width=8, height=1, sparkline=True)
        self.res_sparks[spark_key] = spark
        l.addWidget(spark)
        
        return card

    def build_tab_results(self):
        layout = QVBoxLayout(self.tab_res)
        layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        c = QWidget(); c_l = QVBoxLayout(c); c_l.setContentsMargins(40, 40, 40, 40); c_l.setSpacing(25)
        
        banner = QHBoxLayout()
        lbl_out = QLabel("CEASE FINAL ENGINEERING REPORT")
        lbl_out.setStyleSheet("font-size: 24px; color: #58a6ff; font-weight: bold;")
        banner.addWidget(lbl_out)
        banner.addStretch()
        
        btn_export = QPushButton("EXPORT TO TXT")
        btn_export.setStyleSheet("background-color: #1f6feb; border: none; font-size: 14px; font-weight: bold; color: white; padding: 10px 20px;")
        btn_export.clicked.connect(self.export_report)
        banner.addWidget(btn_export)
        c_l.addLayout(banner)
        
        b_box = QHBoxLayout()
        self.b_status = QLabel("✔ Stable Simulation")
        self.b_pressure = QLabel("✔ Pressure Check Nominal")
        self.b_valve = QLabel("✔ Valve Operated Correctly")
        for b in [self.b_status, self.b_pressure, self.b_valve]:
            b.setStyleSheet("background-color: #2ea043; color: white; padding: 8px 15px; border-radius: 4px; font-weight: bold;")
            b_box.addWidget(b)
        b_box.addStretch()
        c_l.addLayout(b_box)
        
        p_card = self.create_report_card("PRESSURE DYNAMICS", [("p_ch", "Peak Chamber\nPressure (bar)"), ("p_pump", "Peak Pump\nPressure (bar)"), ("p_launch", "Peak Launch\nPressure (bar)")], "spark_p")
        c_l.addWidget(p_card)
        piston_card = self.create_report_card("PISTON KINEMATICS", [("p_peak_v", "Peak Velocity\n(m/s)"), ("p_final_l", "Final Length\n(mm)"), ("p_extrusion", "Extrusion Depth\n(mm)")], "spark_piston")
        c_l.addWidget(piston_card)
        proj_card = self.create_report_card("PROJECTILE DYNAMICS", [("proj_muz", "Muzzle Velocity\n(m/s)"), ("proj_accel", "Peak Acceleration\n(m/s²)"), ("proj_impact", "Impact Velocity\n(m/s)")], "spark_proj")
        c_l.addWidget(proj_card)
        time_card = self.create_report_card("EVENT TIMING", [("t_burst", "Valve Burst\nTime (ms)"), ("t_start", "Projectile Start\nTime (ms)"), ("t_firing", "Total Firing\nTime (ms)")], "spark_time")
        c_l.addWidget(time_card)
        
        c_l.addStretch()
        scroll.setWidget(c)
        layout.addWidget(scroll)

    def attach_toolbar(self, widget, canvas):
        toolbar = NavigationToolbar2QT(canvas, self)
        toolbar.setStyleSheet("background-color: #21262d; border: none; padding: 5px; border-radius: 4px;")
        l = widget.layout()
        l.insertWidget(0, toolbar)

    def build_tab_analysis(self):
        layout = QVBoxLayout(self.tab_ana)
        layout.setContentsMargins(20, 20, 20, 20)
        
        self.ana_tabs = QTabWidget()
        self.ana_tabs.setStyleSheet("QTabWidget::pane { border: none; } QTabBar::tab { padding: 10px 20px; font-size: 15px; }")
        
        self.a_tab_p = QWidget(); l_p = QVBoxLayout(self.a_tab_p)
        self.canvas_a_p = InteractiveCanvas(self, width=10, height=8); l_p.addWidget(self.canvas_a_p)
        self.attach_toolbar(self.a_tab_p, self.canvas_a_p)
        self.ana_tabs.addTab(self.a_tab_p, "Pressure vs Time")
        
        self.a_tab_v = QWidget(); l_v = QVBoxLayout(self.a_tab_v)
        self.canvas_a_v = InteractiveCanvas(self, width=10, height=8); l_v.addWidget(self.canvas_a_v)
        self.attach_toolbar(self.a_tab_v, self.canvas_a_v)
        self.ana_tabs.addTab(self.a_tab_v, "Velocity vs Time")
        
        self.a_tab_x = QWidget(); l_x = QVBoxLayout(self.a_tab_x)
        self.canvas_a_x = InteractiveCanvas(self, width=10, height=8); l_x.addWidget(self.canvas_a_x)
        self.attach_toolbar(self.a_tab_x, self.canvas_a_x)
        self.ana_tabs.addTab(self.a_tab_x, "Position vs Time")
        
        self.a_tab_e = QWidget(); l_e = QVBoxLayout(self.a_tab_e)
        self.canvas_a_e = InteractiveCanvas(self, width=10, height=8); l_e.addWidget(self.canvas_a_e)
        self.attach_toolbar(self.a_tab_e, self.canvas_a_e)
        self.ana_tabs.addTab(self.a_tab_e, "Energy Breakdown")
        
        layout.addWidget(self.ana_tabs)

    def build_tab_comparison(self):
        layout = QVBoxLayout(self.tab_comp)
        layout.addWidget(QLabel("Gas Comparison Analysis Pipeline - Offline", styleSheet="color:#B0BEC5; font-size: 24px; font-weight: bold;"))

    def run_simulation(self):
        try:
            # Constraints Validation Checks
            P_burst = float(self.inp_p_burst.text())
            P_init = float(self.inp_P0.text())
            m_pow = float(self.inp_m1.text())
            m_proj = float(self.inp_mproj.text())

            if not (200 <= P_burst <= 1500):
                raise ValueError(f"CRITICAL BOUNDARY ERROR: Valve Burst Pressure must be strictly between 200 and 1500 bar. (Got {P_burst})")
            if not (5 <= P_init <= 100):
                raise ValueError(f"CRITICAL BOUNDARY ERROR: Initial Gas Pressure must be strictly between 5 and 100 bar. (Got {P_init})")
            if not (0.5 <= m_pow <= 5.0):
                raise ValueError(f"CRITICAL BOUNDARY ERROR: Powder Mass must be strictly between 0.5 and 5 kg. (Got {m_pow})")
            if not (0.01 <= m_proj <= 0.2):
                raise ValueError(f"CRITICAL BOUNDARY ERROR: Projectile Mass must be strictly between 0.01 and 0.2 kg. (Got {m_proj})")

            L1 = float(self.inp_L1.text()); D1 = float(self.inp_D1.text())
            L2 = float(self.inp_L2.text()); D2 = float(self.inp_D2.text())
            L3 = float(self.inp_L3.text()); D3 = float(self.inp_D3.text())
            
            num_perfs = int(self.inp_num_perfs.text()); outer_rad = float(self.inp_outer_rad.text())
            inner_rad = float(self.inp_inner_rad.text()); grain_len = float(self.inp_grain_len.text())
            web_thick = float(self.inp_web_thick.text()); density = float(self.inp_density.text())
            flame_temp = float(self.inp_flame_temp.text()); energy = float(self.inp_energy.text())
            poly_ratio = float(self.inp_poly_ratio.text()); covolume = float(self.inp_covolume.text())
            viva = float(self.inp_viva.text()); alpha = float(self.inp_alpha.text())
            f = float(self.inp_f.text()); m1 = float(self.inp_m1.text())
            
            coeffs_str = self.inp_shape_coeffs.text().split(',')
            coeffs = [float(c.strip()) for c in coeffs_str] if len(coeffs_str) == 3 else [0.1, 0.5, 0.2, 0.0, 0.0, 0.0]
            if len(coeffs) < 6: coeffs.extend([0]*(6-len(coeffs)))
            shape_dict = {'A': coeffs[0], 'B': coeffs[1], 'C': coeffs[2], 'D': coeffs[3], 'E': coeffs[4], 'F': coeffs[5]}
            
            powder = PowderModel(num_perfs, outer_rad, inner_rad, grain_len, web_thick, density, flame_temp, energy, poly_ratio, covolume, shape_dict, viva, alpha, f, m1)
            
            gas_choice = self.cmb_gas.currentText()
            gamma = float(self.inp_gamma.text())
            cp = float(self.inp_cp.text())
            cv = float(self.inp_cv.text())
            molar_mass = 0.004 if "Helium" in gas_choice else (0.002 if "Hydrogen" in gas_choice else 0.028)
            gas = GasModel(gas_choice.split(" ")[0], gamma, cp, cv, molar_mass)
            
            shot = ShotParameters(
                "Dash Launcher", gas.gas_type, float(self.inp_P0.text()), 300.0, float(self.inp_mp.text()),
                L2, 1.0, float(self.inp_coeff_p.text()), 0.0, float(self.inp_p_burst.text()),
                float(self.inp_valve_delay.text()), 0.0, "Air", 300.0, float(self.inp_p_residual.text()),
                float(self.inp_mproj.text()), float(self.inp_coeff_proj.text()), "Custom",
                float(self.inp_cd.text()), float(self.inp_a_valve.text())
            )
        except Exception as e:
            QMessageBox.critical(self, "Invalid Physical Range", str(e))
            return
            
        self.btn_run.setText("COMPUTING..."); self.btn_run.setEnabled(False)
        self.lbl_sim_status.setText("Iterating differential mass flow states...")
        self.lbl_sim_status.setStyleSheet("color: #d29922; font-size: 16px; font-style: italic; padding-left: 20px;")
        QApplication.processEvents()
        
        try:
            solver = MultiZoneSolver(L1, D1, L2, D2, L3, D3, powder, gas, shot)
            results = solver.run()
            
            t_arr = results['t'] * 1000
            p1_b = results['p1'] / 1e5; p2_b = results['p2'] / 1e5; p3_b = results['p3'] / 1e5
            vp = results['v_p']; vproj = results['v_proj']; aproj = results['a_proj']
            xp = results['x_p']; xproj = results['x_proj']
            
            self.anim_data = {'t': t_arr, 'p1': p1_b, 'p2': p2_b, 'p3': p3_b, 'xp': xp, 'xproj': xproj, 'vp': vp, 'vproj': vproj}
            self.anim_L1 = L1; self.anim_L2 = L2; self.anim_L3 = L3
            
            piston_peak_vel = np.max(vp) if len(vp) else 0.0
            extrusion = np.max(xp) if len(xp) else 0.0
            piston_final_len = 1.0 + extrusion
            
            proj_peak_vel = np.max(vproj) if len(vproj) else 0.0
            muzzle_vel = vproj[-1] if len(vproj) else 0.0
            peak_accel = np.max(aproj) if len(aproj) else 0.0
            
            p_ch_peak = np.max(p1_b) if len(p1_b) else 0.0
            p_pump_peak = np.max(p2_b) if len(p2_b) else 0.0
            p_laun_peak = np.max(p3_b) if len(p3_b) else 0.0
            p_res_muzzle = p3_b[-1] if len(p3_b) else 0.0
            
            burst_time = results['burst_time'] * 1000 if results['burst_time'] >= 0 else 0.0
            firing_time = t_arr[-1] if len(t_arr) else 0.0
            idx_start = np.argmax(xproj > 0.0) if np.any(xproj > 0.0) else -1
            proj_start = t_arr[idx_start] if idx_start >= 0 else 0.0
            
            self.res_labels['p_ch'].setText(f"{p_ch_peak:.1f}")
            self.res_labels['p_pump'].setText(f"{p_pump_peak:.1f}")
            self.res_labels['p_launch'].setText(f"{p_laun_peak:.1f}")
            self.res_labels['p_peak_v'].setText(f"{piston_peak_vel:.1f}")
            self.res_labels['p_final_l'].setText(f"{piston_final_len*1000:.1f}")
            self.res_labels['p_extrusion'].setText(f"{extrusion*1000:.1f}")
            self.res_labels['proj_muz'].setText(f"{muzzle_vel:.1f}")
            self.res_labels['proj_accel'].setText(f"{peak_accel:.2e}")
            self.res_labels['proj_impact'].setText(f"{muzzle_vel:.1f}")
            self.res_labels['t_burst'].setText(f"{burst_time:.2f}")
            self.res_labels['t_start'].setText(f"{proj_start:.2f}")
            self.res_labels['t_firing'].setText(f"{firing_time:.2f}")
            
            ax_sp = self.res_sparks['spark_p'].axes[0]; ax_sp.clear(); ax_sp.axis('off')
            ax_sp.plot(t_arr, p1_b, color='#f85149', lw=2); ax_sp.plot(t_arr, p2_b, color='#d29922', lw=2); ax_sp.plot(t_arr, p3_b, color='#1f6feb', lw=2); self.res_sparks['spark_p'].draw()
            ax_spis = self.res_sparks['spark_piston'].axes[0]; ax_spis.clear(); ax_spis.axis('off')
            ax_spis.plot(t_arr, vp, color='#1f6feb', lw=2); self.res_sparks['spark_piston'].draw()
            ax_spr = self.res_sparks['spark_proj'].axes[0]; ax_spr.clear(); ax_spr.axis('off')
            ax_spr.plot(t_arr, vproj, color='#2ea043', lw=2); self.res_sparks['spark_proj'].draw()
            ax_tim = self.res_sparks['spark_time'].axes[0]; ax_tim.clear(); ax_tim.axis('off')
            ax_tim.plot(t_arr, xproj, color='#c9d1d9', lw=2)
            ax_tim.axvline(x=burst_time, color='#d29922', linestyle='--')
            self.res_sparks['spark_time'].draw()
            
            # --- Status Validation Engine ---
            sim_passed = True
            
            if muzzle_vel < 10.0:
                self.b_status.setText("⚠ Critical Firing Failure (V_proj ~ 0)")
                self.b_status.setStyleSheet("background-color: #f85149; color: white; padding: 8px 15px; border-radius: 4px; font-weight: bold;")
                sim_passed = False
            else:
                self.b_status.setText("✔ Stable Simulation")
                self.b_status.setStyleSheet("background-color: #2ea043; color: white; padding: 8px 15px; border-radius: 4px; font-weight: bold;")
                
            if p_ch_peak > 5000 or p_pump_peak > 5000:
                self.b_pressure.setText("⚠ Critical Overpressure Warning (>5000 bar)")
                self.b_pressure.setStyleSheet("background-color: #f85149; color: white; padding: 8px 15px; border-radius: 4px; font-weight: bold;")
                sim_passed = False
            elif p_laun_peak <= 0.1:
                self.b_pressure.setText("⚠ Zero Launch Pressure Detected")
                self.b_pressure.setStyleSheet("background-color: #f85149; color: white; padding: 8px 15px; border-radius: 4px; font-weight: bold;")
                sim_passed = False
            else:
                self.b_pressure.setText("✔ Pressure Checks Nominal")
                self.b_pressure.setStyleSheet("background-color: #2ea043; color: white; padding: 8px 15px; border-radius: 4px; font-weight: bold;")
                
            if burst_time <= 0:
                self.b_valve.setText("⚠ Valve Never Opened")
                self.b_valve.setStyleSheet("background-color: #f85149; color: white; padding: 8px 15px; border-radius: 4px; font-weight: bold;")
                sim_passed = False
            elif len(vp) > 0 and vp[-1] < -10.0:
                self.b_valve.setText("⚠ Unstable Piston Bounce Detected")
                self.b_valve.setStyleSheet("background-color: #d29922; color: white; padding: 8px 15px; border-radius: 4px; font-weight: bold;")
            else:
                self.b_valve.setText("✔ Valve Operated Correctly")
                self.b_valve.setStyleSheet("background-color: #2ea043; color: white; padding: 8px 15px; border-radius: 4px; font-weight: bold;")

            if not sim_passed:
                QMessageBox.warning(self, "Sanity Check Failed", "Simulation executed but resulted in non-physical kinematic states (Zero Velocity, Failed Valve, or Extreme Overpressure).")

            gas_val = self.cmb_gas.currentText()
            self.raw_report_text = f"""----------------------------------------
SIMULATION FINAL REPORT
----------------------------------------

Launcher Configuration: [Auto]
Working Gas: [{gas_val}]

--- PRESSURE SUMMARY ---
Peak Chamber Pressure: {p_ch_peak:.2f} bar
Peak Pump Pressure: {p_pump_peak:.2f} bar
Peak Launch Pressure: {p_laun_peak:.2f} bar
Projectile Base Peak Pressure: {p_laun_peak:.2f} bar
Residual Muzzle Pressure: {p_res_muzzle:.2f} bar

--- PISTON SUMMARY ---
Piston Peak Velocity: {piston_peak_vel:.2f} m/s
Piston Final Length: {piston_final_len*1000:.2f} mm
Extrusion Depth: {extrusion*1000:.2f} mm

--- PROJECTILE SUMMARY ---
Projectile Peak Velocity: {proj_peak_vel:.2f} m/s
Muzzle Velocity: {muzzle_vel:.2f} m/s
Peak Acceleration: {peak_accel:.2e} m/s²

--- TIMING SUMMARY ---
Valve Burst Time: {burst_time:.2f} ms
Projectile Start Time: {proj_start:.2f} ms
Firing Time: {firing_time:.2f} ms

----------------------------------------"""
            
            # --- POPULATE ANALYSIS GRAPHS ---
            ax_p = self.canvas_a_p.axes[0]; ax_p.clear()
            ax_p.plot(t_arr, p1_b, color='#f85149', label='Chamber', lw=2)
            ax_p.plot(t_arr, p2_b, color='#d29922', label='Pump Tube', lw=2)
            ax_p.plot(t_arr, p3_b, color='#1f6feb', label='Launch Tube', lw=2)
            
            p1_max_idx = np.argmax(p1_b)
            p2_max_idx = np.argmax(p2_b)
            ax_p.plot(t_arr[p1_max_idx], p1_b[p1_max_idx], 'x', color='white', markersize=10)
            ax_p.annotate(f"Peak (Ch): {p1_b[p1_max_idx]:.0f} bar", xy=(t_arr[p1_max_idx], p1_b[p1_max_idx]), xytext=(10, 10), textcoords="offset points", color='white', fontweight='bold', bbox=dict(boxstyle="round4", fc="#21262d", ec="#f85149"))
            
            ax_p.plot(t_arr[p2_max_idx], p2_b[p2_max_idx], 'x', color='white', markersize=10)
            ax_p.annotate(f"Peak (Pump): {p2_b[p2_max_idx]:.0f} bar", xy=(t_arr[p2_max_idx], p2_b[p2_max_idx]), xytext=(10, 10), textcoords="offset points", color='white', fontweight='bold', bbox=dict(boxstyle="round4", fc="#21262d", ec="#d29922"))
            
            ax_p.set_xlabel("Time (ms)"); ax_p.set_ylabel("Pressure (bar)")
            ax_p.legend(loc='upper right', frameon=True, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
            self.canvas_a_p.draw()
            
            ax_v = self.canvas_a_v.axes[0]; ax_v.clear()
            ax_v.plot(t_arr, vp, color='#1f6feb', label='V_Piston', lw=2)
            ax_v.plot(t_arr, vproj, color='#2ea043', label='V_Projectile', lw=2)
            vp_max_idx = np.argmax(vp)
            ax_v.plot(t_arr[vp_max_idx], vp[vp_max_idx], 'x', color='white', markersize=10)
            ax_v.annotate(f"Peak: {vp[vp_max_idx]:.0f} m/s", xy=(t_arr[vp_max_idx], vp[vp_max_idx]), xytext=(10, 10), textcoords="offset points", color='white', fontweight='bold', bbox=dict(boxstyle="round4", fc="#21262d", ec="#1f6feb"))
            
            ax_v.set_xlabel("Time (ms)"); ax_v.set_ylabel("Velocity (m/s)")
            ax_v.legend(loc='upper left', frameon=True, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
            self.canvas_a_v.draw()
            
            ax_x = self.canvas_a_x.axes[0]; ax_x.clear()
            ax_x.plot(t_arr, xp, color='#1f6feb', label='X_Piston', lw=2)
            ax_x.plot(t_arr, xproj, color='#2ea043', label='X_Projectile', lw=2)
            ax_x.axvline(x=burst_time, color='#d29922', linestyle='--', label='Valve Burst')
            ax_x.set_xlabel("Time (ms)"); ax_x.set_ylabel("Position (m)")
            ax_x.legend(loc='upper left', frameon=True, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
            self.canvas_a_x.draw()
            
            ax_e = self.canvas_a_e.axes[0]; ax_e.clear()
            if 'e_powder' in results:
                ax_e.plot(t_arr, results['e_powder']/1e3, color='#c9d1d9', label='E_Powder (Total)', lw=2, linestyle='--')
                ax_e.plot(t_arr, results['e_gas']/1e3, color='#f85149', label='E_Gas (Thermal)', lw=2)
                ax_e.plot(t_arr, results['e_ke_piston']/1e3, color='#1f6feb', label='KE_Piston', lw=2)
                ax_e.plot(t_arr, results['e_ke_proj']/1e3, color='#2ea043', label='KE_Projectile', lw=2)
                ax_e.plot(t_arr, results['e_loss']/1e3, color='#8b949e', label='E_Loss (Leak/Heat)', lw=2)
                ax_e.set_xlabel("Time (ms)"); ax_e.set_ylabel("Energy (kJ)")
                ax_e.legend(loc='upper left', frameon=True, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
                self.canvas_a_e.draw()

            self.lbl_sim_status.setText("Simulation Complete.")
            self.lbl_sim_status.setStyleSheet("color: #2ea043; font-size: 16px; font-style: italic; padding-left: 20px;")
            self.play_animation()
            
        except Exception as e:
            self.lbl_sim_status.setText("Simulation Failed")
            self.lbl_sim_status.setStyleSheet("color: #f85149; font-size: 16px; font-style: italic; padding-left: 20px;")
        finally:
            self.btn_run.setText("RUN SIMULATION")
            self.btn_run.setEnabled(True)

    def export_report(self):
        if not self.raw_report_text:
            QMessageBox.warning(self, "Export Failed", "Please run a simulation before exporting.")
            return
            
        try:
            target = os.path.join(os.getcwd(), "cease_final_report.txt")
            with open(target, "w") as f:
                f.write(self.raw_report_text)
            QMessageBox.information(self, "Export Successful", f"Report saved cleanly to:\n{target}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    def play_animation(self):
        self.canvas_anim.setup_scene(self.anim_L1, self.anim_L2, self.anim_L3, float(self.inp_D1.text()), float(self.inp_D2.text()), float(self.inp_D3.text()))
        self.anim_index = 0
        self.anim_timer.start(25) 

    def anim_step(self):
        if self.anim_index < len(self.anim_data['t']):
            xp = self.anim_data['xp'][self.anim_index]
            xproj = self.anim_data['xproj'][self.anim_index]
            t = self.anim_data['t'][self.anim_index]
            p1 = self.anim_data['p1'][self.anim_index]
            p2 = self.anim_data['p2'][self.anim_index]
            p3 = self.anim_data['p3'][self.anim_index]
            vp = self.anim_data['vp'][self.anim_index]
            vproj = self.anim_data['vproj'][self.anim_index]
            
            self.canvas_anim.update_positions(xp, xproj, self.anim_L1, self.anim_L2, self.anim_L3, p1, p2, p3)
            
            self.live_lbls[0].setText(f"Time: {t:.2f} ms")
            self.live_lbls[1].setText(f"P_ch: {p1:.0f} bar")
            self.live_lbls[2].setText(f"P_pump: {p2:.0f} bar")
            self.live_lbls[3].setText(f"P_launch: {p3:.0f} bar")
            self.live_lbls[4].setText(f"Vel Piston: {vp:.1f} m/s")
            self.live_lbls[5].setText(f"Vel Proj: {vproj:.1f} m/s")
            
            base_step = max(1, len(self.anim_data['t']) // 100)
            self.anim_index += base_step
        else:
            self.anim_timer.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
