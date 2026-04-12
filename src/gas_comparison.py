import numpy as np
from src.combustion import PowderCombustion
from src.coupled_solver import CoupledSolver

def run_simulation_for_gas(params, gas_name, gamma):
    comb = PowderCombustion(
        length=params['L1'], diameter=params['D1'], mass_powder=params['m1'], 
        f=params['f'], alpha=params['alpha'], beta=params['beta'], viva=params['viva']
    )
    solver = CoupledSolver(
        m_p=params['mp'], D_p=params['D2'], L_p=params['L2'], 
        m_proj=params['mproj'], D_proj=params['D3'], L_proj=params['L3'], 
        P0_gas=params['P0'], gamma=gamma
    )
    
    max_time = 1.0
    dt = comb.dt
    t_current = 0.0
    exited = False
    
    t_arr = []
    p2_arr = []
    plot_step = 0
    max_p2 = 0.0
    
    while t_current < max_time and not exited:
        comb.step()
        exited = solver.step(comb.P)
        
        p2_current = solver.P_gas
        if p2_current > max_p2:
            max_p2 = p2_current
            
        # Optimization: Don't store everything, just enough to plot
        if plot_step % 200 == 0 or exited:
            t_arr.append(t_current)
            p2_arr.append(p2_current)
            
        t_current += dt
        plot_step += 1
        
    final_v = solver.v_proj if exited else 0.0
    burst_t = solver.burst_time if solver.burst_time >= 0.0 else -1.0
    firing_t = solver.exit_time if exited else -1.0
    
    return {
        "velocity": final_v,
        "peak_pressure": max_p2 / 1e5,  # bar
        "burst_time": burst_t * 1000, # ms
        "firing_time": firing_t * 1000, # ms
        "t_arr": np.array(t_arr) * 1000, # ms
        "p2_arr": np.array(p2_arr) / 1e5 # bar
    }

def compare_all_gases(params, progress_callback=None):
    gases = {
        "Hydrogen": 1.4,
        "Helium": 1.66,
        "Nitrogen": 1.4
    }
    
    results = {}
    for gas_name, gamma in gases.items():
        if progress_callback:
            progress_callback(gas_name)
        results[gas_name] = run_simulation_for_gas(params, gas_name, gamma)
        
    return results
