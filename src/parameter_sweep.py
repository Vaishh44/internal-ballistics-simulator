import numpy as np
import copy
from src.combustion import PowderCombustion
from src.coupled_solver import CoupledSolver

def run_single_simulation(params, gamma):
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
    
    max_p2 = 0.0
    
    while t_current < max_time and not exited:
        comb.step()
        exited = solver.step(comb.P)
        
        p2_current = solver.P_gas
        if p2_current > max_p2:
            max_p2 = p2_current
            
        t_current += dt
        
    burst_t = solver.burst_time if solver.burst_time >= 0.0 else -1.0
    
    # Physics Validation Check
    if burst_t < 0.0 or not exited:
        print(f"WARNING: Valve did not burst (or projectile did not exit) for VIVA={params['viva']:.2f}")
        final_v = 0.0
    else:
        final_v = solver.v_proj
    
    return {
        "velocity": final_v,
        "peak_pressure": max_p2 / 1e5,  # bar
        "burst_time": burst_t * 1000 # ms
    }

def run_sweep(base_params, gamma, start_viva, end_viva, steps, progress_callback=None):
    viva_values = np.linspace(start_viva, end_viva, int(steps))
    results = []
    
    for i, viva in enumerate(viva_values):
        if progress_callback:
            progress_callback(i + 1, steps, viva)
            
        params_copy = copy.deepcopy(base_params)
        params_copy['viva'] = viva
        
        sim_res = run_single_simulation(params_copy, gamma)
        
        results.append({
            "viva": viva,
            "velocity": sim_res["velocity"],
            "peak_pressure": sim_res["peak_pressure"],
            "burst_time": sim_res["burst_time"]
        })
        
    return results
