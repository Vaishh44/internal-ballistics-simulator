import numpy as np

from src.powder_model import PowderModel
from src.gas_model import GasModel
from src.shot_parameters import ShotParameters
from src.multi_zone_solver import MultiZoneSolver

def run_test():
    # STEP 2 — CREATE TEST INPUTS
    
    # 1. Powder Model (Sample geometry)
    powder = PowderModel(
        num_perforations=7,
        outer_radius=0.005,
        inner_radius=0.0005,
        grain_length=0.012,
        web_thickness=0.002,
        density=1600.0,
        flame_temperature=3000.0,
        energy=1000000.0,
        polytropic_ratio=1.25,
        co_volume=0.001,
        shape_coeffs={'A': 0.1, 'B': 0.5, 'C': 0.2, 'D': 0.1, 'E': 0.05, 'F': 0.05},
        viva=2500.0,
        alpha=0.8,
        specific_force=1e6,
        powder_mass_kg=1.6
    )

    # 2. Gas Model (Helium)
    gas = GasModel(
        gas_type="Helium",
        gamma=1.66,
        cp=5192.6,
        cv=3115.6,
        molar_mass=0.0040026
    )

    # 3. Shot Parameters
    shot = ShotParameters(
        launcher_name="Test Launcher",
        light_gas_type="Helium",
        initial_gas_pressure_bar=5.0,     # bar
        initial_temp_k=300.0,             # K
        piston_mass_kg=4.5,
        initial_piston_length_m=0.105,
        initial_piston_base_pressure_bar=1.0,
        piston_friction_coeff=0.0,
        piston_friction_equiv_pressure_bar=0.1, # 0.1 bar effective friction
        valve_burst_pressure_bar=1000.0,        # bar
        valve_opening_delay_us=50.0,            # microseconds
        distance_valve_to_proj_m=0.05,
        residual_gas_type="Air",
        residual_gas_temp_k=300.0,
        residual_gas_pressure_bar=0.1,          # Vacuum-ish residual
        projectile_mass_kg=0.0279,
        projectile_friction_coeff=0.0,
        powder_type_reference="TEST-POWDER",
        cd_valve=0.6,
        a_valve_m2=0.002
    )

    # Geometry: L1, D1, L2, D2, L3, D3
    L1 = 0.440
    D1 = 0.1646
    L2 = 12.245
    D2 = 0.105
    L3 = 8.1
    D3 = 0.036

    # STEP 3 — RUN SOLVER
    solver = MultiZoneSolver(
        L1=L1, D1=D1, L2=L2, D2=D2, L3=L3, D3=D3,
        powder=powder, gas=gas, shot=shot
    )

    print("Running solver...")
    result = solver.run()

    # STEP 4 — PRINT RESULTS
    t_ms = result['t'] * 1000.0
    v_proj = result['v_proj']
    p1_bar = result['p1'] / 1e5
    p2_bar = result['p2'] / 1e5
    p3_bar = result['p3'] / 1e5
    a_proj = result['a_proj']

    muzzle_velocity = v_proj[-1]
    peak_chamber = np.max(p1_bar)
    peak_pump = np.max(p2_bar)
    peak_launch = np.max(p3_bar)
    peak_accel = np.max(a_proj)
    firing_time = t_ms[-1]

    print("\n" + "="*40)
    print(f"Muzzle Velocity:          {muzzle_velocity:.2f} m/s")
    print(f"Peak Chamber Pressure:    {peak_chamber:.2f} bar")
    print(f"Peak Pump Pressure:       {peak_pump:.2f} bar")
    print(f"Peak Launch Pressure:     {peak_launch:.2f} bar")
    print(f"Peak Projectile Accel:    {peak_accel:.2e} m/s^2")
    print(f"Firing Time:              {firing_time:.2f} ms")
    if result['burst_time'] >= 0:
        print(f"Valve Burst Time:         {result['burst_time']*1000:.2f} ms")
    else:
        print("Valve Burst Time:         DID NOT BURST")
    print("="*40 + "\n")

    # STEP 5 & 6 — BASIC VALIDATION & DEBUG OUTPUT
    print("Validation & Debug Checks:")
    if muzzle_velocity > 0:
        print("[PASS] Projectile moved.")
    else:
        print("[FAIL] Projectile failed to move.")

    if result['burst_time'] >= 0:
        print("[PASS] Valve burst occurred.")
    else:
        print("[FAIL] Valve did not burst.")

    if not np.any(np.isnan(p1_bar)):
        print("[PASS] No NaNs in chamber pressure array.")
    else:
        print("[FAIL] NaNs detected in pressure array.")

    print("\nSample Time-Steps (every ~1 ms):")
    idx = 0
    while idx < len(t_ms):
        print(f"t={t_ms[idx]:.2f}ms | P_ch={p1_bar[idx]:.1f}bar | P_pump={p2_bar[idx]:.1f}bar | v_p={result['v_p'][idx]:.1f}m/s | v_proj={v_proj[idx]:.1f}m/s")
        target_t = t_ms[idx] + 1.0
        next_indices = np.where(t_ms >= target_t)[0]
        if len(next_indices) == 0:
            break
        idx = next_indices[0]

if __name__ == "__main__":
    run_test()
