class ShotParameters:
    def __init__(self, launcher_name: str, 
                 light_gas_type: str, initial_gas_pressure_bar: float, initial_temp_k: float,
                 piston_mass_kg: float, initial_piston_length_m: float, initial_piston_base_pressure_bar: float,
                 piston_friction_coeff: float, piston_friction_equiv_pressure_bar: float,
                 valve_burst_pressure_bar: float, valve_opening_delay_us: float, distance_valve_to_proj_m: float,
                 residual_gas_type: str, residual_gas_temp_k: float, residual_gas_pressure_bar: float,
                 projectile_mass_kg: float, projectile_friction_coeff: float, 
                 powder_type_reference: str, 
                 cd_valve: float, a_valve_m2: float):
        """
        Global unified parameters for a specific shot. Handles Unit Conversion to SI immediately.
        """
        self.launcher_name = launcher_name
        
        self.light_gas_type = light_gas_type
        # bar to Pascal
        self.initial_gas_pressure = initial_gas_pressure_bar * 1e5
        self.initial_temp = initial_temp_k
        
        self.piston_mass = piston_mass_kg
        self.initial_piston_length = initial_piston_length_m
        self.initial_piston_base_pressure = initial_piston_base_pressure_bar * 1e5
        
        self.piston_friction_coeff = piston_friction_coeff
        self.piston_friction_equiv_pressure = piston_friction_equiv_pressure_bar * 1e5
        
        self.valve_burst_pressure = valve_burst_pressure_bar * 1e5
        # microseconds to seconds
        self.valve_opening_delay = valve_opening_delay_us * 1e-6
        self.distance_valve_to_proj = distance_valve_to_proj_m
        self.cd_valve = cd_valve
        self.a_valve = a_valve_m2
        
        self.residual_gas_type = residual_gas_type
        self.residual_gas_temp = residual_gas_temp_k
        self.residual_gas_pressure = residual_gas_pressure_bar * 1e5
        
        self.projectile_mass = projectile_mass_kg
        self.projectile_friction_coeff = projectile_friction_coeff
        
        self.powder_type_reference = powder_type_reference
