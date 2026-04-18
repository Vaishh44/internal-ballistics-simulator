class PowderModel:
    def __init__(self, num_perforations: int, outer_radius: float, inner_radius: float, 
                 grain_length: float, web_thickness: float, density: float, 
                 flame_temperature: float, energy: float, polytropic_ratio: float, 
                 co_volume: float, shape_coeffs: dict, 
                 viva: float, alpha: float, specific_force: float, powder_mass_kg: float):
        """
        CEASE-style Powder Model encapsulating the properties and structural shape
        of the propellant.
        """
        self.num_perforations = num_perforations
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.grain_length = grain_length
        self.web_thickness = web_thickness
        self.density = density
        self.flame_temperature = flame_temperature
        self.energy = energy  # J/kg
        self.polytropic_ratio = polytropic_ratio
        self.co_volume = co_volume
        
        # New Explicit inputs
        self.viva = viva
        self.alpha = alpha
        self.specific_force = specific_force
        self.powder_mass = powder_mass_kg
        
        # Shape coefficients A through F
        self.shape_coeffs = shape_coeffs
        
        # Dynamic state properties for solver
        self.z = 0.0          # burn fraction
        self.dz_dt = 0.0      # rate of burn
    
    def compute_shape_function(self, z: float) -> float:
        """
        psi(z) = A + Bz + Cz^2 + Dz^3 + Ez^4 + Fz^5
        """
        A = self.shape_coeffs.get('A', 0.0)
        B = self.shape_coeffs.get('B', 0.0)
        C = self.shape_coeffs.get('C', 0.0)
        D = self.shape_coeffs.get('D', 0.0)
        E = self.shape_coeffs.get('E', 0.0)
        F = self.shape_coeffs.get('F', 0.0)
        
        return A + B*z + C*(z**2) + D*(z**3) + E*(z**4) + F*(z**5)
