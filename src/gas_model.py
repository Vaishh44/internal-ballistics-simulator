class GasModel:
    def __init__(self, gas_type: str, gamma: float, cp: float, cv: float, molar_mass: float):
        """
        Thermodynamic Gas model representing working, push, or residual gases.
        """
        self.gas_type = gas_type
        self.gamma = gamma
        self.cp = cp
        self.cv = cv
        self.molar_mass = molar_mass
        
        # Universal Gas Constant (J / (mol K))
        self.R_universal = 8.31446261815324
        # Specific Gas Constant
        self.R_specific = self.R_universal / self.molar_mass

    def get_density(self, pressure: float, temperature: float) -> float:
        """
        Ideal gas law: rho = P / (R_specific * T)
        """
        if temperature <= 0:
            return 0.0
        return pressure / (self.R_specific * temperature)
    
    def get_temperature(self, pressure: float, density: float) -> float:
        """
        Ideal gas law: T = P / (rho * R_specific)
        """
        if density <= 0:
            return 0.0
        return pressure / (density * self.R_specific)
