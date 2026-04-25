import numpy as np

class BurstDisk:

    def __init__(self, burst_pressure, A_max):

        self.burst_pressure = burst_pressure
        self.A_max = A_max
        self.burst_time = -1
        self.open = False

    def area(self, P, t):

        if P >= self.burst_pressure and self.burst_time < 0:
            self.burst_time = t

        if self.burst_time < 0:
            return 0

        dt = t - self.burst_time

        if dt < 0:
            return 0

        A = self.A_max * (1 - np.exp(-dt / 5e-5))

        if A > self.A_max:
            A = self.A_max

        self.open = True

        return A