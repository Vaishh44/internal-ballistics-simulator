# Two-Stage Light Gas Gun Simulator: Theory & User Guide

## A. Project Overview
The Two-Stage Light Gas Gun Internal Ballistics Simulator is an educational and analytical tool designed to accurately predict the dynamics inside a hypervelocity testing mechanism. By solving a series of coupled ordinary differential equations (ODEs), it tracks everything from powder combustion rates to thermodynamic gas compression and supersonic projectile launch. 

It acts as a sandbox for ballistics students, physicists, and engineers to tweak physical configurations computationally before embarking on expensive, dangerous live testing.

---

## B. Physics Explanation

### 1. Combustion Model (Stage 1)
Instead of relying on instantaneous explosion assumptions, the software accurately charts powder regression.
Combustion is evaluated using empirical relationships tracking **fraction of powder burned (z)** based on pressure `P` over time:
`dz/dt = (V_p / r_tube) * (P / P_atm)^alpha`
The resulting liberated gases generate a dynamic pressure expansion pushing precisely against the Piston face based directly upon Noble-Abel Equations of State for dense high-volume combustion gases.

### 2. Gas Compression Model (Stage 2)
The Piston acts as a massive dampening ram. As it accelerates down the pump tube, it ruthlessly compresses a fixed pocket of Light Gas (Hydrogen, Helium, or Nitrogen). 
This violates ideal gas generalizations because the high-speed stroke mimics an **Adiabatic System (Isentropic Compression)**:
`P2 = P_0 * (Vol_0 / Vol_current)^gamma`
Volume shrinks exponentially as the piston drives forward, forcing the pressure up exponentially.

### 3. Projectile Motion (Stage 3)
Once the internal Light Gas forces overcome the mechanical shear limit of the petal valve (P = 470 bar), the physical barrier bursts safely dumping the extreme pressure behind the tiny geometry of the projectile (the sabot). Driven by Force mass conversions (`F = P * Area_projectile`), the projectile is hurled down the launch barrel breaching extreme velocities. 

---

## C. Coupled Dynamics Explained
The true strength of the Two-Stage model relies totally on the physical bridging between these states. Single-stage explosives suffer because the heavy atomic weight of explosive gas limits expansion speeds to around 3 km/s.
By coupling the explosion against a physical Piston, we transfer the *stored energy of combustion* into a *highly efficient Light Gas* (like Hydrogen with minimal mass).
The changing volume (driven by the heavy piston) operates as an artificial energy lens: focusing the massive, slow kinetic explosion of Stage 1 into a tiny, unbelievably fast kinetic acceleration for the Stage 3 Projectile.

---

## D. Parameter Explanations

1. **VIVA**: The empirical burning rate constant of the gunpowder. Higher VIVA = faster explosive progression resulting in aggressive pressure spikes. Lower VIVA drives a slower push, mitigating damage.
2. **beta & alpha**: Combustion sensitivity indices referencing how aggressively the powder burning rate accelerates purely as a function of the pressure surrounding it. (Burning rate is recursive!)
3. **Specific Force (f)**: The total structural energy contained inside 1kg of the gunpowder.
4. **Gamma (γ)**: The Heat Capacity Ratio. It determines how "stiff" the Light Gas acts when compressed. Hydrogen (1.40) behaves predictably. Helium (1.66) heats up much quicker yielding substantially differing velocity characteristics under load.

---

## E. & F. Features Overview & Usage Guide (How to Use)

### 1. INPUT PARAMETERS Tab
Work sequentially down the geometric panels ensuring measurements are entered in standard metric (kg, meters, bar). Select your target Light Gas configuration from the dropdown. 

### 2. SIMULATION Tab
Initiate the `[RUN SIMULATION]` calculation hook. 
- *Insight*: Depending on your bounds, this solves thousands of distinct array indexes traversing `dt` variables across fractions of microseconds tracking exact bounds safely.

### 3. RESULTS Tab
Investigate nominal numerical readouts (Peak Pressure, Burst Time). Check visually against empirical Matplotlib readouts (Velocity & Pressure time-to-time ratios).
- *Insight*: Look for the "Plateau Effect" on the projectile velocity graph. It indicates when your propellant forces zero out rendering a pointless barrel extension constraint!

### 4. GAS COMPARISON Tool
Need to determine if Hydrogen is required over pure Nitrogen?
Hit `[RUN COMPARISON]`. The engine spawns 3 isolated parallel simulations substituting gas models dynamically, rendering simultaneous metrics on a single view window ensuring you can gauge diminishing returns clearly.

### 5. PARAMETER SWEEP Tool
The most effective asset for engineers. Feed a parameter (e.g., VIVA from 1000 to 4000 across 10 steps). 
The engine structurally computes every combination plotting a line curve detailing exactly where theoretical `Muzzle Velocity` peaks against input bounds.
- *Insight*: Maximum VIVA does *not* necessarily translate to maximum kinetic output; find the sweet spot where powder burn exactly matches tube length volume!

### 6. ANIMATION Canvas
Select `▶ PLAY` to structurally witness your mathematical arrays executed geometrically.
Observe the dynamic compression forces rendering real-time against the boundaries until the precise moment Pressure evaluates higher than rupture point!
