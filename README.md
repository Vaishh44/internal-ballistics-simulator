# Two-Stage Light Gas Gun Simulator

## 1. Offline Execution Guide

This software is designed to operate 100% offline without requiring internet connectivity for licensing, telemetry, or server calculations.

### System Requirements
*   **Operating System**: Windows / macOS / Linux
*   **Python Target**: Python 3.8 to 3.12+

### Required Libraries
The entire physics engine and GUI depend strictly on three core Python libraries:
1.  **PyQt5** (Handles the UI and Application threading loop)
2.  **NumPy** (Handles heavy mathematical array operations and solvers)
3.  **Matplotlib** (Handles the graph plotting and the 2D Animation canvas)

### Installation (Offline Mode via .whl Wheels)
If deploying to a secure, offline PC without internet access, follow these steps:
1.  **On a connected PC**, download the `.whl` files for the target libraries bypassing installation:
    `pip download PyQt5 numpy matplotlib`
2.  Transfer the downloaded package folder via USB to the offline PC.
3.  **On the offline PC**, point `pip` to install directly from the folder:
    `pip install --no-index --find-links=/path/to/usb/folder PyQt5 numpy matplotlib`

### Launching the Application
Once the dependencies are installed, navigate to the directory terminal and invoke the main loop:
`python main.py`

---

## 2. Error Handling & Troubleshooting Guide

| Symptom / Error | Root Cause | Solution |
| :--- | :--- | :--- |
| **`ImportError: No module named PyQt5`** | The UI library is not installed in the python environment. | Ensure `pip install PyQt5` ran successfully. If using environments (conda/venv), verify it is activated before running. |
| **`ImportError: cannot import name FigureCanvas`** | Matplotlib Qt5 backend is missing or mismatched. | Try running `pip install --upgrade matplotlib PyQt5` to synchronize versions. Ensure headless mode isn't forced on your linux kernel. |
| **GUI Freezes during "RUNNING..."** | A massive calculation loop (e.g., Parameter Sweep) was given bounds that exceed millions of steps without `processEvents()`. | We have implemented `QApplication.processEvents()` to prevent this natively, however, if running Sweeps at very small step granularities (e.g. 1000 bounds), simply wait. The solver is purely sequential. |
| **`NaN` / Infinity Math Warnings** | Physical bounds were set beyond the laws of thermodynamics (e.g., negative volume / impossible dimensions). | The Physics Solver (CoupledSolver) traps standard math domain faults, but ensure user-input lengths (L1, L2) are realistic sizes (> 0.0). |
| **"Valve Did Not Burst" Console Warning** | Calculated Stage 1 Combustion + Stage 2 stroke didn't exceed `470 bar` minimum constraint. | Increase `VIVA`, Powder Mass, or decrease Pump Tube limits. The application safely falls back to `Velocity = 0.0`. |
