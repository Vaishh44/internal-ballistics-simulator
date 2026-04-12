# Deployment Guide: Standalone Windows Executable

This guide outlines exactly how to transform the Python-based Two-Stage Light Gas Gun Simulator into a completely standalone `.exe` Windows application. By following these steps, you will generate a package that works securely on any target Windows PC without needing Python installed locally.

---

## 1. Building the Executable using PyInstaller

Start by installing PyInstaller in your Python environment:
```bash
pip install pyinstaller
```

There are two primary methods for building the `.exe`. 

### Recommended Command (One-Folder)
By default, compiling the application to a single directory is highly recommended for stability and faster boot times. The executable pulls local DLLs directly from its accompanying `/dist/` folder immediately.
```bash
pyinstaller --windowed --add-data "src;src" --hidden-import=PyQt5 --hidden-import=matplotlib main.py
```

### Alternative Command (One-File)
If you require absolute portability where the entire program lives in literally a single `.exe` structure:
```bash
pyinstaller --windowed --onefile --add-data "src;src" --hidden-import=PyQt5 --hidden-import=matplotlib main.py
```
> [!IMPORTANT]
> The `--onefile` format possesses a noticeably slower boot sequence when first launched. Behind the scenes, the `.exe` must actively decompress itself into a temporary AppData folder on the operating system before establishing the PyQt interface, which takes several seconds longer than the "One-Folder" output method.

### Custom Application Icon (Optional)
To brand your software professionally, generate an `.ico` image file, save it to the root path, and append it:
```bash
pyinstaller --windowed --add-data "src;src" --icon=app.ico main.py
```

---

## 2. Handling Dependencies & Project Files

When generating the build, specific command flags are critical to prevent silent failures on target machines:
- `--windowed` (or `-w`): Forces Windows to suppress launching a background command-prompt terminal console.
- `--add-data "src;src"`: Informs PyInstaller that our `src` folder (containing `main_ui.py`, `combustion.py`, etc.) must be literally transported and maintained alongside the core script. If this is omitted, the EXE will crash immediately hunting for Physics models.
- `--hidden-import`: In complex environments where Dynamic C-Libraries load dynamically (like `PyQt5` Qt engines and `matplotlib` plotting bases), PyInstaller sometimes misses dependencies. Explicitly flagging them guarantees they bundle.

---

## 3. Output Structure

Once PyInstaller finishes its build sequence, the `.exe` output resides exactly in the generated `./dist/` directory.

**One-Folder Output:**
```text
/dist/
   /main/           <- Distributed package
      main.exe      <- Your Application!
      Qt5Core.dll
      python3.dll
      /src/
      ...
```

**One-File Output:**
```text
/dist/
   main.exe         <- Single self-extracting application
```

---

## 4. Fixing Common Issues

If the `.exe` refuses to launch on target equipment, identify the problem here:

| Error / Issue | Solution |
| :--- | :--- |
| **PyQt5 GUI doesn't open** | Disable `--windowed` during compile and launch the `.exe` via terminal. Check the fatal text readout! Ensure Windows Visual C++ Redistributables are updated on the target PC. |
| **Matplotlib Backend Error** | `main.py` has deliberately been injected with `matplotlib.use('Qt5Agg')` structurally. Ensure `--hidden-import=matplotlib` wasn't omitted during compression. |
| **Simulation Instability (NaN / Zero div)** | The application calculates offline. Verify mathematical edge limitations exist safely. Numerical errors on a target PC indicate CPU/Precision mismatch parameters compared to source processing limits. | 

---

## 5. Testing the Executable

**Local Test Protocol**
1. Navigate directly to `/dist/main/`. 
2. Double-click `main.exe`.
3. Validate that standard simulation bounds execute and visual Canvas elements (`Graph`, `Animation`) render perfectly.

**Isolated Test Protocol (No-Python PC)**
1. Transfer the `.exe` (or the `/main/` folder) to a secondary Windows laptop containing **no Python installations**.
2. Run identical simulation tests.
3. Validate that Qt threads and UI elements dynamically scale accurately without hunting for local `.pyc` libraries.

---

## 6. Final User Instructions (Delivery)

When delivering the software to a client, an engineer, or a classroom:

1. Provide the complete `dist/main` directory. 
2. Optionally pack it via a standard installer packaging software like **Inno Setup** (`.iss`), which bundles everything into a professional `setup.exe` installing files automatically inside `C:\Program Files\`.
3. Instruct users:
   - Double-click `main.exe`
   - Review Input Properties securely.
   - Operate intuitively via the `Simulation` routines!
