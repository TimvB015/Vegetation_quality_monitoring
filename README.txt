This is a step by step instruction on initializing Pixi in VS code and adding the GitHub repo.

=====================================
USING THE PIXI ENVIRONMENT IN VS CODE
=====================================

1. **Clone the repository**
   - Repository location:  
     ```
     https://github.com/TimvB015/Vegetation_quality_monitoring.git
     ```
   - Use Git to clone the project to your working directory:
     ```
     git clone https://github.com/TimvB015/Vegetation_quality_monitoring.git
     ```


2. **Install Pixi**
   - Open PowerShell.
   - Download and install Pixi by running:
     ```
     irm https://pixi.sh/install.ps1 | iex
     ```
   - This will download `pixi.exe` and place it in your user folder (typically `C:\Users\<YourUser>\.pixi\bin`).
   - Add to the pixi-folder the following files found in the repo under the pixi_installation folder:
      -> config.toml 
      -> ruff.toml 


3. **Open VS Code**
   - Launch VS Code and open your cloned project folder.

4. **Open the terminal in VS Code**
   - Go to Terminal > New Terminal.
   - Ensure you are in your project folder (e.g., `PS C:\...\YourRepoName>`).

5. **Activate and sync the Pixi environment**
   - Run:
     ```
     pixi install
     pixi run setup
     pixi run jupyter
     ```
   - Pixi will automatically install and resolve all dependencies from `pixi.toml` and `pixi.lock`. This may take a while on first setup.

6. **(Optional, if you encounter Execution Policies limitations)**
   - Run:
     ```
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
     ```

7. **If Pixi gives SSL errors (e.g., "UnknownIssuer")**
   - Before running Pixi commands, set:
     ```
     $env:CONDA_SSL_VERIFY = "false"
     ```
   - Then run Pixi as above.

8. **If Pixi fails to install packages (e.g. ipykernel not installed), manually bootstrap pip and ipykernel in the Pixi shell:**
   a) Run:
      ```
      python -m ensurepip
      ```
   b) Run:
      ```
      python -m pip install --upgrade pip
      ```
   c) Run:
      ```
      python -m pip install ipykernel
      ```

9. **Restart VS Code**

10. **Open your Jupyter notebook (.ipynb) in VS Code**

11. **Select the kernel**
    - Click "Select Kernel" in the top right of your notebook.
    - Look under "Jupyter Kernels" for the name you specified (e.g., "Python (vegqual_env)").
    - Refresh the kernel list if needed and select your environment.

---

**Troubleshooting:**
- If you have persistent SSL issues, contact your IT department about trusted certificates or try on an unrestricted network.
- If the kernel does not appear, double-check that you registered it with `python -m ipykernel install ...` in the Pixi shell.
- Use `pip list` in the Pixi shell to see which Python packages are installed.


================
FOLDER STRUCTURE
================

Main_dir/
├── README.txt
├── pixi.lock
├── pyproject.toml
├── pixi_installation/
│   ├── config.toml
│   ├── ruff.toml
│   └── commands.txt
├── functions/
│   └── *.py
├── notebooks_dir/
│   ├── _00_common_dfs/
│   │   ├── _support/
│   │   │   └── *.py
│   │   ├── _n01_habitat_reference_dfs.ipynb
│   │   └── *.ipynb
│   └── _.._.../
│       ├── _support/
│       │   └── *.py
│       └── *.ipynb
└── paths/
    └── OG.paths.py
