# --- File: validation/validationPoiseuille.py ---
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from vtkio import vtk_array

# Resolve project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def validate_poiseuille_convergence():
    # --- 1. CONFIGURATION ---
    U_IN = 0.01  
    
    FOLDERS = ["poiseuille_flow_2", "poiseuille_flow_3", "poiseuille_flow_4"]
    ALPHAS = [0.3, 0.65, 1.0] 
    
    BASE_RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))
    
    convergence_data = []
    profiles_data = []

    print("=== HAGEN-POISEUILLE GRID CONVERGENCE ===")
    
    # --- 2. AUTONOMOUS DATA EXTRACTION ---
    for i, folder in enumerate(FOLDERS):
        results_dir = os.path.join(BASE_RESULTS_DIR, folder)
        vtr_files = sorted(glob.glob(os.path.join(results_dir, "*.vtr")))
        
        if not vtr_files:
            print(f"[WARNING] No .vtr files found in {folder}. Skipping.")
            continue
            
        last_vtr = vtr_files[-1]
        print(f"\n--- Analyzing: {folder} -> {os.path.basename(last_vtr)} ---")
        
        # XML Parsing
        tree = ET.parse(last_vtr)
        root = tree.getroot()
        
        # (A) Extract Mesh Dimensions Autonomously
        coords = root.findall('.//Coordinates/DataArray')
        x_nodes = vtk_array(coords[0])
        y_nodes = vtk_array(coords[1])
        
        Lx = x_nodes[-1] - x_nodes[0]
        Ly = y_nodes[-1] - y_nodes[0]
        nx = len(x_nodes) - 1
        ny = len(y_nodes) - 1
        dy = Ly / ny
        
        # (B) Extract Velocity
        vel_flat = vtk_array(root.find('.//CellData/DataArray[@Name="Velocity"]'))
        vel_3d = vel_flat.reshape((ny, nx, 3))
        u_vel = vel_3d[:, :, 0]
        
        # (C) Extract Density (for Mass Flow Check)
        rho_flat = vtk_array(root.find('.//CellData/DataArray[@Name="Density"]'))
        rho_2d = rho_flat.reshape((ny, nx))
        
        # (D) Slice Extraction (95% of domain length)
        ix_eval = int(nx * 0.95)
        x_eval = (ix_eval + 0.5) * (Lx / nx)
        u_num = u_vel[:, ix_eval]
        rho_num = rho_2d[:, ix_eval]
        
        y_c = np.linspace(dy/2.0, Ly - dy/2.0, ny)
        
        # (E) Analytical Solution
        u_ana = (6.0 * U_IN / Ly**2) * y_c * (Ly - y_c)
        
        # (F) Error & Flow Computations
        l2_error = np.sqrt(np.mean((u_num - u_ana)**2)) / np.max(u_ana)
        vol_flow = np.sum(u_num) * dy
        mass_flow = np.sum(rho_num * u_num) * dy
        
        print(f"Mesh Dedotta:     {nx} x {ny} celle (Lx={Lx:.2f}m, Ly={Ly:.2f}m)")
        print(f"Piano estrazione: x = {x_eval:.2f} m")
        print(f"Portata di MASSA: {mass_flow:.5f} kg/s (Attesa: {1.0 * U_IN * Ly:.5f})")
        print(f"Picco Velocità:   {np.max(u_num):.5f} m/s  (Attesa: {np.max(u_ana):.5f})")
        print(f"Errore L2 Norm:   {l2_error:.4e}")
        
        # Store for plotting
        convergence_data.append((dy, l2_error, ny))
        profiles_data.append({
            'y_c': y_c, 'u_num': u_num, 'nx': nx, 'ny': ny, 
            'alpha': ALPHAS[i], 'Ly': Ly
        })

    if not convergence_data:
        print("\n[ERROR] Nessun dato estratto. Esco dallo script.")
        return

    # --- 3. PLOTTING ---
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Hagen-Poiseuille Flow: CFD Validation & Grid Convergence', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    cmap = plt.cm.Blues
    colors = cmap(np.linspace(0.4, 1.0, len(profiles_data)))

    # === SUBPLOT 1: VELOCITY PROFILES ===
    Ly_ref = profiles_data[-1]['Ly']
    y_fine = np.linspace(0, Ly_ref, 200)
    u_ana_fine = (6.0 * U_IN / Ly_ref**2) * y_fine * (Ly_ref - y_fine)
    
    ax1.plot(u_ana_fine, y_fine, color='#2c3e50', linestyle='-', linewidth=2.5, 
             label='Exact Analytical Solution', zorder=1)
    
    for p, color in zip(profiles_data, colors):
        ax1.plot(p['u_num'], p['y_c'], marker='o', linestyle='', color=color,
                 markeredgecolor='white', markeredgewidth=0.5,
                 markersize=7, label=f"CFD ($N_y={p['ny']}$)", zorder=2)
                    
    ax1.set_xlabel('Velocity, $u$ [m/s]')
    ax1.set_ylabel('Vertical Coordinate, $y$ [m]')
    ax1.set_title('Fully Developed Velocity Profiles')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower center', frameon=True, shadow=True)
    ax1.set_ylim([0, Ly_ref])
    ax1.set_xlim(left=0.0)
    
    # === SUBPLOT 2: GRID CONVERGENCE (LOG-LOG) ===
    dys = np.array([d[0] for d in convergence_data])
    errs = np.array([d[1] for d in convergence_data])
    nys = [d[2] for d in convergence_data]
    
    ax2.loglog(dys, errs, marker='s', color='#2980b9', markerfacecolor='#3498db', 
               markeredgecolor='black', markersize=8, linestyle='-', linewidth=2, 
               label='CFD $L_2$ Error', zorder=4)
               
    if len(dys) > 1:
        slope, intercept = np.polyfit(np.log(dys), np.log(errs), 1)
        trend_errs = np.exp(intercept) * dys**slope
        
        ax2.loglog(dys, trend_errs, color='#e74c3c', linestyle='--', linewidth=2, 
                   label=rf'Empirical Fit ($\mathcal{{O}} \approx {slope:.2f}$)', zorder=3)
        print(f"\n{'='*40}")
        print(f">>> ORDINE DI ACCURATEZZA RILEVATO: {slope:.2f} <<<")
        print(f"{'='*40}\n")

        ref_dy = dys
        ref_err_1st = errs[0] * (ref_dy / ref_dy[0])**1.0
        # ref_err_2nd = errs[0] * (ref_dy / ref_dy[0])**2.0
        
        ax2.loglog(ref_dy, ref_err_1st, color='gray', linestyle=':', linewidth=1.5, 
                   label='Ideal 1st Order ($\mathcal{O}(h)$)', zorder=1)
        # ax2.loglog(ref_dy, ref_err_2nd, color='black', linestyle=':', linewidth=1.5, 
        #            label='Ideal 2nd Order ($\mathcal{O}(h^2)$)', zorder=1)
    
    for i, txt in enumerate(nys):
        ax2.annotate(f"$N_y={txt}$", (dys[i], errs[i]), textcoords="offset points", 
                     xytext=(0, -18), ha='center', fontsize=9, 
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
                     
    ax2.set_xlabel('Grid Spacing, $\Delta y$ [m]')
    ax2.set_ylabel('Normalized $L_2$ Error')
    ax2.set_title('Grid Convergence Study')
    ax2.grid(True, which="both", linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right', frameon=True, shadow=True)
    
    plt.tight_layout()
    
    output_img = os.path.join(BASE_RESULTS_DIR, "grid_convergence_report.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"[PLOT] Grafico salvato ad alta risoluzione in: {output_img}")
    
    plt.show()

if __name__ == "__main__":
    validate_poiseuille_convergence()