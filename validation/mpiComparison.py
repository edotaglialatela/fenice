# --- File: validation/validationPoiseuille_MPI.py ---
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

def compare_poiseuille_mpi():
    # --- 1. CONFIGURATION ---
    U_IN = 0.01  
    
    # Cartelle per il confronto
    FOLDERS = ["poiseuille_flow", "poiseuille_flow_mpi"]
    RUN_LABELS = ["Seriale (No MPI)", "MPI"]
    
    BASE_RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))
    
    profiles_data = []

    print("=== HAGEN-POISEUILLE: SERIAL VS MPI COMPARISON ===")
    
    # --- 2. AUTONOMOUS DATA EXTRACTION ---
    for i, folder in enumerate(FOLDERS):
        results_dir = os.path.join(BASE_RESULTS_DIR, folder)
        vtr_files = sorted(glob.glob(os.path.join(results_dir, "*.vtr")))
        
        if not vtr_files:
            print(f"[WARNING] Nessun file .vtr trovato in {folder}. Salto.")
            continue
            
        last_vtr = vtr_files[-1]
        print(f"\n--- Analizzando: {folder} ({RUN_LABELS[i]}) -> {os.path.basename(last_vtr)} ---")
        
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
        mass_flow = np.sum(rho_num * u_num) * dy
        
        print(f"Mesh Dedotta:     {nx} x {ny} celle (Lx={Lx:.2f}m, Ly={Ly:.2f}m)")
        print(f"Piano estrazione: x = {x_eval:.2f} m")
        print(f"Portata di MASSA: {mass_flow:.5f} kg/s (Attesa: {1.0 * U_IN * Ly:.5f})")
        print(f"Picco Velocità:   {np.max(u_num):.5f} m/s  (Attesa: {np.max(u_ana):.5f})")
        print(f"Errore L2 Norm:   {l2_error:.4e}")
        
        # Store for plotting
        profiles_data.append({
            'label': RUN_LABELS[i], 'y_c': y_c, 'u_num': u_num, 
            'nx': nx, 'ny': ny, 'Ly': Ly, 'l2_error': l2_error
        })

    if len(profiles_data) < 2:
        print("\n[ERRORE] Non ho trovato i dati per entrambe le esecuzioni. Esco dallo script.")
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
    fig.suptitle('Hagen-Poiseuille Flow: Serial vs MPI Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)

    # === SUBPLOT 1: VELOCITY PROFILES ===
    Ly_ref = profiles_data[-1]['Ly']
    y_fine = np.linspace(0, Ly_ref, 200)
    u_ana_fine = (6.0 * U_IN / Ly_ref**2) * y_fine * (Ly_ref - y_fine)
    
    ax1.plot(u_ana_fine, y_fine, color='#2c3e50', linestyle='-', linewidth=2.5, 
             label='Soluzione Analitica', zorder=1)
    
    colors = ['#2980b9', '#e74c3c'] # Blu per seriale, Rosso per MPI
    markers = ['o', 's']
    
    for p, color, marker in zip(profiles_data, colors, markers):
        ax1.plot(p['u_num'], p['y_c'], marker=marker, linestyle='', color=color,
                 markeredgecolor='white', markeredgewidth=0.5, alpha=0.8,
                 markersize=7, label=f"{p['label']} (CFD)", zorder=2)
                    
    ax1.set_xlabel('Velocità, $u$ [m/s]')
    ax1.set_ylabel('Coordinata Verticale, $y$ [m]')
    ax1.set_title('Profili di Velocità Sviluppati')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower center', frameon=True, shadow=True)
    ax1.set_ylim([0, Ly_ref])
    ax1.set_xlim(left=0.0)
    
    # === SUBPLOT 2: ABSOLUTE DIFFERENCE (SERIAL VS MPI) ===
    # Controlliamo che le mesh siano uguali per fare la sottrazione punto a punto
    if profiles_data[0]['ny'] == profiles_data[1]['ny']:
        diff_u = np.abs(profiles_data[0]['u_num'] - profiles_data[1]['u_num'])
        y_c_common = profiles_data[0]['y_c']
        
        ax2.plot(diff_u, y_c_common, marker='D', color='#8e44ad', linestyle='-', 
                 linewidth=2, markersize=6, label='|Seriale - MPI|', zorder=3)
        
        ax2.set_xlabel('Differenza Assoluta di Velocità [m/s]')
        ax2.set_ylabel('Coordinata Verticale, $y$ [m]')
        ax2.set_title('Discrepanza Puntuale (Seriale vs MPI)')
        
        # Aggiungiamo un testo per far notare la tolleranza
        max_diff = np.max(diff_u)
        ax2.annotate(f"Max Diff: {max_diff:.3e}", xy=(0.5, 0.9), xycoords='axes fraction',
                     ha='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
                     
    else:
        # Se le mesh sono diverse, mostriamo un grafico a barre degli errori L2
        labels = [p['label'] for p in profiles_data]
        l2_errs = [p['l2_error'] for p in profiles_data]
        ax2.bar(labels, l2_errs, color=colors, edgecolor='black')
        ax2.set_ylabel('Errore Norma $L_2$')
        ax2.set_title('Confronto Errori (Mesh Differenti)')

    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', frameon=True, shadow=True)
    
    plt.tight_layout()
    
    output_img = os.path.join(BASE_RESULTS_DIR, "mpi_comparison_report.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"\n[PLOT] Grafico salvato ad alta risoluzione in: {output_img}")
    
    plt.show()

if __name__ == "__main__":
    compare_poiseuille_mpi()