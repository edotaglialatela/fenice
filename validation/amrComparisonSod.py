import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# Resolve project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

class ExactRiemannSolver:
    """
    Exact Analytical Solver for the 1D Riemann Problem (Toro, 2009).
    Calculates the exact solution for the Sod Shock Tube at a given time t.
    """
    def __init__(self, rho_L=1.0, p_L=1.0, u_L=0.0, rho_R=0.125, p_R=0.1, u_R=0.0, gamma=1.4):
        self.rhoL, self.pL, self.uL = rho_L, p_L, u_L
        self.rhoR, self.pR, self.uR = rho_R, p_R, u_R
        self.gamma = gamma
        self.cL = np.sqrt(gamma * p_L / rho_L)
        self.cR = np.sqrt(gamma * p_R / rho_R)
        
    def _fk(self, p, dk, pk, ck):
        A = 2.0 / ((self.gamma + 1.0) * dk)
        B = (self.gamma - 1.0) / (self.gamma + 1.0) * pk
        if p > pk:
            f = (p - pk) * np.sqrt(A / (p + B))
            df = np.sqrt(A / (p + B)) * (1.0 - (p - pk) / (2.0 * (p + B)))
        else:
            pratio = p / pk
            f = (2.0 * ck / (self.gamma - 1.0)) * (pratio**((self.gamma - 1.0) / (2.0 * self.gamma)) - 1.0)
            df = (1.0 / (dk * ck)) * pratio**(-(self.gamma + 1.0) / (2.0 * self.gamma))
        return f, df

    def solve_star_region(self):
        p_star = 0.5 * (self.pL + self.pR)
        tol = 1e-6
        for _ in range(100):
            fL, dfL = self._fk(p_star, self.rhoL, self.pL, self.cL)
            fR, dfR = self._fk(p_star, self.rhoR, self.pR, self.cR)
            f = fL + fR + self.uR - self.uL
            df = dfL + dfR
            dp = -f / df
            p_star += dp
            if abs(dp) < tol * p_star:
                break
        u_star = 0.5 * (self.uL + self.uR + fR - fL)
        return p_star, u_star

    def sample(self, x, t, x0=0.5):
        p_star, u_star = self.solve_star_region()
        
        if p_star > self.pL:
            r = p_star / self.pL
            rho_starL = self.rhoL * (r + (self.gamma-1)/(self.gamma+1)) / (r * (self.gamma-1)/(self.gamma+1) + 1.0)
        else:
            rho_starL = self.rhoL * (p_star / self.pL)**(1.0 / self.gamma)
            
        if p_star > self.pR:
            r = p_star / self.pR
            rho_starR = self.rhoR * (r + (self.gamma-1)/(self.gamma+1)) / (r * (self.gamma-1)/(self.gamma+1) + 1.0)
        else:
            rho_starR = self.rhoR * (p_star / self.pR)**(1.0 / self.gamma)
            
        c_starL = np.sqrt(self.gamma * p_star / rho_starL)
        c_starR = np.sqrt(self.gamma * p_star / rho_starR)
        
        S = (x - x0) / t if t > 0 else 0.0
        
        if S < u_star:
            if p_star > self.pL:
                S_L = self.uL - self.cL * np.sqrt((self.gamma+1)/(2*self.gamma) * p_star/self.pL + (self.gamma-1)/(2*self.gamma))
                if S < S_L: return self.rhoL, self.uL, self.pL
                else:       return rho_starL, u_star, p_star
            else:
                S_HL = self.uL - self.cL
                S_TL = u_star - c_starL
                if S < S_HL:   return self.rhoL, self.uL, self.pL
                elif S > S_TL: return rho_starL, u_star, p_star
                else:
                    u_fan = 2.0/(self.gamma+1) * (self.cL + (self.gamma-1)/2.0 * self.uL + S)
                    c_fan = 2.0/(self.gamma+1) * (self.cL + (self.gamma-1)/2.0 * (self.uL - S))
                    p_fan = self.pL * (c_fan / self.cL)**(2.0 * self.gamma / (self.gamma - 1.0))
                    rho_fan = self.rhoL * (c_fan / self.cL)**(2.0 / (self.gamma - 1.0))
                    return rho_fan, u_fan, p_fan
        else:
            if p_star > self.pR:
                S_R = self.uR + self.cR * np.sqrt((self.gamma+1)/(2*self.gamma) * p_star/self.pR + (self.gamma-1)/(2*self.gamma))
                if S > S_R: return self.rhoR, self.uR, self.pR
                else:       return rho_starR, u_star, p_star
            else:
                S_HR = self.uR + self.cR
                S_TR = u_star + c_starR
                if S > S_HR:   return self.rhoR, self.uR, self.pR
                elif S < S_TR: return rho_starR, u_star, p_star
                else:
                    u_fan = 2.0/(self.gamma+1) * (-self.cR + (self.gamma-1)/2.0 * self.uR + S)
                    c_fan = 2.0/(self.gamma+1) * (self.cR - (self.gamma-1)/2.0 * (self.uR - S))
                    p_fan = self.pR * (c_fan / self.cR)**(2.0 * self.gamma / (self.gamma - 1.0))
                    rho_fan = self.rhoR * (c_fan / self.cR)**(2.0 / (self.gamma - 1.0))
                    return rho_fan, u_fan, p_fan


def extract_vtk_data(filepath):
    """
    Recursively reads VTK files. If it finds a .vtm index file, it enters the folder,
    reads all .vtu/.vtr blocks, and concatenates the results.
    """
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except Exception as e:
        print(f"      [!] Cannot read {filepath}: {e}")
        return [], [], 0

    base_dir = os.path.dirname(filepath)
    x_all, rho_all = [], []
    total_cells = 0

    # 1. Multi-Block File Case (.vtm)
    if root.tag == "VTKFile" and root.attrib.get("type") == "vtkMultiBlockDataSet":
        for dataset in root.findall('.//DataSet'):
            rel_path = dataset.attrib.get('file')
            if rel_path:
                child_path = os.path.join(base_dir, rel_path)
                if os.path.exists(child_path):
                    x_c, rho_c, t_c = extract_vtk_data(child_path)
                    x_all.extend(x_c)
                    rho_all.extend(rho_c)
                    total_cells += t_c
        return np.array(x_all), np.array(rho_all), total_cells

    # 2. Standard Data File Case (.vtr, .vtu)
    def get_array(name):
        nodes = root.findall(f'.//CellData/DataArray[@Name="{name}"]')
        if not nodes:
            nodes = root.findall(f'.//DataArray[@Name="{name}"]')
        if not nodes:
            return None
        return np.concatenate([np.array(n.text.split(), dtype=float) for n in nodes])

    rho = get_array("Density")
    if rho is None:
        return [], [], 0
        
    t_c = len(rho)
    x_cells = None
    
    piece = root.find('.//Piece')
    if piece is not None and 'Extent' in piece.attrib:
        # Structured Grid (VTR)
        ext = list(map(int, piece.attrib['Extent'].split()))
        nx = ext[1] - ext[0]
        ny = ext[3] - ext[2]
        if ny == 0: ny = 1
        
        x_nodes_xml = root.find('.//Coordinates/DataArray[@Name="x"]')
        if x_nodes_xml is not None:
            x_edges = np.array(x_nodes_xml.text.split(), dtype=float)
            x_cells = 0.5 * (x_edges[:-1] + x_edges[1:])
        else:
            x_cells = np.linspace(0.0, 1.0, nx)
            
        if ny > 1:
            rho_2d = rho.reshape((ny, nx))
            rho = rho_2d[ny // 2, :]
    else:
        # Unstructured Grid or AMR Block (VTU)
        x_centers = get_array("x_center")
        if x_centers is not None:
            x_cells = x_centers
        else:
            # Generic fallback if centers and extents are not saved
            x_cells = np.linspace(0.0, 1.0, len(rho))
                
    return np.array(x_cells), np.array(rho), t_c


def compare_amr_performance():
    # --- 1. CONFIGURATION ---
    FOLDERS = ["sod_shock_tube_constant", "sod_shock_tube_amr"]
    RUN_LABELS = ["Uniform Mesh", "AMR Mesh"]
    
    BASE_RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))
    
    profiles_data = []
    print("=== SOD SHOCK TUBE: UNIFORM VS AMR COMPARISON ===")
    
    exact_solver = ExactRiemannSolver()

    # --- 2. AUTONOMOUS DATA EXTRACTION ---
    for i, folder in enumerate(FOLDERS):
        results_dir = os.path.join(BASE_RESULTS_DIR, folder)
        pvd_files = glob.glob(os.path.join(results_dir, "*.pvd"))
        
        if not pvd_files:
            print(f"[WARNING] No .pvd files found in {folder}. Skipping.")
            continue
            
        pvd_filepath = pvd_files[0]
        pvd_tree = ET.parse(pvd_filepath)
        last_dataset = pvd_tree.getroot().findall('.//DataSet')[-1]
        
        time = float(last_dataset.attrib['timestep'])
        vtr_filename = last_dataset.attrib['file']
        vtr_filepath = os.path.join(results_dir, vtr_filename)
        
        print(f"\n--- Analyzing: {folder} ({RUN_LABELS[i]}) -> {vtr_filename} (t={time:.4f}s) ---")
        
        # Uses recursive parser supporting .vtm format
        x_cells, rho_num, total_cells = extract_vtk_data(vtr_filepath)
        
        if len(rho_num) == 0:
            print(f"[ERROR] Cannot extract Density from {vtr_filename}. Format unsupported.")
            continue

        # L2 Error Calculation (interpolated on the extracted grid points)
        rho_exact_sampled = np.array([exact_solver.sample(x, time)[0] for x in x_cells])
        l2_error = np.sqrt(np.mean((rho_num - rho_exact_sampled)**2)) / np.max(rho_exact_sampled)
        
        print(f"Total Cells:       {total_cells}")
        print(f"L2 Error (Dens.):  {l2_error:.4e}")
        
        profiles_data.append({
            'label': RUN_LABELS[i], 
            'x_cells': x_cells, 
            'rho_num': rho_num, 
            'total_cells': total_cells,
            'time': time,
            'l2_error': l2_error
        })

    if len(profiles_data) < 2:
        print("\n[ERROR] Insufficient data for comparison. Exiting script.")
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

    eval_time = profiles_data[0]['time']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Sod Shock Tube ($t = {eval_time:.4f}$ s): Uniform vs AMR Performance', 
                 fontsize=16, fontweight='bold', y=1.02)

    # === SUBPLOT 1: DENSITY PROFILE ===
    x_fine = np.linspace(0.0, 1.0, 1000)
    rho_exact_fine = [exact_solver.sample(x, eval_time)[0] for x in x_fine]
    
    ax1.plot(x_fine, rho_exact_fine, color='#2c3e50', linestyle='-', linewidth=2.5, 
             label='Analytical Solution', zorder=1)
    
    colors = ['#2980b9', '#e74c3c'] 
    markers = ['o', '^']
    
    for p, color, marker in zip(profiles_data, colors, markers):
        # Because the recursive parser reads scattered blocks, points might not
        # be in order along the X axis. argsort puts them in order before plotting.
        sort_idx = np.argsort(p['x_cells'])
        
        ax1.plot(p['x_cells'][sort_idx], p['rho_num'][sort_idx], marker=marker, linestyle='', color=color,
                 markeredgecolor='white', markeredgewidth=0.5, alpha=0.85,
                 markersize=6, label=f"{p['label']}", zorder=2)
                 
    ax1.set_xlabel('Position, $x$ [m]')
    ax1.set_ylabel(r'Density, $\rho$ [kg/m$^3$]')
    ax1.set_title(r'Density Profile Comparison')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True, shadow=True)
    ax1.set_xlim([0, 1.0])
    
    # === SUBPLOT 2: EFFICIENCY (CELL COUNT VS L2 ERROR) ===
    labels = [p['label'] for p in profiles_data]
    cells = [p['total_cells'] for p in profiles_data]
    l2_errs = [p['l2_error'] for p in profiles_data]
    
    x_pos = np.arange(len(labels))
    width = 0.4
    
    # Updated Colors to Blue and Dark Green
    bar1 = ax2.bar(x_pos, cells, width, color=['#7fb3d5', "#2af74c"], edgecolor='black', alpha=0.8) 
    ax2.set_ylabel('Total Number of Cells', color='#34495e', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.tick_params(axis='y', labelcolor='#34495e')
    
    for idx, rect in enumerate(bar1):
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., height + (max(cells)*0.02),
                 f"{int(height)}", ha='center', va='bottom', fontweight='bold')

    ax2_err = ax2.twinx()
    ax2_err.plot(x_pos, l2_errs, color='#8e44ad', marker='D', markersize=10, 
                 linestyle='--', linewidth=2, markeredgecolor='white', markeredgewidth=1.5)
    ax2_err.set_ylabel(r'$L_2$ Norm Error ($\rho$)', color='#8e44ad', fontweight='bold')
    ax2_err.tick_params(axis='y', labelcolor='#8e44ad')
    
    for i, err in enumerate(l2_errs):
        ax2_err.text(x_pos[i] + 0.1, err, f"{err:.2e}", color='#8e44ad', 
                     va='center', fontweight='bold', 
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#8e44ad", alpha=0.8))

    ax2.set_title('AMR Efficiency: Cells vs Error')
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    output_img = os.path.join(BASE_RESULTS_DIR, "sod_amr_comparison_report.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"\n[PLOT] High resolution plot saved to: {output_img}")
    
    plt.show()

if __name__ == "__main__":
    compare_amr_performance()