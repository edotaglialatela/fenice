# --- File: validation/validationCavityConvergence.py ---
import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path: sys.path.insert(0, ROOT_DIR)

def validate_cavity_convergence():
    FOLDERS = ["lid_driven_cavity_0", "lid_driven_cavity_1", "lid_driven_cavity_2", "lid_driven_cavity_3"]
    BASE_RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))
    profiles_data = []

    # for folder in FOLDERS:
    #     vtr = sorted(glob.glob(os.path.join(BASE_RESULTS_DIR, folder, "*.vtr")))[-1]
    #     root = ET.parse(vtr).getroot()
    #     extent_vals = list(map(int, root.find('.//Piece').attrib['Extent'].split()))
    #     nx = extent_vals[1] - extent_vals[0]
    #     ny = extent_vals[3] - extent_vals[2]
        
    #     vel = np.fromstring(root.find('.//CellData/DataArray[@Name="Velocity"]').text, sep=' ').reshape((ny, nx, 3))
    #     u, v = vel[:, :, 0], vel[:, :, 1]
        
    #     # Divergenza calcolata numericamente con differenze finite centrali
    #     du_dx = (u[:, 2:] - u[:, :-2]) / (2.0 / nx)
    #     dv_dy = (v[2:, :] - v[:-2, :]) / (2.0 / ny)
    #     div_u = np.abs(du_dx[1:-1, :] + dv_dy[:, 1:-1])
    #     print(f"[{folder}] Max |div(u)|: {np.max(div_u):.4e}")
        
    #     profiles_data.append({'u': u[:, nx//2], 'v': v[ny//2, :], 'ny': ny, 'nx': nx})

    # Ghia Reference Data
    ghia_y, ghia_u = np.array([1.0, 0.9766, 0.9688, 0.8516, 0.7344, 0.5, 0.2813, 0.1016, 0.0]), np.array([1.0, 0.5808, 0.4724, 0.1712, 0.0519, -0.1148, -0.3273, -0.2139, 0.0])
    ghia_x, ghia_v = np.array([1.0, 0.9688, 0.9063, 0.8594, 0.5, 0.2344, 0.0938, 0.0]), np.array([0.0, -0.1215, -0.2383, -0.2245, 0.0519, 0.3020, 0.2222, 0.0])

    for folder in FOLDERS:
        vtr = sorted(glob.glob(os.path.join(BASE_RESULTS_DIR, folder, "*.vtr")))[-1]
        root = ET.parse(vtr).getroot()
        extent = list(map(int, root.find('.//Piece').attrib['Extent'].split()))
        nx, ny = extent[1] - extent[0], extent[3] - extent[2]
        
        vel_flat = np.fromstring(root.find('.//CellData/DataArray[@Name="Velocity"]').text, sep=' ')
        vel_3d = vel_flat.reshape((ny, nx, 3))
        profiles_data.append({'u': vel_3d[:, :, 0], 'v': vel_3d[:, :, 1], 'ny': ny, 'nx': nx})

    # Error Calculation
    err_ghia = []
    for p in profiles_data:
        u_interp = np.interp(ghia_y, np.linspace(0, 1, p['ny']), p['u'][:, p['nx']//2][::-1])
        err_ghia.append((p['ny'], np.sqrt(np.mean((u_interp - ghia_u)**2))))
    
    err_self = []
    ref = profiles_data[-1]
    for p in profiles_data[:-1]:
        u_interp = np.interp(np.linspace(0, 1, ref['ny']), np.linspace(0, 1, p['ny']), p['u'][:, p['nx']//2])
        err_self.append((p['ny'], np.sqrt(np.mean((u_interp - ref['u'][:, ref['nx']//2])**2))))

    # --- 3. PLOTTING ---
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm'})
    
    # FIG 1: Comparison vs Ghia and numerical convergence
    fig1 = plt.figure(figsize=(18, 6))
    gs = fig1.add_gridspec(1, 3)
    ax1, ax2, ax3 = fig1.add_subplot(gs[0, 0]), fig1.add_subplot(gs[0, 1]), fig1.add_subplot(gs[0, 2])
    
    ax1.plot(ghia_u, ghia_y, 'k-', lw=2, label='Ghia (1982)')
    ax2.plot(ghia_x, ghia_v, 'k-', lw=2, label='Ghia (1982)')
    colors = plt.cm.Blues(np.linspace(0.5, 1.0, len(profiles_data)))
    for p, col in zip(profiles_data, colors):
        ax1.plot(p['u'][:, p['nx']//2], np.linspace(0,1,p['ny']), 'o', mfc='none', mec=col, ms=5, label=f'Ny={p["ny"]}')
        ax2.plot(np.linspace(0,1,p['nx']), p['v'][p['ny']//2, :], 'o', mfc='none', mec=col, ms=5, label=f'Nx={p["nx"]}')
    
    # Errors
    ax3.loglog([e[0] for e in err_ghia], [e[1] for e in err_ghia], 's-', color='darkred', label='vs Ghia')
    ax3.loglog([e[0] for e in err_self], [e[1] for e in err_self], 'o-', color='darkblue', label='Self-Conv')
    ax3.set_title("L2 Norm Error")
    
    for ax in [ax1, ax2, ax3]: ax.grid(True, linestyle='--', alpha=0.5); ax.legend()
    plt.tight_layout()
    fig1.savefig(os.path.join(BASE_RESULTS_DIR, "validation_profiles.pdf"), dpi=300)
    print(f"[PDF] Salvato: validation_profiles.pdf")

    # FIG 2: Streamlines
    fig2 = plt.figure(figsize=(10, 8))
    ax4 = fig2.add_subplot(1, 1, 1)
    p_ref = profiles_data[-1]
    X, Y = np.meshgrid(np.linspace(0, 1, p_ref['nx']), np.linspace(0, 1, p_ref['ny']))
    strm = ax4.streamplot(X, Y, p_ref['u'], p_ref['v'], color=np.sqrt(p_ref['u']**2 + p_ref['v']**2), cmap='viridis', density=1.5)
    ax4.set_title(f"Streamlines (Resolution {p_ref['nx']}x{p_ref['ny']})")
    plt.colorbar(strm.lines, ax=ax4, label='Velocity Magnitude')
    
    fig2.savefig(os.path.join(BASE_RESULTS_DIR, "validation_streamlines.pdf"), dpi=300)
    print(f"[PDF] Salvato: validation_streamlines.pdf")

if __name__ == "__main__":
    validate_cavity_convergence()