import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import RegularGridInterpolator, interp1d, make_interp_spline
from scipy.fft import fft, fftfreq
from vtkio import vtk_array

# =============================================================================
# PARAMETERS
# =============================================================================
PVD_FILE = "../results/ibm_flow_over_cylinder/ibm_flow_over_cylinder.pvd" # Path to your PVD file
RHO_INF = 1.0
U_INF = 1.0
D = 0.2
CX, CY = 1.0, 1.0

R_INT = D / 2.0 + 0.015 

# Expected theoretical/empirical values for Re = 100
ST_REF = 0.165
CD_REF = 1.35
CL_AMP_REF = 0.3
# =============================================================================

def parse_pvd(pvd_filepath):
    tree = ET.parse(pvd_filepath)
    root = tree.getroot()
    datasets = root.findall('.//DataSet')
    
    times = []
    vtr_files = []
    for ds in datasets:
        times.append(float(ds.attrib['timestep']))
        vtr_files.append(ds.attrib['file'])
        
    return times, vtr_files

def get_vtr_data(vtr_filepath):
    tree = ET.parse(vtr_filepath)
    root = tree.getroot()
    
    piece = root.find('.//Piece')
    extent = list(map(int, piece.attrib['Extent'].split()))
    nx, ny = extent[1], extent[3]
    
    x_nodes = vtk_array(root.find('.//Coordinates/DataArray[@Name="x"]'))
    y_nodes = vtk_array(root.find('.//Coordinates/DataArray[@Name="y"]'))
    
    x_cells = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    y_cells = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    
    p_node = root.find('.//DataArray[@Name="Pressure"]')
    p_data = vtk_array(p_node).reshape(ny, nx)
    
    return x_cells, y_cells, p_data

def calculate_forces(x_cells, y_cells, p_field):
    interp_p = RegularGridInterpolator((y_cells, x_cells), p_field, bounds_error=False, fill_value=None)
    
    N_theta = 360
    theta = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)
    
    x_circle = CX + R_INT * np.cos(theta)
    y_circle = CY + R_INT * np.sin(theta)
    
    pts = np.column_stack((y_circle, x_circle))
    p_circle = interp_p(pts)
    
    dtheta = 2 * np.pi / N_theta
    Fx = -np.sum(p_circle * np.cos(theta)) * R_INT * dtheta
    Fy = -np.sum(p_circle * np.sin(theta)) * R_INT * dtheta
    
    return Fx, Fy

def calculate_strouhal(t_history, Cl_history, D, U_inf, t_start=5.0):
    t_arr = np.array(t_history)
    cl_arr = np.array(Cl_history)
    
    valid_idx = t_arr >= t_start
    t_steady = t_arr[valid_idx]
    cl_steady = cl_arr[valid_idx]
    
    if len(t_steady) < 10:
        print("[FFT] Insufficient steady-state data for spectral analysis.")
        return None
        
    dt_uniform = np.mean(np.diff(t_steady))
    t_uniform = np.arange(t_steady[0], t_steady[-1], dt_uniform)
    
    interp_func = interp1d(t_steady, cl_steady, kind='cubic')
    cl_uniform = interp_func(t_uniform)
    
    cl_zero_mean = cl_uniform - np.mean(cl_uniform)
    
    N = len(t_uniform)
    yf = fft(cl_zero_mean)
    xf = fftfreq(N, dt_uniform)[:N//2]
    
    amplitudes = 2.0/N * np.abs(yf[0:N//2])
    
    idx_peak = np.argmax(amplitudes)
    f_shedding = xf[idx_peak]
    St = (f_shedding * D) / U_inf
    
    plt.figure(figsize=(9, 5))
    plt.plot(xf, amplitudes, color='#8E44AD', lw=2.5)
    plt.bar(xf, amplitudes, width=(xf[1]-xf[0])*0.8, color='#8E44AD', alpha=0.4)
    
    plt.plot(f_shedding, amplitudes[idx_peak], marker='o', markersize=9, 
             markerfacecolor="#F1C40F00", markeredgecolor='#D35400', markeredgewidth=2,
             label=f'Computed Peak: St = {St:.3f}\n(Expected Ref: St ≈ {ST_REF})')
    
    plt.title("Lift Coefficient Spectral Analysis (FFT)", fontsize=14, fontweight='bold')
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    
    # Auto-scaling
    max_x_view = min(xf[-1], f_shedding * 6)
    plt.xlim(0, max_x_view)
    plt.ylim(0, amplitudes[idx_peak] * 1.20)
    
    plt.grid(True, linestyle=':', alpha=0.7, color='gray')
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
    plt.tight_layout()
    
    out_img = os.path.join(os.path.dirname(PVD_FILE), "strouhal_fft.png")
    plt.savefig(out_img, dpi=200)
    plt.show()
    
    return St

if __name__ == "__main__":
    if not os.path.exists(PVD_FILE):
        print(f"[ERROR] PVD file not found: {PVD_FILE}")
        exit()
        
    pvd_dir = os.path.dirname(PVD_FILE)
    times, vtr_files = parse_pvd(PVD_FILE)
    
    q_dyn = 0.5 * RHO_INF * U_INF**2 * D
    
    t_history = []
    Cd_history = []
    Cl_history = []
    
    print(f"Starting data extraction from {len(times)} frames...")
    
    for i, (t, vtr_file) in enumerate(zip(times, vtr_files)):
        vtr_path = os.path.join(pvd_dir, vtr_file)
        x_cells, y_cells, p_data = get_vtr_data(vtr_path)
        
        Fx, Fy = calculate_forces(x_cells, y_cells, p_data)
        
        Cd = Fx / q_dyn
        Cl = Fy / q_dyn
        
        t_history.append(t)
        Cd_history.append(Cd)
        Cl_history.append(Cl)
        
        if i % 10 == 0:
            print(f"Processed frame {i}/{len(times)} (t = {t:.3f} s)")

    print("Extraction completed. Generating animation...")

    calculate_strouhal(t_history, Cl_history, D, U_INF, t_start=0.0) # skip some initial time if wanted

    # =============================================================================
    # ANIMATION MGLIORATA
    # =============================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Aerodynamic Coefficients Evolution (interpolated)", fontsize=16, fontweight='bold')
    
    # Axes setup Drag
    ax1.set_ylabel("Drag Coefficient ($C_d$)", fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.set_xlim(times[0], times[-1])
    ax1.set_ylim(min(Cd_history) - 0.1, max(Cd_history) + 0.1)
    
    ax1.axhline(CD_REF, color='gray', linestyle='--', alpha=0.6, label=f'Expected $C_d$ ≈ {CD_REF}')
    ax1.legend(loc='lower right', fontsize=10)

    # Axes setup Lift
    ax2.set_ylabel("Lift Coefficient ($C_l$)", fontsize=12)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.set_ylim(min(Cl_history) - 0.1, max(Cl_history) + 0.1)
    
    ax2.axhline(CL_AMP_REF, color='gray', linestyle='--', alpha=0.6, label=f'Expected $C_l$ amp ≈ $\pm${CL_AMP_REF}')
    ax2.axhline(-CL_AMP_REF, color='gray', linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right', fontsize=10)

    line_cd, = ax1.plot([], [], '#C0392B', lw=2.5)
    line_cl, = ax2.plot([], [], '#2980B9', lw=2.5)
    
    head_cd, = ax1.plot([], [], 'o', color='#C0392B', markersize=6)
    head_cl, = ax2.plot([], [], 'o', color='#2980B9', markersize=6)

    text_cd = ax1.text(0, 0, '', color='#C0392B', fontsize=10, fontweight='bold')
    text_cl = ax2.text(0, 0, '', color='#2980B9', fontsize=10, fontweight='bold')

    def init():
        line_cd.set_data([], [])
        line_cl.set_data([], [])
        head_cd.set_data([], [])
        head_cl.set_data([], [])
        text_cd.set_text('')
        text_cl.set_text('')
        return line_cd, line_cl, head_cd, head_cl, text_cd, text_cl

    def update(frame):
        t_curr_arr = np.array(t_history[:frame+1])
        cd_curr_arr = np.array(Cd_history[:frame+1])
        cl_curr_arr = np.array(Cl_history[:frame+1])
        
        t_curr = t_curr_arr[-1]
        cd_curr = cd_curr_arr[-1]
        cl_curr = cl_curr_arr[-1]

        if len(t_curr_arr) >= 4:
            t_smooth = np.linspace(t_curr_arr[0], t_curr_arr[-1], len(t_curr_arr) * 4)
            
            spline_cd = make_interp_spline(t_curr_arr, cd_curr_arr, k=3)
            spline_cl = make_interp_spline(t_curr_arr, cl_curr_arr, k=3)
            
            line_cd.set_data(t_smooth, spline_cd(t_smooth))
            line_cl.set_data(t_smooth, spline_cl(t_smooth))
        else:
            line_cd.set_data(t_curr_arr, cd_curr_arr)
            line_cl.set_data(t_curr_arr, cl_curr_arr)
        
        head_cd.set_data([t_curr], [cd_curr])
        head_cl.set_data([t_curr], [cl_curr])
        
        text_cd.set_position((t_curr, cd_curr))
        text_cd.set_text(f'  {cd_curr:.3f}') 
        
        text_cl.set_position((t_curr, cl_curr))
        text_cl.set_text(f'  {cl_curr:.3f}')
        
        return line_cd, line_cl, head_cd, head_cl, text_cd, text_cl

    skip = max(1, len(times) // 150)
    frames_to_render = range(0, len(times), skip)

    ani = animation.FuncAnimation(
        fig, update, frames=frames_to_render,
        init_func=init, blit=True, interval=50
    )

    out_gif = os.path.join(pvd_dir, "aerodynamics.gif")
    ani.save(out_gif, writer='pillow', fps=10)
    print(f"=== DONE! GIF saved in: {out_gif} ===")
    
    plt.tight_layout()
    plt.show()