import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import RegularGridInterpolator, make_interp_spline
from vtkio import vtk_array

# =============================================================================
# PARAMETERS
# =============================================================================
COARSE_PVD = "../results/ibm_flow_over_cylinder_lowres/ibm_flow_over_cylinder.pvd"
FINE_PVD = "../results/ibm_flow_over_cylinder/ibm_flow_over_cylinder.pvd"

RHO_INF = 1.0
U_INF = 1.0
D = 0.2
CX, CY = 1.0, 1.0
R_INT = D / 2.0 + 0.015 

# Expected reference values for Re = 100
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

def extract_data_from_pvd(pvd_file):
    if not os.path.exists(pvd_file):
        print(f"[ERROR] PVD not found: {pvd_file}")
        return [], [], []
        
    pvd_dir = os.path.dirname(pvd_file)
    times, vtr_files = parse_pvd(pvd_file)
    
    q_dyn = 0.5 * RHO_INF * U_INF**2 * D
    t_hist, cd_hist, cl_hist = [], [], []
    
    for i, (t, vtr_file) in enumerate(zip(times, vtr_files)):
        vtr_path = os.path.join(pvd_dir, vtr_file)
        if not os.path.exists(vtr_path):
            continue
            
        x_cells, y_cells, p_data = get_vtr_data(vtr_path)
        Fx, Fy = calculate_forces(x_cells, y_cells, p_data)
        
        t_hist.append(t)
        cd_hist.append(Fx / q_dyn)
        cl_hist.append(Fy / q_dyn)
        
        if i % 10 == 0:
            print(f"  Processed {i}/{len(times)} (t = {t:.3f} s)")
            
    return t_hist, cd_hist, cl_hist

if __name__ == "__main__":
    print("--- EXTRACTING COARSE MESH DATA ---")
    t_coarse, cd_coarse, cl_coarse = extract_data_from_pvd(COARSE_PVD)
    
    print("\n--- EXTRACTING FINE MESH DATA ---")
    t_fine, cd_fine, cl_fine = extract_data_from_pvd(FINE_PVD)
    
    if not t_coarse or not t_fine:
        print("[ERROR] Impossibile caricare i dati. Controlla i percorsi.")
        exit()

    # --- TIME STITCHING LOGIC ---
    if t_fine[0] < t_coarse[-1]:
        time_offset = t_coarse[-1]
        t_fine = [t + time_offset for t in t_fine]
        print(f"\n[INFO] Time Stitching applicato. Offset: +{time_offset:.3f} s")

    t_all = t_coarse + t_fine
    cd_all = cd_coarse + cd_fine
    cl_all = cl_coarse + cl_fine

    # =============================================================================
    # ANIMATION SETUP
    # =============================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle("Grid Sequencing: Mesh Refinement Impact on Aerodynamics", fontsize=15, fontweight='bold')
    
    # Drag Axis
    ax1.set_ylabel("Drag Coefficient ($C_d$)", fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.set_xlim(0, t_all[-1])
    ax1.set_ylim(min(cd_all) - 0.1, max(cd_all) + 0.1)
    ax1.axhline(CD_REF, color='gray', linestyle='--', alpha=0.6, label=f'Expected $C_d$ ≈ {CD_REF}')
    
    # Lift Axis
    ax2.set_ylabel("Lift Coefficient ($C_l$)", fontsize=12)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.set_ylim(min(cl_all) - 0.1, max(cl_all) + 0.1)
    ax2.axhline(CL_AMP_REF, color='gray', linestyle='--', alpha=0.6, label=f'Expected $C_l$ amp ≈ $\pm${CL_AMP_REF}')
    ax2.axhline(-CL_AMP_REF, color='gray', linestyle='--', alpha=0.6)
    
    switch_time = t_coarse[-1]
    ax1.axvline(switch_time, color='black', linestyle='-.', lw=1.5, alpha=0.8)
    ax2.axvline(switch_time, color='black', linestyle='-.', lw=1.5, alpha=0.8, label='Mesh Refinement (Hot Start)')
    
    ax1.legend(loc='lower right', fontsize=10)
    ax2.legend(loc='lower right', fontsize=10)

    line_cd_c, = ax1.plot([], [], color='#E74C3C', lw=1.5, alpha=0.6) # Rosso chiaro
    line_cd_f, = ax1.plot([], [], color='#900C3F', lw=2.5)            # Rosso scuro vivido
    
    line_cl_c, = ax2.plot([], [], color='#5DADE2', lw=1.5, alpha=0.6) # Blu chiaro
    line_cl_f, = ax2.plot([], [], color='#154360', lw=2.5)            # Blu scuro vivido
    
    head_cd, = ax1.plot([], [], 'o', color='black', markersize=6)
    head_cl, = ax2.plot([], [], 'o', color='black', markersize=6)
    text_cd = ax1.text(0, 0, '', color='black', fontsize=10, fontweight='bold')
    text_cl = ax2.text(0, 0, '', color='black', fontsize=10, fontweight='bold')

    def init():
        line_cd_c.set_data([], [])
        line_cd_f.set_data([], [])
        line_cl_c.set_data([], [])
        line_cl_f.set_data([], [])
        head_cd.set_data([], [])
        head_cl.set_data([], [])
        text_cd.set_text('')
        text_cl.set_text('')
        return line_cd_c, line_cd_f, line_cl_c, line_cl_f, head_cd, head_cl, text_cd, text_cl

    def get_smoothed_data(t_array, y_array, up_to_idx):
        """Helper function per fare lo smoothing sicuro (non incrocia i confini)"""
        t_slice = np.array(t_array[:up_to_idx+1])
        y_slice = np.array(y_array[:up_to_idx+1])
        if len(t_slice) >= 4:
            t_smooth = np.linspace(t_slice[0], t_slice[-1], len(t_slice) * 4)
            spline = make_interp_spline(t_slice, y_slice, k=3)
            return t_smooth, spline(t_smooth)
        return t_slice, y_slice

    def update(frame):
        t_curr = t_all[frame]
        cd_curr = cd_all[frame]
        cl_curr = cl_all[frame]
        
        N_coarse = len(t_coarse)
        
        if frame < N_coarse:
            t_s, cd_s = get_smoothed_data(t_coarse, cd_coarse, frame)
            _, cl_s = get_smoothed_data(t_coarse, cl_coarse, frame)
            
            line_cd_c.set_data(t_s, cd_s)
            line_cl_c.set_data(t_s, cl_s)
            
        else:
            t_s_c, cd_s_c = get_smoothed_data(t_coarse, cd_coarse, N_coarse - 1)
            _, cl_s_c = get_smoothed_data(t_coarse, cl_coarse, N_coarse - 1)
            line_cd_c.set_data(t_s_c, cd_s_c)
            line_cl_c.set_data(t_s_c, cl_s_c)
            
            idx_fine = frame - N_coarse
            t_s_f, cd_s_f = get_smoothed_data(t_fine, cd_fine, idx_fine)
            _, cl_s_f = get_smoothed_data(t_fine, cl_fine, idx_fine)
            
            line_cd_f.set_data(t_s_f, cd_s_f)
            line_cl_f.set_data(t_s_f, cl_s_f)
        
        head_cd.set_data([t_curr], [cd_curr])
        head_cl.set_data([t_curr], [cl_curr])
        text_cd.set_position((t_curr, cd_curr))
        text_cd.set_text(f'  {cd_curr:.3f}') 
        text_cl.set_position((t_curr, cl_curr))
        text_cl.set_text(f'  {cl_curr:.3f}')
        
        return line_cd_c, line_cd_f, line_cl_c, line_cl_f, head_cd, head_cl, text_cd, text_cl

    total_frames = len(t_all)
    skip = max(1, total_frames // 250)
    frames_to_render = range(0, total_frames, skip)

    ani = animation.FuncAnimation(
        fig, update, frames=frames_to_render,
        init_func=init, blit=True, interval=40
    )

    out_gif = os.path.join(os.path.dirname(FINE_PVD), "mesh_transition_aerodynamics.gif")
    ani.save(out_gif, writer='pillow', fps=10)
    print(f"\n=== DONE! Transition GIF saved in: {out_gif} ===")
    
    plt.tight_layout()
    plt.show()