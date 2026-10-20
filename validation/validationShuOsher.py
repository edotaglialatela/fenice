import os
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d

# --- FUNZIONE DI LETTURA (Riciclata e robusta) ---
def read_latest_from_pvd(pvd_filepath):
    pvd_tree = ET.parse(pvd_filepath)
    pvd_root = pvd_tree.getroot()
    
    datasets = pvd_root.findall('.//DataSet')
    last_dataset = datasets[-1]
    
    time = float(last_dataset.attrib['timestep'])
    vtr_filename = last_dataset.attrib['file']
    
    vtr_filepath = os.path.join(os.path.dirname(pvd_filepath), vtr_filename)
    
    vtr_tree = ET.parse(vtr_filepath)
    vtr_root = vtr_tree.getroot()
    
    piece = vtr_root.find('.//Piece')
    extent = list(map(int, piece.attrib['Extent'].split()))
    nx, ny = extent[1], extent[3]
    
    def get_array(name, is_vector=False):
        node = vtr_root.find(f'.//DataArray[@Name="{name}"]')
        data = np.array(node.text.split(), dtype=float)
        return data.reshape(ny, nx, 3) if is_vector else data.reshape(ny, nx)

    x_nodes = np.array(vtr_root.find('.//Coordinates/DataArray[@Name="x"]').text.split(), dtype=float)
    x_cells = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    dx = x_nodes[1] - x_nodes[0]
    
    mid_y = ny // 2
    rho = get_array("Density")[mid_y, :]
    
    return x_cells, dx, time, rho

# --- MAIN EXECUTABLE ---
if __name__ == "__main__":
    print("=== VALIDAZIONE: SHU-OSHER PROBLEM (SELF-CONVERGENCE) ===")
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(root_dir, "results")
    
    paths = {
        "REF": os.path.join(results_dir, "shu-osher_test_ref", "shu-osher_test_ref.pvd"),
        "Test 1": os.path.join(results_dir, "shu-osher_test_1", "shu-osher_test.pvd"),
        "Test 2": os.path.join(results_dir, "shu-osher_test_2", "shu-osher_test.pvd")
    }
    
    data = {}
    
    # 1. Lettura Dati
    for name, pvd in paths.items():
        if not os.path.exists(pvd):
            print(f"[ERRORE] File PVD non trovato per {name}: {pvd}")
            exit(1)
        x, dx, t, rho = read_latest_from_pvd(pvd)
        data[name] = {'x': x, 'dx': dx, 't': t, 'rho': rho}
        print(f"[{name}] Letto t={t:.3f}s | nx={len(x)} | dx={dx:.5f}")

    # 2. Interpolazione e Calcolo Errore L1
    print("\nCalcolo Errore L1 tramite proiezione sulla Reference Solution...")
    
    # Creiamo un interpolatore cubico continuo basato sulla mesh di riferimento (10000 celle)
    f_ref = interp1d(data["REF"]['x'], data["REF"]['rho'], kind='cubic', bounds_error=False, fill_value="extrapolate")
    
    errors = []
    dxs = []
    
    for name in ["Test 1", "Test 2"]:
        x_test = data[name]['x']
        rho_test = data[name]['rho']
        
        # Proiettiamo la reference sui centri cella del test corrente
        rho_ref_proj = f_ref(x_test)
        
        # Errore norma L1 (Media degli scarti assoluti)
        err_L1 = np.mean(np.abs(rho_test - rho_ref_proj))
        
        data[name]['err_L1'] = err_L1
        errors.append(err_L1)
        dxs.append(data[name]['dx'])
        
        print(f"[{name}] Errore L1: {err_L1:.4e}")

    # Calcolo Ordine di Convergenza (p)
    # p = ln(E1 / E2) / ln(dx1 / dx2)
    p_order = np.log(errors[0] / errors[1]) / np.log(dxs[0] / dxs[1])
    print(f"\n>>> ORDINE DI CONVERGENZA SPERIMENTALE (p): {p_order:.3f} <<<")

    # 3. PLOTTING
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Shu-Osher Shock-Entropy Interaction ($t = {data['REF']['t']:.2f}$ s)\nExperimental Order of Convergence: $p = {p_order:.2f}$", fontsize=15, fontweight='bold')

    # --- Plot Sinistro: Profili di Densità ---
    ax_rho = axs[0]
    
    # Reference (Linea continua nera)
    ax_rho.plot(data["REF"]['x'], data["REF"]['rho'], color='k', linestyle='-', linewidth=2.0, label='Reference (High-Res)', zorder=1)
    
    # Test 1 (Azzurro scuro, trasparenza maggiore)
    ax_rho.scatter(data["Test 1"]['x'], data["Test 1"]['rho'], 
                   marker='o', facecolors='none', edgecolors='darkblue', alpha=0.4, s=20, 
                   label=f'Test 1 (dx={data["Test 1"]["dx"]:.4f})', zorder=2)
                   
    # Test 2 (Azzurro scuro, quasi opaco)
    ax_rho.scatter(data["Test 2"]['x'], data["Test 2"]['rho'], 
                   marker='s', facecolors='none', edgecolors='darkblue', alpha=0.9, s=20, 
                   label=f'Test 2 (dx={data["Test 2"]["dx"]:.4f})', zorder=3)

    ax_rho.set_title(r'Density Profile ($\rho$)')
    ax_rho.set_xlabel('x [m]')
    ax_rho.set_ylabel('Density')
    ax_rho.set_xlim(0, 10)
    ax_rho.grid(True, linestyle='--', alpha=0.5)
    ax_rho.legend(fontsize=10)
    
    # Inset (Zoom sulle onde di entropia dietro l'urto)
    axins = ax_rho.inset_axes([0.1, 0.45, 0.4, 0.4]) # [x0, y0, width, height]
    axins.plot(data["REF"]['x'], data["REF"]['rho'], 'k-', lw=1.5)
    axins.scatter(data["Test 1"]['x'], data["Test 1"]['rho'], marker='o', facecolors='none', edgecolors='darkblue', alpha=0.4, s=15)
    axins.scatter(data["Test 2"]['x'], data["Test 2"]['rho'], marker='s', facecolors='none', edgecolors='darkblue', alpha=0.9, s=15)
    axins.set_xlim(4.5, 7.5)  # Area critica di oscillazione
    axins.set_ylim(2.5, 5.0)
    axins.grid(True, ls='--', alpha=0.5)
    ax_rho.indicate_inset_zoom(axins, edgecolor="gray")

    # --- Plot Destro: Ordine di Convergenza ---
    ax_conv = axs[1]
    
    # Plot dell'errore (log-log)
    ax_conv.loglog(dxs, errors, marker='o', color='tab:red', linestyle='-', linewidth=2, markersize=8, label='L1 Error')
    
    # Creazione del triangolo teorico di pendenza p = 1 e p = 2 per paragone visivo
    dx_ref = np.array([min(dxs), max(dxs)])
    err_ref_1 = errors[0] * (dx_ref / dxs[0])**1.0  # Retta p=1 (1st Order)
    err_ref_2 = errors[0] * (dx_ref / dxs[0])**2.0  # Retta p=2 (2nd Order)
    
    ax_conv.loglog(dx_ref, err_ref_1, 'k--', alpha=0.5, label=r'$\mathcal{O}(\Delta x)$ (1st Order)')
    ax_conv.loglog(dx_ref, err_ref_2, 'k-.', alpha=0.5, label=r'$\mathcal{O}(\Delta x^2)$ (2nd Order)')
    
    ax_conv.set_title('Grid Convergence (Log-Log)')
    ax_conv.set_xlabel('Grid spacing $\Delta x$ [m]')
    ax_conv.set_ylabel('$L_1$ Error Norm')
    ax_conv.grid(True, which="both", linestyle='--', alpha=0.5)
    ax_conv.legend(fontsize=10)

    plt.tight_layout()
    
    pdf_path = os.path.join(results_dir, "validation_ShuOsher.pdf")
    plt.savefig(pdf_path, format="pdf", dpi=300)
    print(f"\nGrafico salvato in: {pdf_path}")
    
    plt.show()