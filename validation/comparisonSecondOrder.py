# --- File: validation/comparisonSecondOrder.py ---
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

class ExactRiemannSolver:
    """
    Risolutore Analitico Esatto per il Problema di Riemann 1D (Toro, 2009).
    Calcola la soluzione esatta per il Tubo di Sod a un dato istante t.
    """
    def __init__(self, rho_L=1.0, p_L=1.0, u_L=0.0, rho_R=0.125, p_R=0.1, u_R=0.0, gamma=1.4):
        self.rhoL, self.pL, self.uL = rho_L, p_L, u_L
        self.rhoR, self.pR, self.uR = rho_R, p_R, u_R
        self.gamma = gamma
        self.cL = np.sqrt(gamma * p_L / rho_L)
        self.cR = np.sqrt(gamma * p_R / rho_R)
        
    def _fk(self, p, dk, pk, ck):
        """Funzione di pressione per gli stati sinistro/destro"""
        A = 2.0 / ((self.gamma + 1.0) * dk)
        B = (self.gamma - 1.0) / (self.gamma + 1.0) * pk
        if p > pk: # Urto
            f = (p - pk) * np.sqrt(A / (p + B))
            df = np.sqrt(A / (p + B)) * (1.0 - (p - pk) / (2.0 * (p + B)))
        else:      # Rarefazione
            pratio = p / pk
            f = (2.0 * ck / (self.gamma - 1.0)) * (pratio**((self.gamma - 1.0) / (2.0 * self.gamma)) - 1.0)
            df = (1.0 / (dk * ck)) * pratio**(-(self.gamma + 1.0) / (2.0 * self.gamma))
        return f, df

    def solve_star_region(self):
        """Trova p_star usando il metodo di Newton-Raphson"""
        p_star = 0.5 * (self.pL + self.pR) # Guess iniziale
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
        """Campiona la soluzione esatta al punto x e tempo t"""
        p_star, u_star = self.solve_star_region()
        
        # Calcolo densità stella
        if p_star > self.pL: # Urto a sinistra
            r = p_star / self.pL
            rho_starL = self.rhoL * (r + (self.gamma-1)/(self.gamma+1)) / (r * (self.gamma-1)/(self.gamma+1) + 1.0)
        else:                # Rarefazione a sinistra
            rho_starL = self.rhoL * (p_star / self.pL)**(1.0 / self.gamma)
            
        if p_star > self.pR: # Urto a destra
            r = p_star / self.pR
            rho_starR = self.rhoR * (r + (self.gamma-1)/(self.gamma+1)) / (r * (self.gamma-1)/(self.gamma+1) + 1.0)
        else:                # Rarefazione a destra
            rho_starR = self.rhoR * (p_star / self.pR)**(1.0 / self.gamma)
            
        c_starL = np.sqrt(self.gamma * p_star / rho_starL)
        c_starR = np.sqrt(self.gamma * p_star / rho_starR)
        
        S = (x - x0) / t if t > 0 else 0.0
        
        if S < u_star: # A sinistra della discontinuità di contatto
            if p_star > self.pL: # Urto
                S_L = self.uL - self.cL * np.sqrt((self.gamma+1)/(2*self.gamma) * p_star/self.pL + (self.gamma-1)/(2*self.gamma))
                if S < S_L: return self.rhoL, self.uL, self.pL
                else:       return rho_starL, u_star, p_star
            else:                # Onda di Rarefazione
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
        else: # A destra della discontinuità di contatto
            if p_star > self.pR: # Urto
                S_R = self.uR + self.cR * np.sqrt((self.gamma+1)/(2*self.gamma) * p_star/self.pR + (self.gamma-1)/(2*self.gamma))
                if S > S_R: return self.rhoR, self.uR, self.pR
                else:       return rho_starR, u_star, p_star
            else:                # Onda di Rarefazione
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

# --- LETTURA PVD ---
def read_latest_from_pvd(pvd_filepath):
    """
    Legge il file indice PVD per trovare l'ultimo step temporale,
    quindi fa il parsing del corrispondente file XML .vtr
    """
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
    nx = extent[1]
    ny = extent[3]
    
    def get_array(name, is_vector=False):
        node = vtr_root.find(f'.//DataArray[@Name="{name}"]')
        data = np.array(node.text.split(), dtype=float)
        if is_vector:
            data = data.reshape(ny, nx, 3)
        else:
            data = data.reshape(ny, nx)
        return data

    x_nodes_text = vtr_root.find('.//Coordinates/DataArray[@Name="x"]').text
    x_nodes = np.array(x_nodes_text.split(), dtype=float)
    x_cells = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    
    mid_y = ny // 2
    
    rho = get_array("Density")[mid_y, :]
    p = get_array("Pressure")[mid_y, :]
    u = get_array("Velocity", is_vector=True)[mid_y, :, 0]
    
    T = p / rho 

    return x_cells, time, rho, p, u, T

# --- ESECUZIONE DELLA VALIDAZIONE ---
if __name__ == "__main__":

    print("=== VALIDAZIONE SOD: ANALISI SCHEMI NUMERICI ===")

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(root_dir, "results")

    # --------------------------------------------------
    # PLOT 1: Godunov vs MUSCL "Base" (Mantiene il tuo codice originale)
    # --------------------------------------------------
    pvd_godunov = os.path.join(results_dir, "sod_shock_tube_1", "sod_shock_tube_1.pvd")
    pvd_muscl = os.path.join(results_dir, "sod_shock_tube_2", "sod_shock_tube_2.pvd")

    time_ref = 0.0

    if os.path.exists(pvd_godunov) and os.path.exists(pvd_muscl):
        print("\n[1/4] Lettura Godunov...")
        x_god, t_god, rho_god, p_god, u_god, T_god = read_latest_from_pvd(pvd_godunov)
        print(f" -> Tempo: {t_god:.5f} s")

        print("\n[2/4] Lettura MUSCL Base...")
        x_muscl, t_muscl, rho_muscl, p_muscl, u_muscl, T_muscl = read_latest_from_pvd(pvd_muscl)
        print(f" -> Tempo: {t_muscl:.5f} s")

        time_ref = min(t_god, t_muscl)
    else:
        print("\n[WARNING] Dataset base (sod_shock_tube_1 o 2) non trovati. Salto Plot 1.")

    # --------------------------------------------------
    # Soluzione analitica
    # --------------------------------------------------
    # Se i pvd base non c'erano, mettiamo un tempo di default per tracciare l'esatta
    if time_ref == 0.0: time_ref = 0.2 
    
    print("\n[3/4] Calcolo soluzione esatta (Toro)...")
    exact_solver = ExactRiemannSolver()
    x_exact = np.linspace(0.0, 1.0, 2000)
    rho_ex, u_ex, p_ex, T_ex = np.zeros_like(x_exact), np.zeros_like(x_exact), np.zeros_like(x_exact), np.zeros_like(x_exact)

    for i, x in enumerate(x_exact):
        rho, vel, press = exact_solver.sample(x, time_ref)
        rho_ex[i] = rho; u_ex[i] = vel; p_ex[i] = press; T_ex[i] = press / rho

    style_exact = {"color": "black", "linewidth": 2.0, "label": "Exact (Toro)"}

    # Grafico 1 se i dati base esistono
    if os.path.exists(pvd_godunov) and os.path.exists(pvd_muscl):
        fig, axs = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(f"Sod Shock Tube: 1st vs 2nd Order\nt = {time_ref:.5f} s", fontsize=15, fontweight="bold")

        style_god = {"marker": "o", "facecolors": "none", "edgecolors": "tab:red", "s": 18, "label": "Godunov 1st Order"}
        style_muscl = {"marker": "^", "facecolors": "none", "edgecolors": "tab:blue", "s": 18, "label": "MUSCL 2nd Order"}

        axs[0,0].plot(x_exact, rho_ex, **style_exact); axs[0,0].scatter(x_god, rho_god, **style_god); axs[0,0].scatter(x_muscl, rho_muscl, **style_muscl); axs[0,0].set_title("Density")
        axs[0,1].plot(x_exact, u_ex, **style_exact);   axs[0,1].scatter(x_god, u_god, **style_god);   axs[0,1].scatter(x_muscl, u_muscl, **style_muscl);   axs[0,1].set_title("Velocity")
        axs[1,0].plot(x_exact, p_ex, **style_exact);   axs[1,0].scatter(x_god, p_god, **style_god);   axs[1,0].scatter(x_muscl, p_muscl, **style_muscl);   axs[1,0].set_title("Pressure")
        axs[1,1].plot(x_exact, T_ex, **style_exact);   axs[1,1].scatter(x_god, T_god, **style_god);   axs[1,1].scatter(x_muscl, T_muscl, **style_muscl);   axs[1,1].set_title("Temperature")

        for ax in axs.flat:
            ax.set_xlim(0.0, 1.0); ax.grid(True, ls="--", alpha=0.5); ax.legend(fontsize=9)
        plt.tight_layout()

        pdf_path_1 = os.path.join(results_dir, "Sod_1st_vs_2nd_Order.pdf")
        fig.savefig(pdf_path_1, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Salvo Plot 1 in: {pdf_path_1}")

    # --------------------------------------------------
    # PLOT 2: Confronto tra Limitatori MUSCL
    # --------------------------------------------------
    print("\n[4/4] Lettura e plot confronto Limiters...")
    
    limiters_map = {
        "Minmod": "Mm",
        "Van Leer": "VL",
        "MC": "Mc",
        "Koren": "K"
    }

    limiter_results = {}
    
    for name, suffix in limiters_map.items():
        folder_name = f"sod_shock_tube_2_{suffix}"
        pvd_path = os.path.join(results_dir, folder_name, f"{folder_name}.pvd")
        
        if os.path.exists(pvd_path):
            print(f"  -> Lettura {name} da {folder_name}...")
            x_lim, t_lim, rho_lim, p_lim, u_lim, T_lim = read_latest_from_pvd(pvd_path)
            limiter_results[name] = {
                "x": x_lim, "rho": rho_lim, "u": u_lim, "p": p_lim, "T": T_lim
            }
        else:
            print(f"  -> [INFO] {folder_name}.pvd non trovato. Verrà ignorato dal plot 2.")

    if limiter_results:
        fig2, axs2 = plt.subplots(2, 2, figsize=(13, 8))
        fig2.suptitle(f"MUSCL Limiters Comparison (MC, Minmod, Koren, Van Leer)\nt = {time_ref:.5f} s", fontsize=15, fontweight="bold")

        # Disegna l'analitica su tutti i subplots
        axs2[0,0].plot(x_exact, rho_ex, **style_exact); axs2[0,0].set_title("Density")
        axs2[0,1].plot(x_exact, u_ex, **style_exact);   axs2[0,1].set_title("Velocity")
        axs2[1,0].plot(x_exact, p_ex, **style_exact);   axs2[1,0].set_title("Pressure")
        axs2[1,1].plot(x_exact, T_ex, **style_exact);   axs2[1,1].set_title("Temperature")

        # Stili per i vari limitatori
        markers = ["o", "s", "^", "D"]
        colors = ["tab:orange", "tab:green", "tab:purple", "tab:cyan"]

        for i, (name, data) in enumerate(limiter_results.items()):
            style_lim = {
                "marker": markers[i % len(markers)],
                "facecolors": "none",
                "edgecolors": colors[i % len(colors)],
                "s": 20,
                "label": f"{name}"
            }

            axs2[0,0].scatter(data["x"], data["rho"], **style_lim)
            axs2[0,1].scatter(data["x"], data["u"], **style_lim)
            axs2[1,0].scatter(data["x"], data["p"], **style_lim)
            axs2[1,1].scatter(data["x"], data["T"], **style_lim)

        for ax in axs2.flat:
            ax.set_xlim(0.0, 1.0)
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend(fontsize=9)

        plt.tight_layout()

        pdf_path_2 = os.path.join(results_dir, "Sod_Limiters_Comparison.pdf")
        fig2.savefig(pdf_path_2, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Salvo Plot 2 in: {pdf_path_2}")

    plt.show()