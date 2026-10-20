# --- File: validation/validationSod.py ---
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

# --- ESECUZIONE DELLA VALIDAZIONE ---
def read_latest_from_pvd(pvd_filepath):
    """
    Legge il file indice PVD per trovare l'ultimo step temporale,
    quindi fa il parsing del corrispondente file XML .vtr
    """
    # 1. Parsing del file .pvd
    pvd_tree = ET.parse(pvd_filepath)
    pvd_root = pvd_tree.getroot()
    
    # Trova tutti i dataset e prendi l'ultimo
    datasets = pvd_root.findall('.//DataSet')
    last_dataset = datasets[-1]
    
    time = float(last_dataset.attrib['timestep'])
    vtr_filename = last_dataset.attrib['file']
    
    # 2. Parsing del file .vtr associato
    vtr_filepath = os.path.join(os.path.dirname(pvd_filepath), vtr_filename)
    print(f"Leggendo i dati numerici da: {vtr_filename} (Tempo: {time:.5f} s)")
    
    vtr_tree = ET.parse(vtr_filepath)
    vtr_root = vtr_tree.getroot()
    
    # Estrazione dimensioni dalla griglia (Extent="0 nx 0 ny 0 0")
    piece = vtr_root.find('.//Piece')
    extent = list(map(int, piece.attrib['Extent'].split()))
    nx = extent[1]
    ny = extent[3]
    
    # Funzione di supporto per estrarre gli array dal testo XML
    def get_array(name, is_vector=False):
        node = vtr_root.find(f'.//DataArray[@Name="{name}"]')
        # Splitta la stringa e converte tutto in float (gestisce spazi e a capo automatici)
        data = np.array(node.text.split(), dtype=float)
        
        if is_vector:
            # I vettori hanno 3 componenti
            data = data.reshape(ny, nx, 3)
        else:
            data = data.reshape(ny, nx)
        return data

    # Coordinate X
    x_nodes_text = vtr_root.find('.//Coordinates/DataArray[@Name="x"]').text
    x_nodes = np.array(x_nodes_text.split(), dtype=float)
    x_cells = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    
    # Estrazione campi e "slicing" lungo la linea di mezzeria orizzontale
    mid_y = ny // 2
    
    rho = get_array("Density")[mid_y, :]
    p = get_array("Pressure")[mid_y, :]
    u = get_array("Velocity", is_vector=True)[mid_y, :, 0] # Prende solo la componente X
    
    T = p / rho # Assumendo gas ideale

    return x_cells, time, rho, p, u, T

# --- ESECUZIONE DELLA VALIDAZIONE ---
if __name__ == "__main__":
    print("=== VALIDAZIONE: SOD SHOCK TUBE ===")
    
    # Cerca il file .pvd del Sod Shock Tube
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(root_dir, "results", "sod_shock_tube")
    pvd_filepath = os.path.join(results_dir, "sod_shock_tube.pvd")
    
    if not os.path.exists(pvd_filepath):
        print(f"[ERRORE] File PVD non trovato in: {pvd_filepath}")
        exit()
        
    # 1. Dati Numerici estratti automaticamente dall'XML
    x_num, time, rho_num, p_num, u_num, T_num = read_latest_from_pvd(pvd_filepath)
    
    # 2. Dati Esatti (Soluzione Analitica)
    print("Calcolo soluzione analitica esatta (Risolutore di Toro)...")
    exact_solver = ExactRiemannSolver()
    
    # Griglia molto fine per una curva liscia
    x_exact = np.linspace(0.0, 1.0, 1000)
    rho_ex, u_ex, p_ex, T_ex = [], [], [], []
    
    for x in x_exact:
        r, v, P = exact_solver.sample(x, time)
        rho_ex.append(r)
        u_ex.append(v)
        p_ex.append(P)
        T_ex.append(P / r)
        
    # 3. PLOTTING
    style_num = {
        'marker': 'o', 
        'facecolors': 'none', 
        'edgecolors': 'darkblue', 
        's': 20,  # <-- 's' sostituisce 'markersize', ed è senza 'linestyle'
        'label': 'Numerical FVM (1st Order)'
    }    
    style_ex = {'color': 'k', 'linestyle': '-', 'linewidth': 1.5, 'label': 'Exact'}

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Validazione Sod Shock Tube - $t = {time:.5f}$ s", fontsize=14, fontweight='bold')

    axs[0, 0].plot(x_exact, rho_ex, **style_ex); axs[0, 0].scatter(x_num, rho_num, **style_num)
    axs[0, 0].set_title('Density ($\\rho$)'); axs[0, 0].grid(True, linestyle='--', alpha=0.5)

    axs[0, 1].plot(x_exact, u_ex, **style_ex); axs[0, 1].scatter(x_num, u_num, **style_num)
    axs[0, 1].set_title('Velocity ($u$)'); axs[0, 1].grid(True, linestyle='--', alpha=0.5)

    axs[1, 0].plot(x_exact, p_ex, **style_ex); axs[1, 0].scatter(x_num, p_num, **style_num)
    axs[1, 0].set_title('Pressure ($p$)'); axs[1, 0].grid(True, linestyle='--', alpha=0.5)

    axs[1, 1].plot(x_exact, T_ex, **style_ex); axs[1, 1].scatter(x_num, T_num, **style_num)
    axs[1, 1].set_title('Temperature ($T = p/\\rho$)'); axs[1, 1].grid(True, linestyle='--', alpha=0.5)

    for ax in axs.flat:
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()