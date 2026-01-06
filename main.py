import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from src.panel_solver import PanelSolver
from src.thwaites_solver import ThwaitesSolver
from src.vii_solver import VIISolver, plot_convergence

def setup():
    c = 1.0
    U_inf = 10.0
    Re = 500000
    return {
        'naca': '0009', 'alpha': 0.0, 'n_panels': 100,
        'c': c, 'U_inf': U_inf, 'rho': 1.225,
        'nu': U_inf * c / Re, 're': Re
    }

def run_oneway(case):
    panel = PanelSolver(case['naca'], case['alpha'], case['n_panels'])
    panel.solve()
    res = panel.get_surface_properties()
    
    # Process Upper Surface
    x, y, Ue = res['x'], res['y'], res['Ue']
    split = len(x) // 2
    xs, ys, us = x[split:], y[split:], Ue[split:]
    
    ds = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    s = np.concatenate(([0], np.cumsum(ds)))
    s_final = np.concatenate(([0.0], s + np.sqrt(xs[0]**2 + ys[0]**2)))
    ue_final = np.concatenate(([0.0], np.abs(us)))
    x_final = np.concatenate(([0.0], xs))
    
    if len(ue_final) > 11:
        ue_final = savgol_filter(ue_final, 9, 3)
        ue_final[0] = 0.0
    
    thwaites = ThwaitesSolver()
    bl = thwaites.calculate_properties(s_final, ue_final*case['U_inf'], case['nu'], case['rho'])
    
    return {
        'x': x_final, 'delta_star': bl['delta_star'], 
        'theta': bl['theta'], 'Cf': bl['Cf'],
        'Cl': 2 * panel.gamma, 'Cd': 2 * bl['theta'][-1]
    }

def run_iterative(case):
    solver = VIISolver(case['naca'], case['alpha'], case['n_panels'], max_iter=60, tol=1e-4, relax=0.1)
    res = solver.solve(case['nu'], case['rho'], case['U_inf'])
    return res

def plot_compare(ow, it, case):
    os.makedirs('results', exist_ok=True)
    it_up = it['upper']
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"NACA {case['naca']} Re={case['re']:.0e} alpha={case['alpha']}")
    
    # Delta Star
    axs[0,0].plot(ow['x'], ow['delta_star']*1000, 'b-', label='One-Way')
    axs[0,0].plot(it_up['x'], it_up['delta_star']*1000, 'r--', label='Iterative')
    axs[0,0].set_ylabel(r'$\delta^*$ [mm]')
    axs[0,0].legend()
    axs[0,0].grid(True, alpha=0.3)
    
    # Theta
    axs[0,1].plot(ow['x'], ow['theta']*1000, 'b-')
    axs[0,1].plot(it_up['x'], it_up['theta']*1000, 'r--')
    axs[0,1].set_ylabel(r'$\theta$ [mm]')
    axs[0,1].grid(True, alpha=0.3)
    
    # Cf
    axs[1,0].plot(ow['x'], ow['Cf'], 'b-')
    axs[1,0].plot(it_up['x'], it_up['Cf'], 'r--')
    axs[1,0].set_ylabel(r'$C_f$')
    axs[1,0].set_ylim(0, 0.01)
    axs[1,0].grid(True, alpha=0.3)
    
    # Shape Factor
    axs[1,1].plot(it_up['x'], it_up['H'], 'r--', label='Iterative H')
    axs[1,1].set_ylabel('H')
    axs[1,1].legend()
    axs[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/comparison.png')

def main():
    case = setup()
    print(f"Running NACA {case['naca']} Re={case['re']}...")
    
    ow = run_oneway(case)
    it = run_iterative(case)
    
    print("\nResults Summary:")
    print(f"{'Metric':<10} {'One-Way':<15} {'Iterative':<15}")
    print("-" * 40)
    print(f"{'Cl':<10} {ow['Cl']:<15.5f} {it['Cl']:<15.5f}")
    print(f"{'Cd':<10} {ow['Cd']:<15.5f} {it['Cd']:<15.5f}")
    print(f"{'d*_TE':<10} {ow['delta_star'][-1]*1000:<15.4f} {it['upper']['delta_star'][-1]*1000:<15.4f}")
    
    plot_compare(ow, it, case)
    plot_convergence(it['history'], 'results/convergence.png')
    print("\nPlots saved to results/")

if __name__ == "__main__":
    main()
