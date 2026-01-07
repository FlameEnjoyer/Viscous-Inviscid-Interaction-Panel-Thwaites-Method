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
    
    # Cl integration
    circulation = np.sum(panel.vt * ds)
    
    return {
        'x': x_final, 'delta_star': bl['delta_star'], 
        'theta': bl['theta'], 'Cf': bl['Cf'],
        'Cl': -2 * circulation, 'Cd': 2 * bl['theta'][-1]
    }

def run_iterative(case):
    solver = VIISolver(case['naca'], case['alpha'], case['n_panels'], max_iter=200, tol=1e-4, relax=0.1)
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

def run_alpha_sweep(naca, re, alphas):
    results_ow = {'alpha': [], 'Cl': [], 'Cd': [], 'L/D': []}
    results_it = {'alpha': [], 'Cl': [], 'Cd': [], 'L/D': []}
    
    print(f"\nStarting Alpha Sweep: {alphas[0]} to {alphas[-1]} degrees...")
    print(f"{'Alpha':<8} {'Cl(It)':<10} {'Cd(It)':<10} {'Conv':<8}")
    print("-" * 40)

    base = setup() # base config
    
    for alpha in alphas:
        case = {
            'naca': naca, 'alpha': alpha, 'n_panels': 100,
            'c': base['c'], 'U_inf': base['U_inf'], 'rho': base['rho'],
            'nu': base['U_inf'] * base['c'] / re, 're': re
        }
        
        # One-way
        try:
            ow = run_oneway(case)
            results_ow['alpha'].append(alpha)
            results_ow['Cl'].append(ow['Cl'])
            results_ow['Cd'].append(ow['Cd'])
            results_ow['L/D'].append(ow['Cl'] / (ow['Cd'] + 1e-9))
        except:
            pass # Skip failed cases
            
        # Iterative
        try:
            solver = VIISolver(case['naca'], case['alpha'], case['n_panels'], max_iter=200, tol=5e-4, relax=0.1)
            it = solver.solve(case['nu'], case['rho'], case['U_inf'])
            
            if it['converged']:
                results_it['alpha'].append(alpha)
                results_it['Cl'].append(it['Cl'])
                results_it['Cd'].append(it['Cd'])
                results_it['L/D'].append(it['Cl'] / (it['Cd'] + 1e-9))
                print(f"{alpha:<8.1f} {it['Cl']:<10.4f} {it['Cd']:<10.5f} {'Yes':<8}")
            else:
                last_res = it['history']['delta_star_residual'][-1] if it['history']['delta_star_residual'] else -1
                print(f"{alpha:<8.1f} {'---':<10} {'---':<10} No ({last_res:.2e})")
        except Exception as e:
             print(f"{alpha:<8.1f} Error: {e}")

    return results_ow, results_it

def plot_polars(ow, it, naca, re):
    os.makedirs('results', exist_ok=True)
    
    # Filter for plotting (convert to numpy)
    a_ow = np.array(ow['alpha'])
    cl_ow = np.array(ow['Cl'])
    cd_ow = np.array(ow['Cd'])
    ld_ow = np.array(ow['L/D'])
    
    a_it = np.array(it['alpha'])
    cl_it = np.array(it['Cl'])
    cd_it = np.array(it['Cd'])
    ld_it = np.array(it['L/D'])

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Aerodynamic Polars: NACA {naca} Re={re:.0e}")

    # 1. Cl vs Cd
    axs[0,0].plot(cd_ow, cl_ow, 'b--', label='One-Way', alpha=0.6)
    axs[0,0].plot(cd_it, cl_it, 'r.-', label='Iterative')
    axs[0,0].set_xlabel('Cd')
    axs[0,0].set_ylabel('Cl')
    axs[0,0].set_title('Drag Polar (Cl vs Cd)')
    axs[0,0].grid(True, alpha=0.3)
    axs[0,0].legend()

    # 2. Cl vs Alpha
    axs[0,1].plot(a_ow, cl_ow, 'b--', alpha=0.6)
    axs[0,1].plot(a_it, cl_it, 'r.-')
    axs[0,1].set_xlabel('Alpha (deg)')
    axs[0,1].set_ylabel('Cl')
    axs[0,1].set_title('Lift Curve (Cl vs Alpha)')
    axs[0,1].grid(True, alpha=0.3)

    # 3. L/D vs Alpha
    axs[1,0].plot(a_ow, ld_ow, 'b--', alpha=0.6)
    axs[1,0].plot(a_it, ld_it, 'r.-')
    axs[1,0].set_xlabel('Alpha (deg)')
    axs[1,0].set_ylabel('Cl / Cd')
    axs[1,0].set_title('Efficiency (L/D vs Alpha)')
    axs[1,0].grid(True, alpha=0.3)

    # 4. Cd vs Alpha
    axs[1,1].plot(a_ow, cd_ow, 'b--', alpha=0.6)
    axs[1,1].plot(a_it, cd_it, 'r.-')
    axs[1,1].set_xlabel('Alpha (deg)')
    axs[1,1].set_ylabel('Cd')
    axs[1,1].set_title('Drag Curve (Cd vs Alpha)')
    axs[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/polars.png')
    print("\nPolars saved to results/polars.png")

def main():
    # Sweep Configuration
    naca = '0009'
    re = 500000  # Back to 500,000
    alphas = np.linspace(-15, 15, 31) # -15 to 15 step 1
    
    try:
        ow, it = run_alpha_sweep(naca, re, alphas)
        plot_polars(ow, it, naca, re)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
