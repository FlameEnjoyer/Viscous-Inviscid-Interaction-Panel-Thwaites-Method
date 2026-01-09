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

    x, y, Ue = res['x'], res['y'], res['Ue']
    split = len(x) // 2

    # Process UPPER surface - smooth then prepend stagnation point
    xs_up, ys_up, us_up = x[split:], y[split:], Ue[split:]
    ds_up = np.sqrt(np.diff(xs_up)**2 + np.diff(ys_up)**2)
    s_up = np.concatenate(([0], np.cumsum(ds_up)))
    ue_final_up = np.abs(us_up)

    # Triple-pass aggressive smoothing to eliminate LE artifacts
    if len(ue_final_up) > 21:
        ue_final_up = savgol_filter(ue_final_up, 21, 3)
        ue_final_up = savgol_filter(ue_final_up, 17, 3)
        ue_final_up = savgol_filter(ue_final_up, 13, 3)
    elif len(ue_final_up) > 15:
        ue_final_up = savgol_filter(ue_final_up, 15, 3)
        ue_final_up = savgol_filter(ue_final_up, 11, 3)
    elif len(ue_final_up) > 11:
        ue_final_up = savgol_filter(ue_final_up, 11, 3)

    # Prepend stagnation point for Hiemenz solution
    dist_to_le_up = np.sqrt(xs_up[0]**2 + ys_up[0]**2)
    s_final_up = np.concatenate(([0.0], s_up + dist_to_le_up))
    ue_final_up = np.concatenate(([0.0], ue_final_up))
    x_final_up = np.concatenate(([0.0], xs_up))

    # Process LOWER surface - smooth then prepend stagnation point
    xs_low, ys_low, us_low = x[:split][::-1], y[:split][::-1], Ue[:split][::-1]
    ds_low = np.sqrt(np.diff(xs_low)**2 + np.diff(ys_low)**2)
    s_low = np.concatenate(([0], np.cumsum(ds_low)))
    ue_final_low = np.abs(us_low)

    # Triple-pass aggressive smoothing to eliminate LE artifacts
    if len(ue_final_low) > 21:
        ue_final_low = savgol_filter(ue_final_low, 21, 3)
        ue_final_low = savgol_filter(ue_final_low, 17, 3)
        ue_final_low = savgol_filter(ue_final_low, 13, 3)
    elif len(ue_final_low) > 15:
        ue_final_low = savgol_filter(ue_final_low, 15, 3)
        ue_final_low = savgol_filter(ue_final_low, 11, 3)
    elif len(ue_final_low) > 11:
        ue_final_low = savgol_filter(ue_final_low, 11, 3)

    # Prepend stagnation point for Hiemenz solution
    dist_to_le_low = np.sqrt(xs_low[0]**2 + ys_low[0]**2)
    s_final_low = np.concatenate(([0.0], s_low + dist_to_le_low))
    ue_final_low = np.concatenate(([0.0], ue_final_low))
    x_final_low = np.concatenate(([0.0], xs_low))

    # Calculate boundary layer for BOTH surfaces with min_sep_x threshold
    thwaites = ThwaitesSolver(min_sep_x=0.05)  # 5% chord minimum for separation
    bl_up = thwaites.calculate_properties(s_final_up, ue_final_up*case['U_inf'], case['nu'], case['rho'], x=x_final_up)
    bl_low = thwaites.calculate_properties(s_final_low, ue_final_low*case['U_inf'], case['nu'], case['rho'], x=x_final_low)

    # Circulation for Cl
    circulation = np.sum(panel.vt * ds_up)

    # FIXED: Cd now includes BOTH upper and lower trailing edge theta
    Cd = 2 * (bl_up['theta'][-1] + bl_low['theta'][-1])

    return {
        'x': x_final_up, 'delta_star': bl_up['delta_star'],
        'theta': bl_up['theta'], 'Cf': bl_up['Cf'],
        'Cl': -2 * circulation, 'Cd': Cd
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

def plot_boundary_layer_properties(result, case):
    """
    Plot boundary layer properties along the chord
    Similar to XFOIL/PyBL comparison plots
    """
    os.makedirs('results', exist_ok=True)

    # Extract upper surface boundary layer properties
    upper = result['upper']
    x = upper['x']
    theta = upper['theta']
    delta_star = upper['delta_star']
    Cf = upper['Cf']
    H = upper['H']

    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Boundary Layer Properties: NACA {case['naca']}, alpha={case['alpha']}deg, Re={case['re']:.0e}")

    # Plot 1: Momentum Thickness (theta)
    axs[0, 0].plot(x, theta, 'k-', linewidth=1.5)
    axs[0, 0].set_xlabel('x (m)')
    axs[0, 0].set_ylabel(r'$\theta$ (m)')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].set_xlim([0, 1])

    # Plot 2: Displacement Thickness (delta*)
    axs[0, 1].plot(x, delta_star, 'k-', linewidth=1.5)
    axs[0, 1].set_xlabel('x (m)')
    axs[0, 1].set_ylabel(r'$\delta^*$ (m)')
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].set_xlim([0, 1])

    # Plot 3: Skin Friction Coefficient (Cf)
    axs[1, 0].plot(x, Cf, 'k-', linewidth=1.5)
    axs[1, 0].set_xlabel('x (m)')
    axs[1, 0].set_ylabel(r'$c_f$')
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].set_xlim([0, 1])
    axs[1, 0].set_ylim([0, None])

    # Plot 4: Shape Factor (H)
    axs[1, 1].plot(x, H, 'k-', linewidth=1.5)
    axs[1, 1].set_xlabel('x (m)')
    axs[1, 1].set_ylabel('H')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].set_xlim([0, 1])
    axs[1, 1].set_ylim([2.0, 4.0])

    # Add separation point annotation if exists
    lam = upper['lambda']
    sep_indices = np.where(lam <= -0.09)[0]
    if len(sep_indices) > 0:
        sep_x = x[sep_indices[0]]
        for ax in axs.flat:
            ax.axvline(sep_x, color='r', linestyle='--', linewidth=1, alpha=0.5, label=f'Sep. x={sep_x:.3f}')
            ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    plt.savefig(f"results/bl_properties_alpha{case['alpha']:.0f}_re{case['re']:.0e}.png", dpi=150)
    print(f"  BL plot: bl_properties_alpha{case['alpha']:.0f}_re{case['re']:.0e}.png")

def plot_multiple_angles(results_dict, naca, re):
    """
    Plot BL properties for multiple angles on a single figure
    results_dict: {'angle': result}
    """
    os.makedirs('results', exist_ok=True)

    angles = sorted(results_dict.keys())
    n_angles = len(angles)

    fig, axs = plt.subplots(n_angles, 4, figsize=(14, 3*n_angles))
    if n_angles == 1:
        axs = axs.reshape(1, -1)

    fig.suptitle(f"Boundary Layer Properties: NACA {naca} at Multiple Angles, Re={re:.0e}", fontsize=14)

    for row, alpha in enumerate(angles):
        result = results_dict[alpha]
        upper = result['upper']
        x = upper['x']
        theta = upper['theta']
        delta_star = upper['delta_star']
        Cf = upper['Cf']
        H = upper['H']
        lam = upper['lambda']

        # Theta
        axs[row, 0].plot(x, theta, 'k-', linewidth=1.5)
        axs[row, 0].set_ylabel(f"alpha={alpha}deg\n" + r"$\theta$ (m)", fontsize=10)
        axs[row, 0].grid(True, alpha=0.3)
        axs[row, 0].set_xlim([0, 1])

        # Delta*
        axs[row, 1].plot(x, delta_star, 'k-', linewidth=1.5)
        axs[row, 1].set_ylabel(f"alpha={alpha}deg\n" + r"$\delta^*$ (m)", fontsize=10)
        axs[row, 1].grid(True, alpha=0.3)
        axs[row, 1].set_xlim([0, 1])

        # Cf
        axs[row, 2].plot(x, Cf, 'k-', linewidth=1.5)
        axs[row, 2].set_ylabel(f"alpha={alpha}deg\n" + r"$c_f$", fontsize=10)
        axs[row, 2].grid(True, alpha=0.3)
        axs[row, 2].set_xlim([0, 1])
        axs[row, 2].set_ylim([0, None])

        # H
        axs[row, 3].plot(x, H, 'k-', linewidth=1.5)
        axs[row, 3].set_ylabel(f"alpha={alpha}deg\n" + "H", fontsize=10)
        axs[row, 3].grid(True, alpha=0.3)
        axs[row, 3].set_xlim([0, 1])
        axs[row, 3].set_ylim([2.0, 4.0])

        # Add separation line
        sep_indices = np.where(lam <= -0.09)[0]
        if len(sep_indices) > 0:
            sep_x = x[sep_indices[0]]
            for col in range(4):
                axs[row, col].axvline(sep_x, color='r', linestyle='--', linewidth=1, alpha=0.5)

        # x labels only on bottom
        if row == n_angles - 1:
            for col in range(4):
                axs[row, col].set_xlabel('x (m)')

    plt.tight_layout()
    plt.savefig(f"results/bl_multi_angle_re{re:.0e}.png", dpi=150)
    print(f"  Multi-angle plot: bl_multi_angle_re{re:.0e}.png")

def plot_reynolds_comparison(results_dict, naca, alpha):
    """
    Plot BL properties for multiple Reynolds numbers
    results_dict: {'reynolds': result}
    """
    os.makedirs('results', exist_ok=True)

    re_values = sorted(results_dict.keys())
    n_re = len(re_values)

    fig, axs = plt.subplots(n_re, 4, figsize=(14, 3*n_re))
    if n_re == 1:
        axs = axs.reshape(1, -1)

    fig.suptitle(f"Boundary Layer Properties: NACA {naca} at alpha={alpha}deg, Multiple Reynolds", fontsize=14)

    for row, re in enumerate(re_values):
        result = results_dict[re]
        upper = result['upper']
        x = upper['x']
        theta = upper['theta']
        delta_star = upper['delta_star']
        Cf = upper['Cf']
        H = upper['H']
        lam = upper['lambda']

        # Theta
        axs[row, 0].plot(x, theta, 'k-', linewidth=1.5)
        axs[row, 0].set_ylabel(f"Re={re:.0e}\n" + r"$\theta$ (m)", fontsize=10)
        axs[row, 0].grid(True, alpha=0.3)
        axs[row, 0].set_xlim([0, 1])

        # Delta*
        axs[row, 1].plot(x, delta_star, 'k-', linewidth=1.5)
        axs[row, 1].set_ylabel(f"Re={re:.0e}\n" + r"$\delta^*$ (m)", fontsize=10)
        axs[row, 1].grid(True, alpha=0.3)
        axs[row, 1].set_xlim([0, 1])

        # Cf
        axs[row, 2].plot(x, Cf, 'k-', linewidth=1.5)
        axs[row, 2].set_ylabel(f"Re={re:.0e}\n" + r"$c_f$", fontsize=10)
        axs[row, 2].grid(True, alpha=0.3)
        axs[row, 2].set_xlim([0, 1])
        axs[row, 2].set_ylim([0, None])

        # H
        axs[row, 3].plot(x, H, 'k-', linewidth=1.5)
        axs[row, 3].set_ylabel(f"Re={re:.0e}\n" + "H", fontsize=10)
        axs[row, 3].grid(True, alpha=0.3)
        axs[row, 3].set_xlim([0, 1])
        axs[row, 3].set_ylim([2.0, 4.0])

        # Add separation line
        sep_indices = np.where(lam <= -0.09)[0]
        if len(sep_indices) > 0:
            sep_x = x[sep_indices[0]]
            for col in range(4):
                axs[row, col].axvline(sep_x, color='r', linestyle='--', linewidth=1, alpha=0.5)

        # x labels only on bottom
        if row == n_re - 1:
            for col in range(4):
                axs[row, col].set_xlabel('x (m)')

    plt.tight_layout()
    plt.savefig(f"results/bl_reynolds_comparison_alpha{alpha:.0f}.png", dpi=150)
    print(f"  Reynolds comparison plot: bl_reynolds_comparison_alpha{alpha:.0f}.png")

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
    # Configuration
    naca = '0009'
    base_re = 500000
    base = setup()

    alphas = np.linspace(-15, 15, 31)

    try:
        # Run alpha sweep at base Re
        print("\n" + "="*80)
        print("AERODYNAMIC SWEEP: NACA 0009, Re=500000")
        print("="*80)
        ow, it = run_alpha_sweep(naca, base_re, alphas)
        plot_polars(ow, it, naca, base_re)

        # =====================================================================
        # PART 1: BOUNDARY LAYER PROPERTIES AT DIFFERENT ANGLES (Re=500k)
        # =====================================================================
        print("\n" + "="*80)
        print("PART 1: Boundary Layer Properties at Multiple Angles (Re=500,000)")
        print("="*80)

        test_angles = [0.0, 5.0, 10.0]
        angle_results = {}

        for alpha in test_angles:
            print(f"\nProcessing alpha = {alpha}deg...")
            case = {
                'naca': naca, 'alpha': alpha, 'n_panels': 100,
                'c': base['c'], 'U_inf': base['U_inf'], 'rho': base['rho'],
                'nu': base['U_inf'] * base['c'] / base_re, 're': base_re
            }

            solver = VIISolver(case['naca'], case['alpha'], case['n_panels'],
                              max_iter=200, tol=5e-4, relax=0.1)
            result = solver.solve(case['nu'], case['rho'], case['U_inf'])

            if result['converged']:
                print(f"  Converged in {result['iterations']} iterations")
                print(f"  Cl = {result['Cl']:.6f}, Cd = {result['Cd']:.6f}")
                plot_boundary_layer_properties(result, case)
                angle_results[alpha] = result
            else:
                print(f"  Warning: Did not converge")

        # Multi-angle comparison
        if len(angle_results) > 0:
            print("\nGenerating multi-angle comparison plot...")
            plot_multiple_angles(angle_results, naca, base_re)

        # =====================================================================
        # PART 2: REYNOLDS NUMBER COMPARISON AT DIFFERENT Re (alpha=5 deg)
        # =====================================================================
        print("\n" + "="*80)
        print("PART 2: Reynolds Number Comparison at alpha=5 deg")
        print("="*80)

        test_re = [500000, 1000000, 2000000]
        test_alpha = 5.0
        re_results = {}

        for re in test_re:
            print(f"\nProcessing Re = {re:.0e}...")
            case = {
                'naca': naca, 'alpha': test_alpha, 'n_panels': 100,
                'c': base['c'], 'U_inf': base['U_inf'], 'rho': base['rho'],
                'nu': base['U_inf'] * base['c'] / re, 're': re
            }

            solver = VIISolver(case['naca'], case['alpha'], case['n_panels'],
                              max_iter=200, tol=5e-4, relax=0.1)
            result = solver.solve(case['nu'], case['rho'], case['U_inf'])

            if result['converged']:
                print(f"  Converged in {result['iterations']} iterations")
                print(f"  Cl = {result['Cl']:.6f}, Cd = {result['Cd']:.6f}")
                plot_boundary_layer_properties(result, case)
                re_results[re] = result
            else:
                print(f"  Warning: Did not converge")

        # Reynolds comparison
        if len(re_results) > 0:
            print(f"\nGenerating Reynolds number comparison plot for alpha={test_alpha}deg...")
            plot_reynolds_comparison(re_results, naca, test_alpha)

        # =====================================================================
        # PART 3: COMPREHENSIVE MULTI-ANGLE COMPARISON AT DIFFERENT Re
        # =====================================================================
        print("\n" + "="*80)
        print("PART 3: Comprehensive Comparison at Multiple Angles & Reynolds Numbers")
        print("="*80)

        comprehensive_results = {}
        for re in test_re:
            print(f"\nRe = {re:.0e}:")
            comprehensive_results[re] = {}
            for alpha in test_angles:
                print(f"  Processing alpha = {alpha}deg...")
                case = {
                    'naca': naca, 'alpha': alpha, 'n_panels': 100,
                    'c': base['c'], 'U_inf': base['U_inf'], 'rho': base['rho'],
                    'nu': base['U_inf'] * base['c'] / re, 're': re
                }

                solver = VIISolver(case['naca'], case['alpha'], case['n_panels'],
                                  max_iter=200, tol=5e-4, relax=0.1)
                result = solver.solve(case['nu'], case['rho'], case['U_inf'])

                if result['converged']:
                    print(f"    -> Cl={result['Cl']:.4f}, Cd={result['Cd']:.5f}")
                    comprehensive_results[re][alpha] = result
                else:
                    print(f"    -> Did not converge")

            # Generate multi-angle plot for each Re
            if len(comprehensive_results[re]) > 0:
                print(f"  Generating multi-angle plot for Re={re:.0e}...")
                plot_multiple_angles(comprehensive_results[re], naca, re)

        print("\n" + "="*80)
        print("ALL PLOTS GENERATED SUCCESSFULLY!")
        print("="*80)
        print("\nGenerated files in 'results/' directory:")
        print("  - polars.png: Aerodynamic polars for Re=500k")
        print("  - bl_properties_alpha*.png: Individual angle/Re combinations")
        print("  - bl_multi_angle_re*.png: Multi-angle comparisons")
        print("  - bl_reynolds_comparison_alpha*.png: Reynolds comparisons")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
