import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from src.panel_solver import PanelSolver
from src.vii_solver import VIISolver

def load_javafoil_data(filepath):
    """Load JavaFoil data from text file"""
    # Read the file, skipping header lines
    df = pd.read_csv(filepath, sep='\t', skiprows=13, skipinitialspace=True)

    # Split into upper and lower surfaces
    # Upper surface: x decreases from 1.0 to 0.0
    # Lower surface: x increases from 0.0 to 1.0

    # Find the split point (where x changes direction)
    upper_data = []
    lower_data = []

    for i in range(len(df)):
        x = df.iloc[i, 0]  # x/c column
        if i < len(df) - 1:
            x_next = df.iloc[i+1, 0]
            if x >= x_next:  # Upper surface (decreasing x)
                upper_data.append(i)
            else:  # Lower surface (increasing x)
                lower_data.append(i)
        else:
            lower_data.append(i)

    upper = df.iloc[upper_data].copy()
    lower = df.iloc[lower_data].copy()

    # Reverse upper surface to have increasing x
    upper = upper.iloc[::-1].reset_index(drop=True)
    lower = lower.reset_index(drop=True)

    return upper, lower, df

def run_simulation(Re=100000, alpha=0.0):
    """Run VII simulation"""
    naca = '0009'
    n_panels = 100
    c = 1.0
    U_inf = 10.0
    nu = U_inf * c / Re
    rho = 1.225

    print(f"\nRunning Pohlhausen Method simulation:")
    print(f"  NACA {naca}, Re={Re}, AoA={alpha}°")

    solver = VIISolver(naca, alpha, n_panels, max_iter=200, tol=5e-4, relax=0.1)
    result = solver.solve(nu, rho, U_inf)

    if result['converged']:
        print(f"  Converged in {result['iterations']} iterations")
        print(f"  Cl = {result['Cl']:.6f}, Cd = {result['Cd']:.6f}")
    else:
        print(f"  Warning: Did not converge")

    return result

def plot_validation(result, jf_upper, jf_lower, Re, alpha):
    """Create validation plots comparing with JavaFoil"""
    os.makedirs('results', exist_ok=True)

    # Extract data
    upper = result['upper']
    lower = result['lower']

    # Create 6 subplots (3x2)
    fig = plt.figure(figsize=(16, 18))

    # ========== Plot 1: Pressure Distribution ==========
    ax1 = plt.subplot(3, 2, 1)

    # Compute Cp from velocity: Cp = 1 - (v/V_inf)^2
    jf_cp_upper = 1 - jf_upper.iloc[:, 2].values**2
    jf_cp_lower = 1 - jf_lower.iloc[:, 2].values**2

    # Our Cp (already computed by panel method)
    # Get panel method results for Cp
    panel = PanelSolver('0009', alpha, 100)
    panel.solve()
    res = panel.get_surface_properties()
    split = len(res['Cp']) // 2
    our_cp_upper = res['Cp'][split:]
    our_cp_lower = res['Cp'][:split][::-1]
    our_x_upper = res['x'][split:]
    our_x_lower = res['x'][:split][::-1]

    ax1.plot(our_x_upper, our_cp_upper, 'b-', linewidth=1.5, label='Pohlhausen Method (upper)')
    ax1.plot(jf_upper.iloc[:, 0], jf_cp_upper, 'b--', linewidth=1.5, label='Javafoil (upper)')
    ax1.plot(our_x_lower, our_cp_lower, 'orange', linewidth=1.5, label='Pohlhausen Method (lower)')
    ax1.plot(jf_lower.iloc[:, 0], jf_cp_lower, 'orange', linestyle='--', linewidth=1.5, label='Javafoil (lower)')

    ax1.invert_yaxis()
    ax1.set_xlabel('x')
    ax1.set_ylabel('$C_p$')
    ax1.set_title(f'Pressure Distribution ($C_p$)\\nAoA = {alpha} deg. Re = {Re}\\nNACA 0009')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])

    # ========== Plot 2: Velocity Distribution ==========
    ax2 = plt.subplot(3, 2, 2)

    # JavaFoil v/V data
    jf_v_upper = jf_upper.iloc[:, 2].values
    jf_v_lower = jf_lower.iloc[:, 2].values

    # Our velocity distribution (from edge velocity)
    our_v_upper = upper['Ue'] / 10.0  # Normalize by U_inf
    our_v_lower = lower['Ue'] / 10.0

    ax2.plot(upper['x'], our_v_upper, 'b-', linewidth=1.5, label='Pohlhausen Method (upper)')
    ax2.plot(jf_upper.iloc[:, 0], jf_v_upper, 'b--', linewidth=1.5, label='Javafoil (upper)')
    ax2.plot(lower['x'], our_v_lower, 'orange', linewidth=1.5, label='Pohlhausen Method (lower)')
    ax2.plot(jf_lower.iloc[:, 0], jf_v_lower, 'orange', linestyle='--', linewidth=1.5, label='Javafoil (lower)')

    ax2.set_xlabel('x')
    ax2.set_ylabel('v/V$_\\infty$')
    ax2.set_title(f'Velocity Distribution (v/V$_\\infty$)\\nAoA = {alpha} deg. Re = {Re}\\nNACA 0009')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])

    # ========== Plot 3: Momentum Thickness ==========
    ax3 = plt.subplot(3, 2, 3)

    # JavaFoil momentum thickness (δ_2 = θ)
    jf_theta_upper = jf_upper.iloc[:, 4].values  # Column δ_2
    jf_theta_lower = jf_lower.iloc[:, 4].values

    # Our momentum thickness
    our_theta_upper = upper['theta']
    our_theta_lower = lower['theta']

    ax3.plot(upper['x'], our_theta_upper, 'b-', linewidth=1.5, label='Pohlhausen Method (upper)')
    ax3.plot(jf_upper.iloc[:, 0], jf_theta_upper, 'b--', linewidth=1.5, label='Javafoil (upper)')
    ax3.plot(lower['x'], our_theta_lower, 'orange', linewidth=1.5, label='Pohlhausen Method (lower)')
    ax3.plot(jf_lower.iloc[:, 0], jf_theta_lower, 'orange', linestyle='--', linewidth=1.5, label='Javafoil (lower)')

    ax3.set_xlabel('x')
    ax3.set_ylabel(r'Momentum thickness ($\theta$)')
    ax3.set_title(f'Momentum thickness ($\\theta$)\\nAoA = {alpha} deg. Re = {Re}\\nNACA 0009')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])

    # ========== Plot 4: Displacement Thickness ==========
    ax4 = plt.subplot(3, 2, 4)

    # JavaFoil displacement thickness (δ_1 = δ*)
    jf_delta_upper = jf_upper.iloc[:, 3].values  # Column δ_1
    jf_delta_lower = jf_lower.iloc[:, 3].values

    # Our displacement thickness
    our_delta_upper = upper['delta_star']
    our_delta_lower = lower['delta_star']

    ax4.plot(upper['x'], our_delta_upper, 'b-', linewidth=1.5, label='Pohlhausen Method (upper)')
    ax4.plot(jf_upper.iloc[:, 0], jf_delta_upper, 'b--', linewidth=1.5, label='Javafoil (upper)')
    ax4.plot(lower['x'], our_delta_lower, 'orange', linewidth=1.5, label='Pohlhausen Method (lower)')
    ax4.plot(jf_lower.iloc[:, 0], jf_delta_lower, 'orange', linestyle='--', linewidth=1.5, label='Javafoil (lower)')

    ax4.set_xlabel('x')
    ax4.set_ylabel(r'Displacement thickness ($\delta^*$)')
    ax4.set_title(f'Displacement thickness ($\\delta^*$)\\nAoA = {alpha} deg. Re = {Re}\\nNACA 0009')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 1])

    # ========== Plot 5: Friction Coefficient ==========
    ax5 = plt.subplot(3, 2, 5)

    # JavaFoil friction coefficient
    jf_cf_upper = jf_upper.iloc[:, 7].values  # Column C_f
    jf_cf_lower = jf_lower.iloc[:, 7].values

    # Our friction coefficient
    our_cf_upper = upper['Cf']
    our_cf_lower = lower['Cf']

    ax5.plot(upper['x'], our_cf_upper, 'b-', linewidth=1.5, label='Pohlhausen Method (upper)')
    ax5.plot(jf_upper.iloc[:, 0], jf_cf_upper, 'b--', linewidth=1.5, label='Javafoil (upper)')
    ax5.plot(lower['x'], our_cf_lower, 'orange', linewidth=1.5, label='Pohlhausen Method (lower)')
    ax5.plot(jf_lower.iloc[:, 0], jf_cf_lower, 'orange', linestyle='--', linewidth=1.5, label='Javafoil (lower)')

    ax5.set_xlabel('x')
    ax5.set_ylabel('$c_f$')
    ax5.set_title(f'Friction Coefficient ($c_f$)\\nAoA = {alpha} deg. Re = {Re}\\nNACA 0009')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, None])

    # ========== Plot 6: Airfoil Shape with Boundary Layer ==========
    ax6 = plt.subplot(3, 2, 6)

    # Get airfoil geometry
    panel = PanelSolver('0009', alpha, 100)
    airfoil_x = panel.x
    airfoil_y = panel.y

    # Plot airfoil
    ax6.plot(airfoil_x, airfoil_y, 'k-', linewidth=2, label='Airfoil')

    # Plot boundary layer edges (airfoil + delta*)
    split = len(airfoil_x) // 2

    # Upper surface BL edge
    upper_x = airfoil_x[split:]
    upper_y = airfoil_y[split:]
    # Interpolate delta* to airfoil points
    upper_delta_interp = np.interp(upper_x, upper['x'], upper['delta_star'])
    # Add delta* normal to surface (approximate as vertical for thin airfoil)
    upper_bl_y = upper_y + upper_delta_interp
    ax6.plot(upper_x, upper_bl_y, 'b-', linewidth=1.5, label='BL edge (upper)')

    # Lower surface BL edge
    lower_x = airfoil_x[:split+1][::-1]
    lower_y = airfoil_y[:split+1][::-1]
    lower_delta_interp = np.interp(lower_x, lower['x'], lower['delta_star'])
    lower_bl_y = lower_y - lower_delta_interp
    ax6.plot(lower_x, lower_bl_y, 'orange', linewidth=1.5, label='BL edge (lower)')

    ax6.set_xlabel('x/c')
    ax6.set_ylabel('Thickness (x/c)')
    ax6.set_title('NACA-0009\\nAirfoil Shape and Boundary Layer')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axis('equal')
    ax6.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(f'results/validation_javafoil_Re{Re}_alpha{alpha:.0f}.png', dpi=150)
    print(f"\nValidation plots saved: validation_javafoil_Re{Re}_alpha{alpha:.0f}.png")
    plt.close()

def main():
    # Load JavaFoil data
    jf_file = 'assets/java_foil_data.txt'
    print(f"Loading JavaFoil data from: {jf_file}")
    jf_upper, jf_lower, jf_all = load_javafoil_data(jf_file)
    print(f"  Loaded {len(jf_upper)} upper surface points")
    print(f"  Loaded {len(jf_lower)} lower surface points")

    # Run simulation
    Re = 100000
    alpha = 0.0
    result = run_simulation(Re=Re, alpha=alpha)

    # Create validation plots
    print("\nGenerating validation plots...")
    plot_validation(result, jf_upper, jf_lower, Re, alpha)

    print("\n" + "="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - results/validation_javafoil_Re{Re}_alpha{int(alpha)}.png")

if __name__ == "__main__":
    main()
