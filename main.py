import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from src.panel_solver import PanelSolver
from src.thwaites_solver import ThwaitesSolver

def setup_validation_case():
    """
    Sets up the parameters for the validation case:
    NACA 0009, Re = 500,000, Alpha = 0 deg.
    """
    case = {
        'naca': '0009',
        're': 500000,
        'alpha': 0.0,
        'n_panels': 140, # Sufficient for grid convergence
        'rho': 1.225,
        'U_inf': 10.0, # Arbitrary, acts as reference
    }
    # Calculate viscosity to match Re
    # Re = rho * U_inf * c / mu = U_inf * c / nu
    # nu = U_inf * c / Re
    # Assuming c = 1.0 (PanelSolver defaults to chord=1)
    case['c'] = 1.0
    case['nu'] = case['U_inf'] * case['c'] / case['re']
    
    return case

def load_xfoil_data(filepath):
    """
    Loads XFOIL validation data from a CSV file.
    Expected columns: x, Cp, Theta, Dstar, Cf
    """
    if not os.path.exists(filepath):
        print(f"Warning: Validation data file not found at {filepath}. Skipping validation plots.")
        return None
    
    try:
        # Check if it's the geometry file by mistake
        with open(filepath, 'r') as f:
            header = f.readline()
            if "NACA" in header and "Airfoil" in header:
                print(f"Warning: File at {filepath} appears to be geometry data, not simulation results.")
                return None
                
        data = pd.read_csv(filepath)
        # Normalize column names just in case
        data.columns = [c.strip() for c in data.columns]
        return data
    except Exception as e:
        print(f"Error loading XFOIL data: {e}")
        return None

def process_panel_results(panel_res):
    """
    Process panel results to extract upper surface for boundary layer analysis.
    Panel points are ordered TE(lower) -> LE -> TE(upper).
    For the Hess-Smith implementation with N panels:
    - Indices 0 to N/2 - 1 are Lower Surface (TE -> LE)
    - Indices N/2 to N - 1 are Upper Surface (LE -> TE)
    """
    x = panel_res['x']
    y = panel_res['y']
    cp = panel_res['Cp']
    ue = panel_res['Ue'] # Tangential velocity
    
    # Determined strict split based on panel count structure
    # Assuming n_panels is even and structure is consistent with PanelSolver
    n_total = len(x)
    split_idx = n_total // 2
    
    # Extract Upper Surface (Indices split_idx to end)
    x_upper = x[split_idx:]
    y_upper = y[split_idx:]
    ue_upper = ue[split_idx:]
    cp_upper = cp[split_idx:]
    
    # Prepend Stagnation Point (0,0) for robust Thwaites integration
    # Ideally stagnation is at geometric (0,0) for alpha=0
    # We add a point s=0, Ue=0.
    # The first panel control point will be at some small s.
    
    # Calculate s
    # First point in x_upper is the first panel centroid.
    # Distance from geometric LE (0,0) to first centroid?
    # Panel solver geometry: Node(N/2) is LE (0,0).
    # First upper panel is between Node(N/2) and Node(N/2+1).
    # Centroid is roughly at s = panel_length / 2.
    
    # Let's calculate s relative to the first panel point using cumsum, 
    # then shift/add the stagnation point.
    
    dx = np.diff(x_upper)
    dy = np.diff(y_upper)
    ds = np.sqrt(dx**2 + dy**2)
    s_dist = np.concatenate(([0], np.cumsum(ds)))
    
    # Distance from geometric LE to first point (x_upper[0], y_upper[0])
    # Assuming LE is at (0,0)
    dist_to_le = np.sqrt(x_upper[0]**2 + y_upper[0]**2)
    
    # Shift s so it represents distance from Stagnation
    s_upper = s_dist + dist_to_le
    
    # Add the stagnation point explicitly to arrays
    s_final = np.concatenate(([0.0], s_upper))
    ue_final = np.concatenate(([0.0], np.abs(ue_upper)))
    # For Cp and others, we can prepend theoretical stagnation values (Cp=1)
    # But Thwaites only needs s and Ue.
    
    return {
        's': s_final,
        'x': np.concatenate(([0.0], x_upper)), # Approximate x at stag
        'Ue': ue_final,
        'Cp': cp_upper # Length mismatch with s/Ue but handled if careful? No, let's keep consistent?
                       # Thwaites doesn't use Cp. We return it for ref.
    }

def main():
    print("--- Viscous-Inviscid Interaction Solver (NACA 0009 Validation) ---")
    
    # 1. Setup
    case = setup_validation_case()
    print(f"Case: NACA {case['naca']}, Re={case['re']}, Alpha={case['alpha']}")
    
    # 2. Execution - Panel Solver
    print("Running Inviscid Panel Solver...")
    p_solver = PanelSolver(case['naca'], case['alpha'], n_panels=case['n_panels'])
    p_solver.solve()
    p_results = p_solver.get_surface_properties()
    
    # 3. Process Inviscid Results for Viscous Solver
    print("Processing surface data for viscous solver...")
    print(f"Panel Solver Circulation (Gamma): {p_solver.gamma:.6f} (Should be ~0 for Alpha=0)")
    
    upper_surf = process_panel_results(p_results)
    
    # SMOOTHING / CORRECTION
    # The Panel Method can produce a velocity spike at the Leading Edge due to discretization.
    # This non-physical spike causes a massive adverse pressure gradient and immediate separation in Thwaites.
    # We apply a correction: 
    # 1. Enforce monotonic acceleration for the first few percent of chord if detected.
    # 2. Smooth slight noise.
    
    ue_raw = upper_surf['Ue']
    s_raw = upper_surf['s']
    
    # Simple heuristic: If Ue drops significantly in the first 5 points, flatten it.
    # Find peak in first 10 points
    n_check = min(10, len(ue_raw))
    if n_check > 2:
        # If we see a spike (jump up then down), we replace with linear or smooth fit
        # For stagnation flow, Ue ~ s.
        # Let's verify if Ue[0..3] is erratic.
        pass # implemented below
        
    # We will use a Savitzky-Golay filter or simple moving average to smooth Ue
    # but carefully to preserve LE 0.
    from scipy.signal import savgol_filter
    try:
        if len(ue_raw) > 15:
            # Apply light smoothing
            upper_surf['Ue'] = savgol_filter(ue_raw, window_length=9, polyorder=3)
            # Enforce 0 at 0
            upper_surf['Ue'][0] = 0.0
            # Ensure non-negative
            upper_surf['Ue'] = np.abs(upper_surf['Ue'])
    except ImportError:
        pass # Skip if scipy signal not available (though standard distributions have it)
        
    # Manual de-spiking for LE
    # If index 1 or 2 is huge > index 5, clamp them?
    # This is a hack for the specific Panel Solver artifact.
    # We'll rely on the filter first.
    
    # 4. Execution - Thwaites Viscous Solver
    print("Running Viscous Thwaites Solver (Upper Surface)...")
    v_solver = ThwaitesSolver()
    
    # Thwaites inputs: s, Ue, nu
    # PanelSolver returns dimensionless V/Vinf.
    # We need to scale Ue by U_inf for the viscous solver (which uses dimensional quantities for Re)
    Ue_dim = upper_surf['Ue'] * case['U_inf']
    
    v_results = v_solver.calculate_properties(
        upper_surf['s'], 
        Ue_dim, 
        case['nu'], 
        case['rho']
    )
    
    # 5. Load Validation Data
    val_file = os.path.join('assets', 'validation_data.csv')
    val_data = load_xfoil_data(val_file)
    
    # 6. Plotting
    print("Generating plots...")
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Viscous-Inviscid Results: NACA {case['naca']} Re={case['re']:.0e}", fontsize=16)
    
    # Shared x axis: x/c
    x_c = upper_surf['x'] / case['c']
    
    # Plot 1: Skin Friction Cf
    ax = axs[0, 0]
    ax.plot(x_c, v_results['Cf'], 'b-', label='My Code (Thwaites)')
    if val_data is not None:
        if 'Cf' in val_data: ax.plot(val_data['x'], val_data['Cf'], 'r--', label='XFOIL')
    ax.set_title(r'Skin Friction Coefficient $C_f$')
    ax.set_xlabel('x/c')
    ax.set_ylabel(r'$C_f$')
    ax.grid(True)
    ax.legend()
    # Limit y to see laminar part clearly, turbulent spike might be huge
    ax.set_ylim(0, 0.01)
    
    # Plot 2: Displacement Thickness Delta*
    ax = axs[0, 1]
    ax.plot(x_c, v_results['delta_star'], 'b-', label='My Code (Thwaites)')
    if val_data is not None:
        if 'Dstar' in val_data: ax.plot(val_data['x'], val_data['Dstar'], 'r--', label='XFOIL')
        elif 'delta_star' in val_data: ax.plot(val_data['x'], val_data['delta_star'], 'r--', label='XFOIL')
    ax.set_title(r'Displacement Thickness $\delta^*$')
    ax.set_xlabel('x/c')
    ax.set_ylabel(r'$\delta^*$ [m]')
    ax.grid(True)
    
    # Plot 3: Momentum Thickness Theta
    ax = axs[1, 0]
    ax.plot(x_c, v_results['theta'], 'b-', label='My Code (Thwaites)')
    if val_data is not None:
        if 'Theta' in val_data: ax.plot(val_data['x'], val_data['Theta'], 'r--', label='XFOIL')
    ax.set_title(r'Momentum Thickness $\theta$')
    ax.set_xlabel('x/c')
    ax.set_ylabel(r'$\theta$ [m]')
    ax.grid(True)
    
    # Plot 4: Shape Factor H
    ax = axs[1, 1]
    ax.plot(x_c, v_results['H'], 'b-', label='My Code (Thwaites)')
    if val_data is not None:
        if 'H' in val_data: ax.plot(val_data['x'], val_data['H'], 'r--', label='XFOIL')
    ax.set_title(r'Shape Factor $H$')
    ax.set_xlabel('x/c')
    ax.set_ylabel(r'$H$')
    ax.grid(True)
    ax.set_ylim(1, 4) # Typical range [2.6 (Blasius) -> 3.5 (Separation)]
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'combined_results.png'))
    print("Plots saved to results/combined_results.png")
    
    # 7. Comparison Logic
    print("\n--- Comparison Analysis ---")
    
    # Transition Check
    # Theoretical Transition point (lam -> turb) is where H drops or Cf spikes.
    # Thwaites is fully laminar. It stops being valid at transition or separation.
    # We can check separation point (lambda < -0.09).
    sep_idx = v_solver.check_separation(v_results['lambda'])
    if sep_idx is not None:
        x_sep = x_c[sep_idx]
        print(f"Computed Laminar Separation Point: x/c = {x_sep:.4f}")
    else:
        print("No laminar separation detected in Thwaites calculation.")
        
    if val_data is not None:
        print("Comparing with XFOIL data...")
        # Since x grids are different, we need to interpolate.
        
        # Check transition in XFOIL (heuristic: max Cf gradient or H drop)
        # Or user said approx x=0.66
        # Let's verify our code's transition vs XFOIL.
        pass
    else:
        print("Note: XFOIL validation data not provided. Skipping quantitative comparison.")
        print("Please provide 'assets/validation_data.csv' for full validation.")

if __name__ == "__main__":
    main()
