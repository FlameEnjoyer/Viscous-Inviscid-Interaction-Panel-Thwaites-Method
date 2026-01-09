import numpy as np
import matplotlib.pyplot as plt
import os
from src.vii_solver import VIISolver

def read_javafoil_data(filepath):
    """
    Reads JavaFoil data from text file.
    Assumes structure:
    Lines 1-13: Header
    Line 14: Units/Labels
    Line 15-65: Upper Surface (x=1.0 -> 0.0)
    Line 66-115: Lower Surface (x=0.0 -> 1.0)
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Start reading from line 15 (index 14)
    # JavaFoil file format based on inspection:
    # 13: x/c y/c ...
    # 15: 1.0000 0.0000 ... (Start of data)
    
    raw_data = []
    for line in lines[14:]: # Index 14 is line 15
        if not line.strip(): continue
        parts = line.split()
        if len(parts) < 5: continue
        try:
            # Parse columns: x/c, y/c, v/V, delta1, delta2, delta3, Re_d2, Cf, H12, H32, State, y1
            row = [float(p) for p in parts[:10]] # Take first 10 numerical columns
            raw_data.append(row)
        except ValueError:
            continue
            
    raw_data = np.array(raw_data)
    
    # Split Upper/Lower
    # Assuming first block is Upper (TE to LE) and second is Lower (LE to TE) based on x/c direction
    # Find index where x starts increasing again (LE)
    
    # Actually, looking at the file viewed:
    # Line 15: 1.0000
    # ...
    # Line 65: 0.0000 (approx LE)
    # Line 66: 0.0010 (Start of Lower)
    
    # Find split point
    x = raw_data[:, 0]
    split_idx = np.argmin(x) # Should be around 0.0
    
    # Upper: 0 to split_idx (inclusive) - strictly it goes 1.0 -> 0.0
    # Lower: split_idx+1 to end - goes 0.0 -> 1.0
    
    # The minimum x is likely at the stagnation point or LE.
    # Let's verify carefully. 
    # Index 50 (approx) in the file was 0.0000.
    # In raw_data, row 50 (index 50) is 0.0000.
    
    upper_data = raw_data[:split_idx+1] # 1.0 down to 0.0
    lower_data = raw_data[split_idx+1:] # 0.0 up to 1.0
    
    # Sort by x for plotting logic if needed, but plotting usually works fine with ordered points
    # For 'Upper', x is decreasing. We might want to sort it for interpolation or just plot as is.
    # The user plots show x 0 to 1.
    
    # Organize into dictionaries
    upper = {
        'x': upper_data[:, 0],
        'v_V': upper_data[:, 2],
        'delta_star': upper_data[:, 3],
        'theta': upper_data[:, 4],
        'Cf': upper_data[:, 7]
    }
    
    lower = {
        'x': lower_data[:, 0],
        'v_V': lower_data[:, 2],
        'delta_star': lower_data[:, 3],
        'theta': lower_data[:, 4],
        'Cf': lower_data[:, 7]
    }
    
    return upper, lower

def run_simulation():
    # Simulation Parameters
    naca = '0009'
    alpha = 0.0
    Re = 100000
    c = 1.0
    U_inf = 10.0
    rho = 1.225
    nu = U_inf * c / Re
    
    print(f"Running Simulation: NACA {naca}, Re={Re}, alpha={alpha}")
    
    case = {
        'naca': naca, 'alpha': alpha, 'n_panels': 140, # increased panels for smoothness
        'c': c, 'U_inf': U_inf, 'rho': rho,
        'nu': nu, 're': Re
    }
    
    solver = VIISolver(case['naca'], case['alpha'], case['n_panels'], max_iter=200, tol=1e-4, relax=0.1)
    res = solver.solve(case['nu'], case['rho'], case['U_inf'])
    
    if not res['converged']:
        print("Warning: Simulation did not converge!")
    else:
        print("Simulation Converged.")
        
    return res

def plot_results(sim_res, java_upper, java_lower):
    os.makedirs('results_custom', exist_ok=True)
    
    # Extract Simulation Data (Symmetric, so Upper is sufficient)
    sim_data = sim_res['upper']
    java_data = java_upper
    
    # Common Plot Settings
    # Style: Thwaites (Solid), Javafoil (Dashed)
    # Color: Blue for both (or distinct). Let's use Blue for Sim, Orange for Validation to match "Just like before" contrast, 
    # OR follow the "Blue Solid vs Blue Dashed" style from the user's reference Figure 11 for a single curve.
    # Looking at Fig 11: Lower was Blue Solid vs Blue Dashed. Upper was Yellow Solid vs Yellow Dashed.
    # Since we are plotting one, I will use Blue Solid vs Blue Dashed.
    
    c_sim = '#1f77b4' # Blue
    c_val = '#ff7f0e' # Orange - Let's use Orange for validation to make it distinct as per "different the line style" and contrast. 
                      # actually, previous was blue/blue and orange/orange. 
                      # If I merge, Blue Solid vs Orange Dashed is very clear.
                      # BUT "just like before" implies keeping the style. 
                      # I will use Blue Solid (Sim) and Blue Dashed (Javafoil) to mimic the "Lower" curves from before, which is a standard "single surface" look.
                      # actually, let's use the colors the user might prefer visually. 
                      # I will use Blue Solid (Thwaites) and Orange Dashed (Javafoil) to ensure they are distinct.
    
    # actually, user said "just like examples". Example Fig 11 had:
    # Pohlhausen (lower) - Blue, Javafoil (lower) - Blue Dashed.
    # I will stick to this monochrome-per-set style. 
    # Thwaites: Blue Solid. Javafoil: Blue Dashed.
    
    color_sim = '#1f77b4'
    color_val = '#1f77b4' # Same color, different style
    
    # 1. Friction Coefficient (Cf)
    plt.figure(figsize=(8, 6))
    plt.plot(sim_data['x'], sim_data['Cf'], color=color_sim, linestyle='-', label='Thwaites Method')
    plt.plot(java_data['x'], java_data['Cf'], color=color_val, linestyle='--', label='Javafoil')
    
    plt.title(f"Friction Coefficient ($c_f$)\nAoA = 0 deg. Re = 100000\nNACA 0009", fontweight='bold')
    plt.xlabel('x')
    plt.ylabel('y') # Keeping 'y' as per previous instruction/plots if consistent, though usually 'Cf'
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    plt.savefig('results_custom/friction_coefficient.png', dpi=150)
    plt.close()
    
    # 2. Velocity Distribution (v/V)
    plt.figure(figsize=(8, 6))
    plt.plot(sim_data['x'], sim_data['Ue'], color=color_sim, linestyle='-', label='Thwaites Method')
    plt.plot(java_data['x'], java_data['v_V'], color=color_val, linestyle='--', label='Javafoil')
    
    plt.title(f"Velocity Distribution ($v/V_\\infty$)\nAoA = 0 deg. Re = 100000\nNACA 0009", fontweight='bold')
    plt.xlabel('x')
    plt.ylabel('v/V')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 1)
    plt.savefig('results_custom/velocity_distribution.png', dpi=150)
    plt.close()
    
    # 3. Pressure Distribution (Cp)
    sim_cp = 1 - sim_data['Ue']**2
    java_cp = 1 - java_data['v_V']**2
    
    plt.figure(figsize=(8, 6))
    plt.plot(sim_data['x'], sim_cp, color=color_sim, linestyle='-', label='Thwaites Method')
    plt.plot(java_data['x'], java_cp, color=color_val, linestyle='--', label='Javafoil')
    
    plt.title(f"Pressure Distribution ($C_p$)\nAoA = 0 deg. Re = 100000\nNACA 0009", fontweight='bold')
    plt.xlabel('x')
    plt.ylabel('Cp')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 1)
    plt.gca().invert_yaxis()
    plt.savefig('results_custom/pressure_distribution.png', dpi=150)
    plt.close()
    
    # 4. Momentum Thickness (theta)
    plt.figure(figsize=(8, 6))
    plt.plot(sim_data['x'], sim_data['theta'], color=color_sim, linestyle='-', label='Thwaites Method')
    plt.plot(java_data['x'], java_data['theta'], color=color_val, linestyle='--', label='Javafoil')
    
    plt.title(f"Momentum thickness ($\\theta$)\nAoA = 0 deg. Re = 100000\nNACA 0009", fontweight='bold')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    plt.savefig('results_custom/momentum_thickness.png', dpi=150)
    plt.close()
    
    # 5. Displacement Thickness (delta*)
    plt.figure(figsize=(8, 6))
    plt.plot(sim_data['x'], sim_data['delta_star'], color=color_sim, linestyle='-', label='Thwaites Method')
    plt.plot(java_data['x'], java_data['delta_star'], color=color_val, linestyle='--', label='Javafoil')
    
    plt.title(f"Displacement thickness ($\\delta^*$)\nAoA = 0 deg. Re = 100000\nNACA 0009", fontweight='bold')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    plt.savefig('results_custom/displacement_thickness.png', dpi=150)
    plt.close()
    
    # 6. Airfoil Shape and Boundary Layer (Full Airfoil: Upper + Lower)
    # Reverting to full airfoil plot as requested
    
    sim_lower = sim_res['lower']
    
    from src.panel_solver import PanelSolver
    base_panel = PanelSolver('0009', 0, 100)
    x_base = base_panel.x # Full coordinates
    y_base = base_panel.y
    
    def naca4(x, code='0009'):
        t = int(code[2:]) / 100.0
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        return yt
        
    y_upper_surf = naca4(sim_data['x'])
    y_lower_surf = -naca4(sim_lower['x'])
    
    y_bl_upper = y_upper_surf + sim_data['delta_star']
    y_bl_lower = y_lower_surf - sim_lower['delta_star']
    
    # Combine into a continuous loop (TE Upper -> LE -> TE Lower)
    # x_upper starts at 0 (LE) to 1 (TE), so we reverse it to go TE->LE
    x_bl_loop = np.concatenate([sim_data['x'][::-1], sim_lower['x'][1:]])
    y_bl_loop = np.concatenate([y_bl_upper[::-1], y_bl_lower[1:]])
    
    plt.figure(figsize=(10, 4))
    # Plot Airfoil (Base)
    plt.plot(x_base, y_base, color='#1f77b4', label='Airfoil')
    
    # Plot Boundary Layers (Continuous Loop)
    plt.plot(x_bl_loop, y_bl_loop, color='#ff7f0e', label='Boundary Layer')
    
    plt.title('NACA-0009\nAirfoil Shape and Boundary Layer', fontweight='bold')
    plt.xlabel('x/c')
    plt.ylabel('Thickness (x/c)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('results_custom/airfoil_bl.png', dpi=150)
    plt.close()

def run_sweep():
    # Sweep Parameters
    naca = '0009'
    Re = 100000
    c = 1.0
    U_inf = 10.0
    rho = 1.225
    
    alphas = np.linspace(-4, 8, 13) # -4 to 8 integers
    cl_results = []
    
    print(f"\nRunning Alpha Sweep: {alphas[0]} to {alphas[-1]} degrees...")
    
    for alpha in alphas:
        nu = U_inf * c / Re
        case = {
            'naca': naca, 'alpha': alpha, 'n_panels': 140,
            'c': c, 'U_inf': U_inf, 'rho': rho,
            'nu': nu, 're': Re
        }
        
        solver = VIISolver(case['naca'], case['alpha'], case['n_panels'], max_iter=200, tol=1e-4, relax=0.1)
        res = solver.solve(case['nu'], case['rho'], case['U_inf'])
        
        if res['converged']:
            cl_results.append(res['Cl'])
        else:
            cl_results.append(np.nan)
            
    return alphas, cl_results

def read_polar_data(filepath):
    """
    Reads polar data (Alpha, Cl) from JavaFoil text file.
    Format: First 5 lines are headers, data starts at line 6.
    Columns are tab-separated: Alpha, Cl, Cd, ...
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None, None
        
    alphas = []
    cls = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines[5:]:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    a = float(parts[0])
                    c = float(parts[1])
                    alphas.append(a)
                    cls.append(c)
                except ValueError:
                    continue
                    
        return np.array(alphas), np.array(cls)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None

def plot_lift_curve(alphas, cls, java_alphas=None, java_cls=None):
    plt.figure(figsize=(8, 6))
    
    # Plot Simulation (Pohlhausen Method)
    # Using the label from the reference image
    plt.plot(alphas, cls, color='#1f77b4', linestyle='-', linewidth=1.5, label='Thwaites Method')
    
    # Plot JavaFoil Validation if available
    if java_alphas is not None and java_cls is not None and len(java_alphas) > 0:
        plt.plot(java_alphas, java_cls, color='#1f77b4', linestyle='--', linewidth=1.5, dashes=(5, 5), label='Javafoil')
    
    plt.title(f"Lift Coefficient ($c_l$)\nRe = 100000\nNACA 0009", fontweight='bold')
    plt.xlabel('AoA')
    plt.ylabel('$c_l$')
    plt.grid(True, alpha=0.3)
    
    # Setup Legend with box
    plt.legend(edgecolor='black', framealpha=1.0, fancybox=False)
    
    plt.xlim(-4, 8)
    # plt.ylim(-0.4, 1.4) # Optional: set limits if needed to match reference exactly, but auto is usually safe
    
    plt.tight_layout()
    plt.savefig('results_custom/lift_coefficient.png', dpi=150)
    print("Saved results_custom/lift_coefficient.png")
    plt.close()

if __name__ == "__main__":
    if not os.path.exists('assets/java_foil_data.txt'):
        print("Error: assets/java_foil_data.txt not found.")
    else:
        # Run Profile Analysis (AoA=0)
        # java_upper, java_lower = read_javafoil_data('assets/java_foil_data.txt')
        # res = run_simulation()
        # plot_results(res, java_upper, java_lower)
        
        # We define a separate run just for this task or comment out the single point analysis
        # to focus on the lift curve if that matches the user intent "Update the results of ... lift_coefficient.png"
        
        # But let's keep the previous logic running to ensure everything works
        print("Running Single Point Analysis (AoA=0)...")
        try:
            java_upper, java_lower = read_javafoil_data('assets/java_foil_data.txt')
            res = run_simulation()
            plot_results(res, java_upper, java_lower)
        except Exception as e:
            print(f"Single point analysis failed: {e}")
        
        # Run Lift Sweep
        print("\nRunning Alpha Sweep...")
        alphas, cls = run_sweep()
        
        # Read JavaFoil Polar Data
        java_polar_file = 'assets/cl_result_javafoil.txt'
        java_alphas, java_cls = read_polar_data(java_polar_file)
        
        if java_alphas is None or len(java_alphas) == 0:
            print(f"Warning: No valid data found in {java_polar_file}. Plotting without comparison.")
        
        plot_lift_curve(alphas, cls, java_alphas, java_cls)
        
        print("Done.")
