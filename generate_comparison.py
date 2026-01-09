import numpy as np
import matplotlib.pyplot as plt
import os
import re
from src.vii_solver import VIISolver

# --- CONFIGURATION ---
NACA = '0009'
RE = 100000
N_PANELS = 100 # Should match JavaFoil density roughly
JAVA_FOIL_POLAR_FILE = 'assets/cl_result_javafoil.txt'
JAVA_FOIL_BL_FILE = 'assets/java_foil_data.txt'
OUTPUT_DIR = 'results_custom'

def parse_javafoil_polar(filepath):
    """Parses the JavaFoil polar text file."""
    data = {'alpha': [], 'Cl': [], 'Cd': [], 'Cm': [], 'x_cp': []}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip header lines until data starts
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('6:'): # The line number in view_file output was 6 for first data point
            # Wait, view_file added line numbers. In real file, it might be different.
            # Looking at file content:
            # Line 6: -4.0 ...
            # Header ends at line 5.
            pass
            
    # Regex to extract float numbers
    # Real file handling (assuming no line numbers from view_file)
    # The file has a header. Data usually starts after units line.
    
    in_data = False
    for line in lines:
        if not line.strip(): continue
        
        # Heuristic to find data lines (start with a number)
        parts = line.split()
        try:
            val = float(parts[0])
            # If successful, likely a data line. 
            # JavaFoil format cols: alpha, Cl, Cd, Cm, TU, TL, SU, SL, L/D, AC, CP
            # index: 0, 1, 2, 3, ..., 10
            if len(parts) >= 11:
                data['alpha'].append(float(parts[0]))
                data['Cl'].append(float(parts[1]))
                data['Cd'].append(float(parts[2]))
                data['Cm'].append(float(parts[3]))
                data['x_cp'].append(float(parts[10]))
        except ValueError:
            continue
            
    return data

def parse_javafoil_bl(filepath):
    """Parses JavaFoil Boundary Layer data (AoA=0)."""
    data = {'x': [], 'delta_star': [], 'theta': [], 'H': [], 'Cf': []}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    start_parsing = False
    for line in lines:
        parts = line.split()
        if len(parts) == 0: continue
        
        # Header check
        if 'x/c' in parts and 'y/c' in parts:
            start_parsing = True
            continue
        
        if not start_parsing: continue
        
        try:
            # Check if first col is float
            xc = float(parts[0])
            # Columns based on file view:
            # x/c (0), y/c(1), v/V(2), delta_1(3), delta_2(4), delta_3(5), ... Cf(7), H12(8) ...
            # delta_1 = displacement (delta*)
            # delta_2 = momentum (theta)
            # H12 = Shape Factor (H)
            
            data['x'].append(xc)
            data['delta_star'].append(float(parts[3]))
            data['theta'].append(float(parts[4]))
            data['Cf'].append(float(parts[7]))
            data['H'].append(float(parts[8]))
        except ValueError:
            continue
            
    return data

def run_vii_sweep(alphas):
    results = {'alpha': [], 'Cl': [], 'Cd': [], 'Cm': [], 'x_cp': []}
    
    print(f"Running VII Sweep for {len(alphas)} angles...")
    
    # Setup setup
    c = 1.0
    U_inf = 10.0
    nu = U_inf * c / RE
    rho = 1.225
    
    for alpha in alphas:
        print(f"  Calculating Alpha = {alpha:.1f}...", end="\r")
        # FIXED: Increased max_iter from 20 to 200, tightened tolerance
        solver = VIISolver(NACA, alpha, N_PANELS, max_iter=200, tol=5e-4, relax=0.1)
        res = solver.solve(nu, rho, U_inf)

        if res['converged']:
            results['alpha'].append(alpha)
            results['Cl'].append(res['Cl'])
            results['Cd'].append(res['Cd'])
            results['Cm'].append(res['Cm'])
            results['x_cp'].append(res['x_cp'])
            print(f"  Alpha = {alpha:.1f}: Converged in {res['iterations']} iters - Cl={res['Cl']:.4f}, Cd={res['Cd']:.5f}")
        else:
            print(f"  Alpha = {alpha:.1f}: FAILED TO CONVERGE after {res['iterations']} iterations")
            # If not converged, maybe append None or last value?
            # For plotting, better to skip or interpolate.
            # Appending valid data only for now.
            pass
            
    print("\nSweep Complete.")
    return results

def run_vii_bl_zero():
    print("Running VII at Alpha=0 for BL comparison...")
    c = 1.0
    U_inf = 10.0
    nu = U_inf * c / RE
    rho = 1.225

    # FIXED: Increased max_iter and set proper tolerance for precision
    solver = VIISolver(NACA, 0.0, N_PANELS, max_iter=200, tol=5e-4, relax=0.1)
    res = solver.solve(nu, rho, U_inf)

    if res['converged']:
        print(f"  Converged in {res['iterations']} iterations")
        print(f"  Cl = {res['Cl']:.6f}, Cd = {res['Cd']:.6f}")
    else:
        print(f"  WARNING: Did not converge after {res['iterations']} iterations!")

    return res

def plot_polars(vii, java):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'NACA {NACA} Re={RE} - VII vs JavaFoil')
    
    # Cl vs Alpha
    axs[0,0].plot(java['alpha'], java['Cl'], 'b--', label='JavaFoil')
    axs[0,0].plot(vii['alpha'], vii['Cl'], 'r-', label='Thwaites Method')
    axs[0,0].set_xlabel('Alpha [deg]')
    axs[0,0].set_ylabel('Cl')
    axs[0,0].grid(True, alpha=0.3)
    axs[0,0].legend()
    
    # Cd vs Alpha (or Cl vs Cd)
    # Let's do Cd vs Alpha
    axs[0,1].plot(java['alpha'], java['Cd'], 'b--', label='JavaFoil')
    axs[0,1].plot(vii['alpha'], vii['Cd'], 'r-', label='Thwaites Method')
    axs[0,1].set_xlabel('Alpha [deg]')
    axs[0,1].set_ylabel('Cd')
    axs[0,1].grid(True, alpha=0.3)
    
    # Cm vs Alpha
    axs[1,0].plot(java['alpha'], java['Cm'], 'b--', label='JavaFoil')
    axs[1,0].plot(vii['alpha'], vii['Cm'], 'r-', label='Thwaites Method')
    axs[1,0].set_xlabel('Alpha [deg]')
    axs[1,0].set_ylabel('Cm (0.25c)')
    axs[1,0].grid(True, alpha=0.3)
    
    # CP vs Alpha
    axs[1,1].plot(java['alpha'], java['x_cp'], 'b--', label='JavaFoil')
    axs[1,1].plot(vii['alpha'], vii['x_cp'], 'r-', label='Thwaites Method')
    axs[1,1].set_xlabel('Alpha [deg]')
    axs[1,1].set_ylabel('x_cp / c')
    axs[1,1].grid(True, alpha=0.3)
    axs[1,1].set_ylim(0, 1) # Range usually 0.25 to 0.5 or inf
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/vii_vs_javafoil_polars.png')
    plt.close()

def plot_bl_comparison(vii_res, java_bl):
    # VII results need to be extracted for Upper Surface
    # JavaFoil data seems to be upper surface (starts at x=1, goes to 0) or similar.
    # checking java_foil_data.txt: x/c from 1.000 to 0.000. This is usually Upper Surface (if y>0) or Lower.
    # Line 5: Upper Surface = 100%. Line 13 headers.
    # Usually JavaFoil outputs for one side or checks both?
    # Let's assume the provided file is Upper Surface (most critical for BL).
    
    # VII Data
    vii_up = vii_res['upper']
    x_vii = vii_up['x']
    ds_vii = vii_up['delta_star']
    th_vii = vii_up['theta']
    h_vii = vii_up['H']
    cf_vii = vii_up['Cf']
    
    # JavaFoil Data
    x_jav = java_bl['x']
    ds_jav = java_bl['delta_star']
    th_jav = java_bl['theta']
    h_jav = java_bl['H']
    cf_jav = java_bl['Cf']
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Boundary Layer (Upper Surface) - Alpha=0, Re={RE}')
    
    # Delta Star
    axs[0,0].plot(x_jav, ds_jav, 'b--', label='JavaFoil')
    axs[0,0].plot(x_vii, ds_vii, 'r-', label='Thwaites')
    axs[0,0].set_xlabel('x/c')
    axs[0,0].set_ylabel('Displacement Thickness (delta*)')
    axs[0,0].grid(True)
    axs[0,0].legend()
    
    # Theta
    axs[0,1].plot(x_jav, th_jav, 'b--')
    axs[0,1].plot(x_vii, th_vii, 'r-')
    axs[0,1].set_xlabel('x/c')
    axs[0,1].set_ylabel('Momentum Thickness (theta)')
    axs[0,1].grid(True)
    
    # H
    axs[1,0].plot(x_jav, h_jav, 'b--')
    axs[1,0].plot(x_vii, h_vii, 'r-')
    axs[1,0].set_xlabel('x/c')
    axs[1,0].set_ylabel('Shape Factor (H)')
    axs[1,0].set_ylim(1, 5)
    axs[1,0].grid(True)
    
    # Cf
    axs[1,1].plot(x_jav, cf_jav, 'b--')
    axs[1,1].plot(x_vii, cf_vii, 'r-')
    axs[1,1].set_xlabel('x/c')
    axs[1,1].set_ylabel('Skin Friction (Cf)')
    axs[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/vii_vs_javafoil_bl_dist.png')
    plt.close()

def write_result_file(data):
    filepath = f'{OUTPUT_DIR}/vii_result_javafoil_format.txt'
    with open(filepath, 'w') as f:
        f.write(f"Name = NACA {NACA}\n")
        f.write(f"Re = {RE}\n")
        f.write("Alpha\tCl\tCd\tCm\tx_cp\n")
        f.write("[deg]\t[-]\t[-]\t[-]\t[-]\n")
        
        for i in range(len(data['alpha'])):
            f.write(f"{data['alpha'][i]:.1f}\t{data['Cl'][i]:.5f}\t{data['Cd'][i]:.5f}\t{data['Cm'][i]:.5f}\t{data['x_cp'][i]:.5f}\n")
            
    print(f"Results written to {filepath}")

def main():
    print("Parsing JavaFoil Data...")
    jf_polar = parse_javafoil_polar(JAVA_FOIL_POLAR_FILE)
    jf_bl = parse_javafoil_bl(JAVA_FOIL_BL_FILE)
    
    # Use Alpha range from JavaFoil file
    alphas = sorted(list(set(jf_polar['alpha'])))
    # Filter to reasonable range if needed, e.g. -4 to 8 as per plot
    alphas = [a for a in alphas if -4.0 <= a <= 8.0]
    
    vii_polar = run_vii_sweep(alphas)
    
    print("Generating Polar Comparisons...")
    plot_polars(vii_polar, jf_polar)
    write_result_file(vii_polar)
    
    print("Analyzing Boundary Layer at Alpha=0...")
    vii_bl_res = run_vii_bl_zero()
    plot_bl_comparison(vii_bl_res, jf_bl)
    
    print("Done.")

if __name__ == "__main__":
    main()
