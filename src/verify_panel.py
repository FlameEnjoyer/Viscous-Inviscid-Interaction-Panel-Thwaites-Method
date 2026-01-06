
import matplotlib.pyplot as plt
import numpy as np
import os
from panel_solver import PanelSolver

def verify_naca0009():
    """
    Runs the Panel Solver for NACA 0009 at 0 and 5 degrees AoA.
    Plots the Cp distribution.
    """
    print("Running verification for NACA 0009...")
    
    # Case 1: Alpha = 0
    solver0 = PanelSolver('0009', 0, n_panels=100)
    solver0.solve()
    res0 = solver0.get_surface_properties()
    
    # Case 2: Alpha = 5
    solver5 = PanelSolver('0009', 5, n_panels=100)
    solver5.solve()
    res5 = solver5.get_surface_properties()
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Alpha 0
    plt.plot(res0['x'], res0['Cp'], label='NACA 0009, Alpha=0deg', linestyle='-')
    
    # Alpha 5
    plt.plot(res5['x'], res5['Cp'], label='NACA 0009, Alpha=5deg', linestyle='--')
    
    plt.gca().invert_yaxis()  # Standard aerodynamic plot: negative Cp up
    plt.title('Cp Distribution for NACA 0009 (Viscous-Inviscid Project)')
    plt.xlabel('x/c')
    plt.ylabel('Cp')
    plt.grid(True)
    plt.legend()
    
    output_path = 'verification_cp_naca0009.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Check Kutta condition indirectly (Cp should be equal at TE)
    print(f"Alpha 0 TE Cp difference: {abs(res0['Cp'][0] - res0['Cp'][-1]):.4f}")
    print(f"Alpha 5 TE Cp difference: {abs(res5['Cp'][0] - res5['Cp'][-1]):.4f}")
    
    # Calculate Cl from Cp integration
    # Cl = - integral(Cp * dx)
    # The integration path is along the surface.
    # We can simple sum -Cp * dx for each panel.
    # dx = x_end - x_start
    
    # Calculate dx for each panel
    dx = np.diff(solver5.x)
    
    # Cp at panels
    cp = res5['Cp']
    
    # Cl integration
    # Note: verify correct sign convention.
    # Lift is Force_y / (q c).
    # dFy = -Cp * dy_projection = -Cp * dx
    # So Cl = sum( -Cp_i * dx_i )
    
    cl_pressure = -np.sum(cp * dx)
    
    print(f"Alpha 5 Calculated Cl (Pressure Integral): {cl_pressure:.4f}")
    
    perimeter = np.sum(np.sqrt(np.diff(solver5.x)**2 + np.diff(solver5.y)**2))
    gamma_strength = solver5.gamma
    print(f"Solver Gamma parameter: {gamma_strength:.4f}")

if __name__ == "__main__":
    verify_naca0009()
