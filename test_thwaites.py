import numpy as np
import matplotlib.pyplot as plt
from src.thwaites_solver import ThwaitesSolver

def test_blasius():
    print("Testing against Blasius (Flat Plate) Solution...")
    
    # Parameters
    U_inf = 10.0
    nu = 1.5e-5
    L = 1.0
    rho = 1.225
    
    # Grid
    x = np.linspace(0, L, 100)
    Ue = np.ones_like(x) * U_inf
    
    # Stagnation point hack: At x=0, boundary layer starts growing, 
    # but Ue is constant, so lambda should be 0 everywhere ideally.
    # However x starts at 0.
    
    solver = ThwaitesSolver()
    results = solver.calculate_properties(x, Ue, nu, rho)
    
    # Analytical Blasius
    # theta = 0.664 * sqrt(nu * x / U_inf)
    # Avoid div by zero
    theta_ana = np.zeros_like(x)
    mask = x > 0
    theta_ana[mask] = 0.664 * np.sqrt(nu * x[mask] / U_inf)
    
    # Compare
    error_theta = np.abs(results['theta'] - theta_ana)
    print(f"Max Error in Theta (excluding x=0): {np.max(error_theta[1:]):.6e}")
    
    # Check Shape Factor
    # Blasius H = 2.59
    print(f"Mean H (expected ~2.59): {np.mean(results['H'][1:]):.4f}")
    
    # Check Cf
    # Cf = 0.664 / sqrt(Rex)
    Rex = U_inf * x / nu
    Cf_ana = np.zeros_like(x)
    Cf_ana[mask] = 0.664 / np.sqrt(Rex[mask])
    
    print(f"Mean Cf error: {np.mean(np.abs(results['Cf'][1:] - Cf_ana[1:])):.6e}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(x, results['theta'], 'o', label='Thwaites')
    plt.plot(x, theta_ana, 'k-', label='Blasius')
    plt.title('Momentum Thickness')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(x, results['H'], 'o', label='Thwaites')
    plt.axhline(2.59, color='k', linestyle='--', label='Blasius (2.59)')
    plt.title('Shape Factor')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(x, results['Cf'], 'o', label='Thwaites')
    plt.plot(x, Cf_ana, 'k-', label='Blasius')
    plt.title('Skin Friction Cf')
    # plt.yscale('log')
    plt.ylim(0, 0.01)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(x, results['lambda'], label='Lambda')
    plt.title('Pressure Gradient (Lambda)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('test_blasius_results.png')
    print("Test plots saved to test_blasius_results.png")

if __name__ == "__main__":
    test_blasius()
