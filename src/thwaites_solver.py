import numpy as np
from scipy import integrate

class ThwaitesSolver:
    # Computes laminar boundary layer properties using Thwaites' Integral Method.


    def __init__(self):
        pass

    def calculate_properties(self, s, Ue, nu, rho):
        # Compute laminar boundary layer properties using Thwaites' method
        # Inputs: s (surface distance), Ue (edge velocity), nu (kinematic viscosity), rho (density)
        # Outputs: theta (momentum thickness), H (shape factor), delta_star (displacement thickness), Cf (skin friction), lambda (Thwaites parameter)

        s = np.asarray(s)
        Ue = np.asarray(Ue)
        
        # 1. Calculate theta^2 using Thwaites approximation
        # theta^2(x) = (0.45 * nu / Ue^6) * integral(Ue^5 dx)
        
        int_Ue5 = integrate.cumulative_trapezoid(Ue**5, s, initial=0)
        theta2 = np.zeros_like(Ue)
        
        # Avoid division by zero at stagnation point
        mask = Ue > 1e-9
        theta2[mask] = (0.45 * nu * int_Ue5[mask]) / (Ue[mask]**6)
        
        # Stagnation point limit (Hiemenz flow):
        if Ue[0] < 1e-9:
             # Calculate derivative dUe/ds at 0 to get k
             if len(s) > 1 and s[1] > 0:
                 k = (Ue[1] - Ue[0]) / (s[1] - s[0])
                 if k > 0:
                     theta2[0] = 0.075 * nu / k
        
        theta = np.sqrt(theta2)
        
        # 2. Calculate dUe/ds
        dUe_ds = np.gradient(Ue, s, edge_order=2)
        
        # 3. Calculate Lambda
        lam = (theta2 / nu) * dUe_ds
        
        # 4. Correlations for S(lambda) and H(lambda)
        S_lam = np.zeros_like(lam)
        H_lam = np.zeros_like(lam)
        
        for i, l in enumerate(lam):
            # Check range
            if l < -0.09:
                # Separated
                S_lam[i] = 0.0
                H_lam[i] = 3.55 # Approximate value at separation
            elif l > 0.25:
                # Beyond correlation validity, clamp or extrapolate
                S_lam[i] = (l + 0.09)**0.62
                H_lam[i] = 2.0
            else:
                # Standard Thwaites correlations
                S_lam[i] = (l + 0.09)**0.62 
                
                # Shape factor H(lambda)
                if l < 0:
                    H_lam[i] = 2.088 + 0.0731 / (l + 0.14)
                else:
                    H_lam[i] = 2.61 - 3.75*l + 5.24*l**2
                    
        # 5. Calculate Outputs
        delta_star = theta * H_lam
        
        # Skin friction coefficient Cf
        # Cf = 2 * nu * S / (Ue * theta)
        
        Cf = np.zeros_like(Ue)
        valid_Cf = (Ue > 1e-9) & (theta > 1e-9)
        Cf[valid_Cf] = 2 * nu * S_lam[valid_Cf] / (Ue[valid_Cf] * theta[valid_Cf])
        
        results = {
            'theta': theta,
            'H': H_lam,
            'delta_star': delta_star,
            'Cf': Cf,
            'lambda': lam,
            'S_lambda': S_lam
        }
        
        return results

    def check_separation(self, lambda_arr):
#Checks for separation in the lambda array.
#Returns the index of separation if found, else None.
        sep_indices = np.where(lambda_arr <= -0.09)[0]
        if len(sep_indices) > 0:
            return sep_indices[0]
        return None
