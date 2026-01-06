import numpy as np
from scipy import integrate

class ThwaitesSolver:
    """
    Computes laminar boundary layer properties using Thwaites' Integral Method.
    """

    def __init__(self):
        pass

    def calculate_properties(self, s, Ue, nu, rho):
        """
        Calculates boundary layer properties.

        Parameters:
        -----------
        s : array_like
            Surface distance (arc length) along the body [m].
        Ue : array_like
            Edge velocity (inviscid velocity) at locations s [m/s].
        nu : float
            Kinematic viscosity [m^2/s].
        rho : float
            Fluid density [kg/m^3] (used for skin friction scaling if needed).

        Returns:
        --------
        dict
            Dictionary containing arrays:
            - 'theta': Momentum thickness [m]
            - 'H': Shape factor
            - 'delta_star': Displacement thickness [m]
            - 'Cf': Skin friction coefficient
            - 'lambda': Thwaites' pressure gradient parameter
        """
        s = np.asarray(s)
        Ue = np.asarray(Ue)
        
        # Ensure Ue is positive for Ue^6 to work without issues, 
        # though Ue can be 0 at stagnation.
        
        # 1. Calculate theta^2 using Thwaites approximation
        # theta^2(x) = (0.45 * nu / Ue^6) * integral(Ue^5 dx)
        
        # Handle stagnation point Ue=0 case safely
        # We integrate Ue^5 with respect to s
        int_Ue5 = integrate.cumulative_trapezoid(Ue**5, s, initial=0)
        
        theta2 = np.zeros_like(Ue)
        
        # Avoid division by zero at stagnation point (assume index 0 is stagnation if Ue[0] is small)
        mask = Ue > 1e-9 # Threshold for division
        
        theta2[mask] = (0.45 * nu * int_Ue5[mask]) / (Ue[mask]**6)
        
        # Stagnation point limit (Hiemenz flow):
        # Near start, Ue ~ k*s. 
        # theta^2 = (0.45 * nu / (k*s)^6) * ( (k*s)^6 / (6*k) ) = 0.075 * nu / k 
        # dUe/ds = k. lambda = theta^2/nu * k = 0.075.
        # We can try to extrapolate or handle explicitly if s[0]==0.
        if Ue[0] < 1e-9:
             # Calculate derivative dUe/ds at 0 to get k
             if len(s) > 1 and s[1] > 0:
                 k = (Ue[1] - Ue[0]) / (s[1] - s[0])
                 if k > 0:
                     theta2[0] = 0.075 * nu / k
        
        theta = np.sqrt(theta2)
        
        # 2. Calculate dUe/ds
        # Use gradient
        dUe_ds = np.gradient(Ue, s, edge_order=2)
        
        # 3. Calculate Lambda
        # lambda = (theta^2 / nu) * dUe/ds
        lam = (theta2 / nu) * dUe_ds
        
        # 4. Correlations for S(lambda) and H(lambda)
        # Using Cebeci and Bradshaw (1977) or White formulations
        
        S_lam = np.zeros_like(lam)
        H_lam = np.zeros_like(lam)
        
        for i, l in enumerate(lam):
            # Check range
            if l < -0.09:
                # Separated
                S_lam[i] = 0.0
                # Shape factor H blows up or is undefined, but usually stays around 3.5-4.0 at separation
                # Standard correlation might not hold.
                H_lam[i] = 3.55 # Approximate value at separation
            elif l > 0.25:
                # Beyond correlation validity, clamp or extrapolate
                S_lam[i] = (l + 0.09)**0.62 # Continue power law
                H_lam[i] = 2.0 # Lower limit roughly
            else:
                # Standard Thwaites correlations
                # Shear function l(lambda) often denoted S(lambda) here
                S_lam[i] = (l + 0.09)**0.62 
                
                # Shape factor H(lambda)
                if l < 0:
                    H_lam[i] = 2.088 + 0.0731 / (l + 0.14)
                else:
                    H_lam[i] = 2.61 - 3.75*l + 5.24*l**2
                    
        # 5. Calculate Outputs
        delta_star = theta * H_lam
        
        # Skin friction coefficient Cf
        # Tw = mu * Ue / theta * S(lambda)
        # Cf = Tw / (0.5 * rho * Ue^2) = (rho * nu * Ue * S / theta) / (0.5 * rho * Ue^2)
        # Cf = 2 * nu * S / (Ue * theta)
        
        Cf = np.zeros_like(Ue)
        valid_Cf = (Ue > 1e-9) & (theta > 1e-9)
        Cf[valid_Cf] = 2 * nu * S_lam[valid_Cf] / (Ue[valid_Cf] * theta[valid_Cf])
        
        # Stagnation Cf is finite? 
        # Cf * sqrt(Re_x) is constant.
        # Ue ~ x. theta ~ const. S ~ const. Cf ~ 1/x -> Infinity at stagnation. 
        # Often we output Cf * sqrt(Re) or just avoid 0.
        
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
        """
        Checks for separation in the lambda array.
        Returns the index of separation if found, else None.
        Separation assumes lambda < -0.09.
        """
        sep_indices = np.where(lambda_arr <= -0.09)[0]
        if len(sep_indices) > 0:
            return sep_indices[0]
        return None
