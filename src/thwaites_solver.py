import numpy as np
from scipy import integrate

class ThwaitesSolver:
    def __init__(self):
        pass

    def calculate_properties(self, s, Ue, nu, rho):
        s, Ue = np.asarray(s), np.asarray(Ue)
        
        # Thwaites: theta^2 = (0.45 * nu / Ue^6) * integral(Ue^5 dx)
        int_Ue5 = integrate.cumulative_trapezoid(Ue**5, s, initial=0)
        theta2 = np.zeros_like(Ue)
        
        mask = Ue > 1e-9
        theta2[mask] = (0.45 * nu * int_Ue5[mask]) / (Ue[mask]**6)
        
        # Stagnation point (Hiemenz flow limit)
        if Ue[0] < 1e-9 and len(s) > 1 and s[1] > 0:
            k = (Ue[1] - Ue[0]) / (s[1] - s[0])
            if k > 0: theta2[0] = 0.075 * nu / k
        
        theta = np.sqrt(theta2)
        dUe_ds = np.gradient(Ue, s, edge_order=2)
        lam = (theta2 / nu) * dUe_ds
        
        # Correlations with separation freezing
        # 1. Find first separation point
        sep_indices = np.where(lam <= -0.09)[0]
        sep_idx = sep_indices[0] if len(sep_indices) > 0 else len(lam)
        
        # 2. Compute properties up to separation
        S_lam = np.zeros_like(lam)
        H_lam = np.zeros_like(lam)
        Cf = np.zeros_like(Ue)
        
        # Pre-separation (Attached flow)
        l_att = lam[:sep_idx]
        mask_pos = l_att >= 0
        mask_neg = l_att < 0
        
        # Positive lambda correlations
        S_lam[:sep_idx][mask_pos] = (l_att[mask_pos] + 0.09)**0.62
        H_lam[:sep_idx][mask_pos] = 2.61 - 3.75*l_att[mask_pos] + 5.24*l_att[mask_pos]**2
        
        # Negative lambda correlations (up to -0.09)
        S_lam[:sep_idx][mask_neg] = (l_att[mask_neg] + 0.09)**0.62
        H_lam[:sep_idx][mask_neg] = 2.088 + 0.0731 / (l_att[mask_neg] + 0.14)
        
        # Post-separation (Frozen/Extrapolated)
        if sep_idx < len(lam):
            # Freeze H and S at separation values
            H_lam[sep_idx:] = H_lam[sep_idx-1]
            S_lam[sep_idx:] = 0.0 # Zero shear
            
            # Linear extrapolation of theta (mild growth)
            dtheta = theta[sep_idx-1] - theta[sep_idx-2]
            for i in range(sep_idx, len(lam)):
                theta[i] = theta[i-1] + dtheta
        
        delta_star = theta * H_lam
        
        valid = (Ue > 1e-9) & (theta > 1e-9)
        Cf[valid] = 2 * nu * S_lam[valid] / (Ue[valid] * theta[valid])
        
        return {
            'theta': theta, 'H': H_lam, 'delta_star': delta_star,
            'Cf': Cf, 'lambda': lam, 'S_lambda': S_lam
        }
