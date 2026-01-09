import numpy as np
from scipy import integrate

class ThwaitesSolver:
    def __init__(self, min_sep_x=0.05):
        """
        Initialize Thwaites solver.

        Parameters:
        -----------
        min_sep_x : float
            Minimum x/c location for valid separation detection.
            Separation detected before this is considered numerical artifact.
            Default 0.05 (5% chord) to avoid false LE separation.
        """
        self.min_sep_x = min_sep_x

    def calculate_properties(self, s, Ue, nu, rho, x=None):
        """
        Calculate boundary layer properties using Thwaites method.

        Parameters:
        -----------
        s : array
            Arc length along surface
        Ue : array
            Edge velocity
        nu : float
            Kinematic viscosity
        rho : float
            Density
        x : array, optional
            x-coordinates for separation threshold check
        """
        s, Ue = np.asarray(s), np.asarray(Ue)
        if x is not None:
            x = np.asarray(x)

        # Thwaites: theta^2 = (0.45 * nu / Ue^6) * integral(Ue^5 dx)
        int_Ue5 = integrate.cumulative_trapezoid(Ue**5, s, initial=0)
        theta2 = np.zeros_like(Ue)

        mask = Ue > 1e-9
        theta2[mask] = (0.45 * nu * int_Ue5[mask]) / (Ue[mask]**6)

        # Stagnation point (Hiemenz flow limit)
        if Ue[0] < 1e-9 and len(s) > 1 and s[1] > 0:
            k = (Ue[1] - Ue[0]) / (s[1] - s[0])
            if k > 0:
                theta2[0] = 0.075 * nu / k

        theta = np.sqrt(theta2)
        dUe_ds = np.gradient(Ue, s, edge_order=2)
        lam = (theta2 / nu) * dUe_ds

        # Find separation point with minimum x threshold
        # Only consider separation valid if x > min_sep_x (avoid LE artifacts)
        sep_indices = np.where(lam <= -0.09)[0]

        if len(sep_indices) > 0 and x is not None:
            # Filter out separation points before min_sep_x
            valid_sep = [i for i in sep_indices if x[i] >= self.min_sep_x]
            sep_idx = valid_sep[0] if len(valid_sep) > 0 else len(lam)
        elif len(sep_indices) > 0:
            sep_idx = sep_indices[0]
        else:
            sep_idx = len(lam)

        # Compute properties
        S_lam = np.zeros_like(lam)
        H_lam = np.zeros_like(lam)
        Cf = np.zeros_like(Ue)

        # Pre-separation (Attached flow)
        l_att = lam[:sep_idx].copy()
        # Clamp lambda to valid range for correlations (-0.09 to avoid singularities)
        l_att = np.clip(l_att, -0.09, None)

        mask_pos = l_att >= 0
        mask_neg = l_att < 0

        # Positive lambda correlations (favorable pressure gradient)
        S_lam[:sep_idx][mask_pos] = (l_att[mask_pos] + 0.09)**0.62
        H_lam[:sep_idx][mask_pos] = 2.61 - 3.75*l_att[mask_pos] + 5.24*l_att[mask_pos]**2

        # Negative lambda correlations (adverse pressure gradient, valid for lambda >= -0.09)
        S_lam[:sep_idx][mask_neg] = (l_att[mask_neg] + 0.09)**0.62
        H_lam[:sep_idx][mask_neg] = 2.088 + 0.0731 / (l_att[mask_neg] + 0.14)

        # Post-separation handling
        if sep_idx < len(lam):
            # Freeze H at separation value, but cap at realistic maximum
            H_sep = min(H_lam[sep_idx-1], 3.5) if sep_idx > 0 else 2.5
            H_lam[sep_idx:] = H_sep
            S_lam[sep_idx:] = 0.0  # Zero skin friction

            # Gradual theta growth post-separation (reduced growth rate)
            if sep_idx > 1:
                dtheta = (theta[sep_idx-1] - theta[sep_idx-2]) * 0.5
                for i in range(sep_idx, len(lam)):
                    theta[i] = theta[i-1] + dtheta
            else:
                for i in range(sep_idx, len(lam)):
                    theta[i] = theta[sep_idx-1] if sep_idx > 0 else 0.001

        # Apply H limits everywhere (cap unphysical values)
        H_lam = np.clip(H_lam, 1.4, 4.0)

        delta_star = theta * H_lam

        valid = (Ue > 1e-9) & (theta > 1e-9)
        Cf[valid] = 2 * nu * S_lam[valid] / (Ue[valid] * theta[valid])

        return {
            'theta': theta, 'H': H_lam, 'delta_star': delta_star,
            'Cf': Cf, 'lambda': lam, 'S_lambda': S_lam,
            'separation_idx': sep_idx if sep_idx < len(lam) else None
        }
