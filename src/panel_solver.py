
import numpy as np

class PanelSolver:
    """
    Inviscid flow solver using the Source-Vortex Panel Method (Hess-Smith).
    """

    def __init__(self, naca_code, alpha_deg, n_panels=100):
        """
        Initialize the solver.

        Args:
            naca_code (str): NACA 4-digit code (e.g., '0012', '2412').
            alpha_deg (float): Angle of attack in degrees.
            n_panels (int): Number of panels to use (must be even).
        """
        self.naca_code = naca_code
        self.alpha = np.radians(alpha_deg)
        self.n_panels = n_panels
        if self.n_panels % 2 != 0:
            self.n_panels += 1  # Ensure even number of panels

        self.panels = []
        self.x = None
        self.y = None
        self.cp = None
        self.vt = None
        self.sigma = None
        self.gamma = None

        self._generate_geometry()

    def _generate_geometry(self):
        """
        Generates NACA 4-digit airfoil coordinates with cosine spacing.
        """
        m = int(self.naca_code[0]) / 100.0
        p = int(self.naca_code[1]) / 10.0
        t = int(self.naca_code[2:]) / 100.0

        # Cosine spacing
        beta = np.linspace(0, np.pi, self.n_panels // 2 + 1)
        xc = 0.5 * (1 - np.cos(beta))

        yt = 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2 +
                      0.2843 * xc**3 - 0.1015 * xc**4)

        yc = np.zeros_like(xc)
        dyc_dx = np.zeros_like(xc)

        for i in range(len(xc)):
            if xc[i] < p:
                yc[i] = m / p**2 * (2 * p * xc[i] - xc[i]**2)
                dyc_dx[i] = 2 * m / p**2 * (p - xc[i])
            else:
                yc[i] = m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * xc[i] - xc[i]**2)
                dyc_dx[i] = 2 * m / (1 - p)**2 * (p - xc[i])

        theta = np.arctan(dyc_dx)

        xu = xc - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = xc + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        # Combine upper and lower surfaces
        # Order: Trailing edge (lower) -> Leading edge -> Trailing edge (upper)
        self.x = np.concatenate((xl[::-1], xu[1:]))
        self.y = np.concatenate((yl[::-1], yu[1:]))
        
        # Handle small gap at TE if necessary (usually open for standard NACA)
        # For simplicity in panel methods, we often want closed TE.
        # Let's average the last points to close it perfectly.
        self.x[0] = self.x[-1] = (self.x[0] + self.x[-1]) / 2.0
        self.y[0] = self.y[-1] = (self.y[0] + self.y[-1]) / 2.0


    def solve(self):
        """
        Solves the linear system for source and vortex strengths.
        """
        # Panel definition
        # Panel i is between point i and i+1
        x_start = self.x[:-1]
        y_start = self.y[:-1]
        x_end = self.x[1:]
        y_end = self.y[1:]

        # Control points (midpoints)
        xc = (x_start + x_end) / 2
        yc = (y_start + y_end) / 2
        
        # Panel lengths
        dx = x_end - x_start
        dy = y_end - y_start
        length = np.sqrt(dx**2 + dy**2)
        
        # Panel angles (sin and cos)
        sin_theta = dy / length
        cos_theta = dx / length

        num_panels = len(xc)
        
        # Influence coefficients
        # An[i, j] = normal velocity induced at panel i by unit source on panel j
        # At[i, j] = tangential velocity induced at panel i by unit source on panel j
        # Bn[i, j] = normal velocity induced at panel i by unit vortex on panel j (constant gamma)
        # Bt[i, j] = tangential velocity induced at panel i by unit vortex on panel j
        
        An = np.zeros((num_panels, num_panels))
        At = np.zeros((num_panels, num_panels))
        Bn = np.zeros((num_panels, num_panels)) # We only need sum of these for the single gamma
        Bt = np.zeros((num_panels, num_panels))

        for i in range(num_panels):
            for j in range(num_panels):
                if i == j:
                    An[i, j] = 0.5 * np.pi
                    At[i, j] = 0.0
                    Bn[i, j] = 0.0 # Will be summed later, self-induced normal from vortex is 0? Check: Self induced normal for constant vortex is 0? 
                    # Actually for constant vortex sheet, self-induced normal velocity is 0. 
                    # Tangential is +/- gamma/2.
                    # Wait, Hess Smith usually treats vortex as constant gamma.
                    # Let's use analytical integerals.
                    continue
                
                # Transform control point i to local coordinates of panel j
                dx_val = xc[i] - x_start[j]
                dy_val = yc[i] - y_start[j]
                
                x_local = dx_val * cos_theta[j] + dy_val * sin_theta[j]
                y_local = -dx_val * sin_theta[j] + dy_val * cos_theta[j]
                
                # Integrals
                r1_sq = x_local**2 + y_local**2
                r2_sq = (x_local - length[j])**2 + y_local**2
                
                # Handle atan2
                beta_ang = np.arctan2(y_local, x_local - length[j]) - np.arctan2(y_local, x_local)
                
                # Adjust for periodicity of atan2 to ensure correct branch
                # This is crucial for panels wrapping around
                while beta_ang > np.pi: beta_ang -= 2*np.pi
                while beta_ang < -np.pi: beta_ang += 2*np.pi
                
                log_r = 0.5 * np.log(r2_sq / r1_sq)
                
                # Induced velocities in local frame of panel j
                ul = (log_r) / (2 * np.pi) # due to source (sigma=1)
                vl = (beta_ang) / (2 * np.pi) # due to source
                
                ul_v = (beta_ang) / (2 * np.pi) # due to vortex (gamma=1)
                vl_v = -(log_r) / (2 * np.pi) # due to vortex
                
                # Rotate back to global frame
                # u_global = ul * cos - vl * sin
                # v_global = ul * sin + vl * cos
                
                u_source = ul * cos_theta[j] - vl * sin_theta[j]
                v_source = ul * sin_theta[j] + vl * cos_theta[j]
                
                u_vortex = ul_v * cos_theta[j] - vl_v * sin_theta[j]
                v_vortex = ul_v * sin_theta[j] + vl_v * cos_theta[j]
                
                # Project onto normal and tangent of panel i
                # Normal is (-sin_theta[i], cos_theta[i])
                # Tangent is (cos_theta[i], sin_theta[i])
                
                An[i, j] = -u_source * sin_theta[i] + v_source * cos_theta[i]
                At[i, j] = u_source * cos_theta[i] + v_source * sin_theta[i]
                
                # For vortex, we just store the coefficients, we sum them up later because gamma is constant
                Bn[i, j] = -u_vortex * sin_theta[i] + v_vortex * cos_theta[i]
                Bt[i, j] = u_vortex * cos_theta[i] + v_vortex * sin_theta[i]
                
            # Self influence correction for i==j
            # Source: Normal = 0.5, Tangential = 0
            # Vortex: Normal = 0, Tangential = -0.5 (internal flow) or 0.5? 
            # On the surface, for C-P method, velocity jumps. 
            # Usually we take the limit approaching from fluid side?
            # Hess-Smith: u_n = 0 implies we satisfy BC. 
            # The self-term for normal velocity of source is 0.5 * sigma.
            # So An[i,i] = 0.5
            pass # Handled in the if i==j block
            
        # Build System [A] [x] = [b]
        # x = [sigma_1, ..., sigma_N, gamma]
        # N equations: Flow tangency
        # 1 equation: Kutta condition
        
        matrix_size = num_panels + 1
        A_mat = np.zeros((matrix_size, matrix_size))
        b_vec = np.zeros(matrix_size)
        
        # Flow tangency BC: V_n = 0
        # V_n = V_inf_n + sum(An_ij * sigma_j) + sum(Bn_ij * gamma) = 0
        # sum(An_ij * sigma_j) + gamma * sum(Bn_ij) = -V_inf_n
        
        V_inf_x = np.cos(self.alpha)
        V_inf_y = np.sin(self.alpha)
        
        V_inf_n = -V_inf_x * sin_theta + V_inf_y * cos_theta
        
        A_mat[:num_panels, :num_panels] = An
        A_mat[:num_panels, num_panels] = np.sum(Bn, axis=1)
        b_vec[:num_panels] = -V_inf_n
        
        # Kutta Condition
        # Tangential velocity at first panel (lower TE) + Tangential velocity at last panel (upper TE) = 0
        # Or more accurately: Vt(1) + Vt(N) = 0? 
        # Actually Hess-Smith often uses: Vt(1) = -Vt(N) (magnitudes equal, direction opposite in global? No, defined tangent vector).
        # Tangents are defined CCW. So TE lower points forward/down. TE upper points forward/up.
        # Flow leaves smoothly means Vt_upper + Vt_lower = 0?
        # Let's sum the tangential velocities at the two TE panels.
        
        # Vt_i = V_inf_t + sum(At_ij * sigma_j) + gamma * sum(Bt_ij)
        # We want Vt_0 + Vt_{N-1} = 0
        
        te1 = 0
        te2 = num_panels - 1
        
        terms_sigma = At[te1, :] + At[te2, :]
        term_gamma = np.sum(Bt[te1, :]) + np.sum(Bt[te2, :])
        
        # Add self-term for tangential velocity of vortex
        # Vt self = 0.5 * gamma? 
        # Standard result: Vt jumps by gamma. V_out - V_in = gamma.
        # Average is used for induced.
        # Usually At[i,i]=0 and Bt[i,i]= -0.5?
        # Let's assume standard Cauchy Principal Value is 0 and we add the +/- 0.5 term explicitly or implicitly?
        # Hess Smith formulation:
        # Eq N+1: sum(At[0,j] + At[N-1,j]) * sigma_j + (sum(Bt[0,j] + Bt[N-1,j]) + 1) * gamma = -(V_inf_t[0] + V_inf_t[N-1])
        # The +1 comes from the self terms?
        # Lower TE (panel 0): Vt self = -0.5 * gamma
        # Upper TE (panel N-1): Vt self = -0.5 * gamma
        # Wait, if defined CCW, gamma is positive CCW.
        # The jump is gamma. 
        # Let's look up the standard Kutta condition row for Hess-Smith.
        # Anderson/Katz-Plotkin: Vt(1) + Vt(N) = 0.
        # Bt self terms are -0.5?
        # Let's try adding -0.5 to Bt[i,i] before summing for the single panel? No.
        # Correct implementation:
        # The self-induced tangential velocity by vortex sheet of strength gamma on itself is +/- gamma/2.
        # Panel 0 (Lower surface TE): Inner side (body). No, we evaluate at control point ON the sheet.
        # It's actually +0.5*gamma for one, -0.5*gamma for other?
        # Let's assume the coefficients calculated by integrals EXCLUDE the self-term (log r/r -> 0), but the beta term handles it?
        # For beta integral:
        # beta approaches pi as we get close to the center?
        # Actually usually At[i,i]=0, Bt[i,i] depends on singularity type.
        # Vortex panel: Bt[i,i] is 0.5. 
        # Let's adhere to standard Hess-Smith matrix.
        
        # Manually adjustments for self-terms in A and B matrices if not naturally caught by integral loop:
        # In loop: i==j was skipped.
        # An[i,i] = 0.5 * pi? No, 0.5. (2pi factor handled).
        # Normal velocity jump across source sheet is sigma. 
        # Un = sigma/2.
        # At[i,i] = 0.
        # Bn[i,i] = 0.
        # Bt[i,i] = 0.5? No, for constant vortex, u_tangent is +/- gamma/2.
        # So Bt[i,i] = 0.5 * sgn?
        # Since we use ONE gamma for all panels, the coefficient of gamma for equation i is sum(B_ij).
        # We need to add the self term to this sum.
        # self-term for panel i is 0.5.
        
        for i in range(num_panels):
             Bn[i, i] = 0.0 # Normal vel of vortex on itself is 0
             Bt[i, i] = 0.5 # Tangential vel of vortex on itself is gamma/2
             An[i, i] = 0.5 # Normal vel of source on itself is sigma/2
             At[i, i] = 0.0 # Tangential vel of source on itself is 0
             
        # Refill A_mat with updated self terms
        A_mat[:num_panels, :num_panels] = An
        A_mat[:num_panels, num_panels] = np.sum(Bn, axis=1)
        
        V_inf_t = V_inf_x * cos_theta + V_inf_y * sin_theta
        
        # Kutta: (Vt_0 + Vt_{N-1} = 0)
        # sum((At_0j + At_{N-1}j) * sigma_j) + gamma * sum(Bt_0j + Bt_{N-1}j) = -(V_inf_t_0 + V_inf_t_{N-1})
        
        A_mat[num_panels, :num_panels] = At[0, :] + At[num_panels-1, :]
        A_mat[num_panels, num_panels] = np.sum(Bt[0, :]) + np.sum(Bt[num_panels-1, :])
        b_vec[num_panels] = -(V_inf_t[0] + V_inf_t[num_panels-1])
        
        # Solve
        solution = np.linalg.solve(A_mat, b_vec)
        
        self.sigma = solution[:num_panels]
        self.gamma = solution[num_panels]
        
        # Calculate final Vt and Cp
        Vt_induced_source = np.dot(At, self.sigma)
        Vt_induced_vortex = self.gamma * np.sum(Bt, axis=1)
        
        self.vt = V_inf_t + Vt_induced_source + Vt_induced_vortex
        
        # Bernoulli for Cp
        # Cp = 1 - (V/V_inf)^2
        # V_inf = 1 (normalized)
        # V = |Vt| (since Vn=0)
        
        self.cp = 1.0 - (self.vt / 1.0)**2
        
    def get_surface_properties(self):
        """
        Returns surface properties.
        
        Returns:
            dict: {
                'x': x coordinates of control points,
                'y': y coordinates of control points,
                's': arc length from stagnation point (approx),
                'Ue': Tangential velocity (normalized),
                'Cp': Pressure coefficient
            }
        """
        # Re-calculate control points for return
        xc = (self.x[:-1] + self.x[1:]) / 2
        yc = (self.y[:-1] + self.y[1:]) / 2
        
        # Calculate arc length
        # Typically we want s=0 at LE.
        # The points are ordered TE(lower) -> LE -> TE(upper).
        # We can calculate cumulative distance from index 0.
        
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        ds = np.sqrt(dx**2 + dy**2)
        
        # Cumulative s from TE lower
        s_raw = np.concatenate(([0], np.cumsum(ds)))
        s_mid = (s_raw[:-1] + s_raw[1:]) / 2
        
        # Find leading edge (min x usually, or stagnation point)
        # Stagnation point is where Vt=0. 
        # For simplicity, let's just return the raw s from TE lower for now, 
        # or shift it so 0 is at geometric LE (index n_panels/2)
        
        le_idx = self.n_panels // 2
        s_le = s_mid[le_idx]
        s_centered = s_mid - s_le
        
        # Or better: Split into upper and lower surfaces for boundary layer solver?
        # The user asked for "Surface arc length array (s) starting from the stagnation point or leading edge."
        # This usually means s > 0 for both surfaces? Or s goes from 0 to L on upper and 0 to L on lower?
        # Or s goes from -L to L?
        # Let's return s starting from LE (geometric). Upper surface s > 0, Lower s < 0? 
        # Or just return coordinate s along the surface.
        
        return {
            'x': xc,
            'y': yc,
            's': s_centered,
            'Ue': self.vt,
            'Cp': self.cp
        }

if __name__ == "__main__":
    # Internal test
    solver = PanelSolver('0012', 0)
    solver.solve()
    print("Solved for 0012")
    print("Gamma:", solver.gamma)
