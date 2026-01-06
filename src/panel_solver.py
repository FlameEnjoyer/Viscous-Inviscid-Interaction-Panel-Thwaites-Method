import numpy as np

class PanelSolver:

    def __init__(self, naca_code, alpha_deg, n_panels=100):
        # Initialize the panel solver with NACA code, angle of attack, and number of panels
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
        # Generate airfoil geometry based on NACA 4-digit code
        m = int(self.naca_code[0]) / 100.0
        p = int(self.naca_code[1]) / 10.0
        t = int(self.naca_code[2:]) / 100.0

        # Cosine spacing
        beta = np.linspace(0, np.pi, self.n_panels // 2 + 1)
        xc = 0.5 * (1 - np.cos(beta))

        yt = 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1015 * xc**4)

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
        self.x = np.concatenate((xl[::-1], xu[1:]))
        self.y = np.concatenate((yl[::-1], yu[1:]))
        
        # Handle small gap at TE if necessary (usually open for standard NACA)
        # For simplicity in panel methods, we often want closed TE.
        self.x[0] = self.x[-1] = (self.x[0] + self.x[-1]) / 2.0
        self.y[0] = self.y[-1] = (self.y[0] + self.y[-1]) / 2.0

    def set_geometry(self, x, y):
        """
        Set airfoil geometry from external arrays.
        Used by VII solver to update geometry with displacement thickness.
        
        Parameters
        ----------
        x : array_like
            X-coordinates of panel nodes (TE_lower -> LE -> TE_upper)
        y : array_like
            Y-coordinates of panel nodes
        """
        self.x = np.asarray(x).copy()
        self.y = np.asarray(y).copy()
        
        # Close TE if needed
        self.x[0] = self.x[-1] = (self.x[0] + self.x[-1]) / 2.0
        self.y[0] = self.y[-1] = (self.y[0] + self.y[-1]) / 2.0

    def solve(self):

        # Solve the panel method to get surface velocities and pressure coefficients
        # Panel definition
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
        Bn = np.zeros((num_panels, num_panels))
        Bt = np.zeros((num_panels, num_panels))

        for i in range(num_panels):
            for j in range(num_panels):
                if i == j:
                    An[i, j] = 0.5 * np.pi
                    At[i, j] = 0.0
                    Bn[i, j] = 0.0 # Will be summed later, self-induced normal from vortex is 0? Check: Self induced normal for constant vortex is 0? 
                    # Actually for constant vortex sheet, self-induced normal velocity is 0. 
                    # Tangential is +/- gamma/2.
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
                
                while beta_ang > np.pi: beta_ang -= 2*np.pi
                while beta_ang < -np.pi: beta_ang += 2*np.pi
                
                log_r = 0.5 * np.log(r2_sq / r1_sq)
                
                # Induced velocities in local frame of panel j
                ul = (log_r) / (2 * np.pi) # due to source (sigma=1)
                vl = (beta_ang) / (2 * np.pi) # due to source
                
                ul_v = (beta_ang) / (2 * np.pi) # due to vortex (gamma=1)
                vl_v = -(log_r) / (2 * np.pi) # due to vortex
                
                # Rotate back to global frame  
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
            pass # Handled in the if i==j block
            
        # Build System [A] [x] = [b]
        # x = [sigma_1, ..., sigma_N, gamma]
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
        
        te1 = 0
        te2 = num_panels - 1
        
        terms_sigma = At[te1, :] + At[te2, :]
        term_gamma = np.sum(Bt[te1, :]) + np.sum(Bt[te2, :])

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
        # Re-calculate control points for return
        xc = (self.x[:-1] + self.x[1:]) / 2
        yc = (self.y[:-1] + self.y[1:]) / 2
        
        # Calculate arc length    
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        ds = np.sqrt(dx**2 + dy**2)
        
        # Cumulative s from TE lower
        s_raw = np.concatenate(([0], np.cumsum(ds)))
        s_mid = (s_raw[:-1] + s_raw[1:]) / 2
        
        # Stagnation point is where Vt=0. 
        # For simplicity, let's just return the raw s from TE lower for now
        
        le_idx = self.n_panels // 2
        s_le = s_mid[le_idx]
        s_centered = s_mid - s_le
        
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
