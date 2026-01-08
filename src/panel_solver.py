import numpy as np

class PanelSolver:
    def __init__(self, naca_code, alpha_deg, n_panels=100):
        self.naca_code = naca_code
        self.alpha = np.radians(alpha_deg)
        self.n_panels = n_panels if n_panels % 2 == 0 else n_panels + 1
        self.x, self.y = None, None
        self.cp, self.vt = None, None
        self.sigma, self.gamma = None, None
        self._generate_geometry()

    def _generate_geometry(self):
        m = int(self.naca_code[0]) / 100.0
        p = int(self.naca_code[1]) / 10.0
        t = int(self.naca_code[2:]) / 100.0

        beta = np.linspace(0, np.pi, self.n_panels // 2 + 1)
        xc = 0.5 * (1 - np.cos(beta))
        yt = 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2 +
                      0.2843 * xc**3 - 0.1015 * xc**4)

        if p == 0:
            yc = np.zeros_like(xc)
            dyc_dx = np.zeros_like(xc)
        else:
            yc = np.where(xc < p,
                          m / p**2 * (2 * p * xc - xc**2),
                          m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * xc - xc**2))

            dyc_dx = np.where(xc < p,
                              2 * m / p**2 * (p - xc),
                              2 * m / (1 - p)**2 * (p - xc))

        theta = np.arctan(dyc_dx)
        xu = xc - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = xc + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        self.x = np.concatenate((xl[::-1], xu[1:]))
        self.y = np.concatenate((yl[::-1], yu[1:]))
        self.x[0] = self.x[-1] = (self.x[0] + self.x[-1]) / 2.0
        self.y[0] = self.y[-1] = (self.y[0] + self.y[-1]) / 2.0

    def set_geometry(self, x, y):
        self.x = np.asarray(x).copy()
        self.y = np.asarray(y).copy()
        self.x[0] = self.x[-1] = (self.x[0] + self.x[-1]) / 2.0
        self.y[0] = self.y[-1] = (self.y[0] + self.y[-1]) / 2.0

    def solve(self):
        x_start, y_start = self.x[:-1], self.y[:-1]
        x_end, y_end = self.x[1:], self.y[1:]

        xc = (x_start + x_end) / 2
        yc = (y_start + y_end) / 2

        dx, dy = x_end - x_start, y_end - y_start
        length = np.sqrt(dx**2 + dy**2)

        # Avoid division by zero for zero-length panels
        length[length < 1e-12] = 1e-12

        sin_theta, cos_theta = dy / length, dx / length

        num_panels = len(xc)
        An = np.zeros((num_panels, num_panels))
        At = np.zeros((num_panels, num_panels))
        Bn = np.zeros((num_panels, num_panels))
        Bt = np.zeros((num_panels, num_panels))

        for i in range(num_panels):
            for j in range(num_panels):
                if i == j:
                    An[i, j], At[i, j] = 0.5 * np.pi, 0.0
                    Bn[i, j], Bt[i, j] = 0.0, 0.5
                    continue

                dx_val, dy_val = xc[i] - x_start[j], yc[i] - y_start[j]
                x_local = dx_val * cos_theta[j] + dy_val * sin_theta[j]
                y_local = -dx_val * sin_theta[j] + dy_val * cos_theta[j]

                r1_sq = x_local**2 + y_local**2
                r2_sq = (x_local - length[j])**2 + y_local**2

                beta_ang = np.arctan2(y_local, x_local - length[j]) - np.arctan2(y_local, x_local)
                if beta_ang > np.pi: beta_ang -= 2 * np.pi
                elif beta_ang < -np.pi: beta_ang += 2 * np.pi

                log_r = 0.5 * np.log(r2_sq / r1_sq)
                ul, vl = log_r / (2 * np.pi), beta_ang / (2 * np.pi)
                ul_v, vl_v = beta_ang / (2 * np.pi), -log_r / (2 * np.pi)

                u_source = ul * cos_theta[j] - vl * sin_theta[j]
                v_source = ul * sin_theta[j] + vl * cos_theta[j]
                u_vortex = ul_v * cos_theta[j] - vl_v * sin_theta[j]
                v_vortex = ul_v * sin_theta[j] + vl_v * cos_theta[j]

                An[i, j] = -u_source * sin_theta[i] + v_source * cos_theta[i]
                At[i, j] = u_source * cos_theta[i] + v_source * sin_theta[i]
                Bn[i, j] = -u_vortex * sin_theta[i] + v_vortex * cos_theta[i]
                Bt[i, j] = u_vortex * cos_theta[i] + v_vortex * sin_theta[i]

            An[i, i] = 0.5
            At[i, i] = 0.0
            Bn[i, i] = 0.0
            Bt[i, i] = 0.5

        A_mat = np.zeros((num_panels + 1, num_panels + 1))
        b_vec = np.zeros(num_panels + 1)

        V_inf_x, V_inf_y = np.cos(self.alpha), np.sin(self.alpha)
        V_inf_n = -V_inf_x * sin_theta + V_inf_y * cos_theta

        A_mat[:num_panels, :num_panels] = An
        A_mat[:num_panels, num_panels] = np.sum(Bn, axis=1)
        b_vec[:num_panels] = -V_inf_n

        A_mat[num_panels, :num_panels] = At[0, :] + At[-1, :]
        A_mat[num_panels, num_panels] = np.sum(Bt[0, :]) + np.sum(Bt[-1, :])
        V_inf_t = V_inf_x * cos_theta + V_inf_y * sin_theta
        b_vec[num_panels] = -(V_inf_t[0] + V_inf_t[-1])

        solution = np.linalg.solve(A_mat, b_vec)
        self.sigma = solution[:num_panels]
        self.gamma = solution[num_panels]

        Vt_induced_source = np.dot(At, self.sigma)
        Vt_induced_vortex = self.gamma * np.sum(Bt, axis=1)
        self.vt = V_inf_t + Vt_induced_source + Vt_induced_vortex
        self.cp = 1.0 - (self.vt)**2

    def get_surface_properties(self):
        xc = (self.x[:-1] + self.x[1:]) / 2
        yc = (self.y[:-1] + self.y[1:]) / 2

        dx = np.diff(self.x)
        dy = np.diff(self.y)
        ds = np.sqrt(dx**2 + dy**2)
        s_raw = np.concatenate(([0], np.cumsum(ds)))
        s_mid = (s_raw[:-1] + s_raw[1:]) / 2

        le_idx = self.n_panels // 2
        s_le = s_mid[le_idx]
        s_centered = s_mid - s_le

        return {
            'x': xc, 'y': yc, 's': s_centered,
            'Ue': self.vt, 'Cp': self.cp
        }
