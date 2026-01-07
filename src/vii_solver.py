import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from .panel_solver import PanelSolver
from .thwaites_solver import ThwaitesSolver

class VIISolver:
    def __init__(self, naca_code, alpha_deg, n_panels=100, max_iter=20, tol=1e-5, relax=0.2):
        self.naca_code = naca_code
        self.alpha_deg = alpha_deg
        self.n_panels = n_panels if n_panels % 2 == 0 else n_panels + 1
        self.max_iter, self.tol, self.relax = max_iter, tol, relax
        
        self._base_panel = PanelSolver(naca_code, alpha_deg, n_panels)
        self.x_base, self.y_base = self._base_panel.x.copy(), self._base_panel.y.copy()
        self.n_nodes = len(self.x_base)
        self.n_half = self.n_nodes // 2
        self.results, self.history = None, None

    def solve(self, nu, rho, U_inf):
        delta_star_nodes = np.zeros(self.n_nodes)
        history = {k: [] for k in ['Cl', 'Cd', 'delta_star_residual']}
        
        converged = False
        iteration = 0
        
        for iteration in range(1, self.max_iter + 1):
            ds_old = delta_star_nodes.copy()
            y_eff = self._apply_displacement(delta_star_nodes)
            
            panel = PanelSolver(self.naca_code, self.alpha_deg, self.n_panels)
            panel.set_geometry(self.x_base, y_eff)
            panel.solve()
            res = panel.get_surface_properties()
            
            upper, lower = self._extract_surf(res, 'upper'), self._extract_surf(res, 'lower')
            thwaites = ThwaitesSolver()
            
            bl_up = thwaites.calculate_properties(upper['s'], upper['Ue']*U_inf, nu, rho)
            bl_low = thwaites.calculate_properties(lower['s'], lower['Ue']*U_inf, nu, rho)
            
            ds_new = self._map_ds_to_nodes(upper, lower, bl_up['delta_star'], bl_low['delta_star'])
            
            # Smoothing to prevent high-freq oscillations
            if len(ds_new) > 15:
                ds_new = savgol_filter(ds_new, window_length=11, polyorder=3)
            
            # Robustness: Clamp delta_star to reasonable range
            ds_new = np.clip(ds_new, 0.0, 0.1) # Max 10% chord displacement
            
            delta_star_nodes = (1 - self.relax) * ds_old + self.relax * ds_new
            
            # Calculate Cl from circulation integration
            # Cl = -2 * Gamma / (U_inf * c), Gamma = integral(Vt ds)
            dx = np.diff(panel.x)
            dy = np.diff(panel.y)
            ds = np.sqrt(dx**2 + dy**2)
            circulation = np.sum(panel.vt * ds)
            Cl = -2 * circulation
            
            Cd = 2 * (bl_up['theta'][-1] + bl_low['theta'][-1])
            residual = np.linalg.norm(delta_star_nodes - ds_old) / (np.linalg.norm(ds_old) + 1e-12)
            
            history['Cl'].append(Cl)
            history['Cd'].append(Cd)
            history['delta_star_residual'].append(residual)
            
            if residual < self.tol:
                converged = True
                break
        
        self.results = {
            'converged': converged, 'iterations': iteration,
            'Cl': Cl, 'Cd': Cd,
            'upper': {**upper, **bl_up},
            'lower': {**lower, **bl_low},
            'history': history,
            'x_final': self.x_base, 'y_final': y_eff
        }
        return self.results

    def _apply_displacement(self, ds):
        y_eff = self.y_base.copy()
        y_eff[:self.n_half+1] -= ds[:self.n_half+1] # Lower
        y_eff[self.n_half:] += ds[self.n_half:]     # Upper
        return y_eff

    def _extract_surf(self, res, side):
        x, y, Ue = res['x'], res['y'], res['Ue']
        split = len(x) // 2
        
        if side == 'upper':
            idx = np.arange(self.n_half, self.n_nodes)
            xs, ys, us = x[split:], y[split:], Ue[split:]
        else:
            idx = np.arange(self.n_half, -1, -1)
            xs, ys, us = x[:split][::-1], y[:split][::-1], Ue[:split][::-1]
            
        ds = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        s = np.concatenate(([0], np.cumsum(ds)))
        dist_le = np.sqrt(xs[0]**2 + ys[0]**2)
        
        s_final = np.concatenate(([0.0], s + dist_le))
        ue_final = np.concatenate(([0.0], np.abs(us)))
        x_final = np.concatenate(([0.0], xs))
        
        if len(ue_final) > 11:
            ue_final = savgol_filter(ue_final, 9, 3)
            ue_final[0] = 0.0
            
        return {'x': x_final, 's': s_final, 'Ue': np.abs(ue_final), 'indices': idx}

    def _map_ds_to_nodes(self, up, low, ds_up, ds_low):
        ds_nodes = np.zeros(self.n_nodes)
        
        for data, ds in [(up, ds_up), (low, ds_low)]:
            if len(data['x']) > 1:
                f = interp1d(data['x'], ds, bounds_error=False, fill_value=(0, ds[-1]))
                for idx in data['indices']:
                    ds_nodes[idx] = f(self.x_base[idx])
        return ds_nodes

def plot_convergence(hist, path=None):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(hist['Cl'], 'b-')
    ax1.set_title('Cl Convergence')
    ax1.grid(True, alpha=0.3)
    
    ax2.semilogy(hist['delta_star_residual'], 'k-')
    ax2.set_title('Residual')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if path: plt.savefig(path)
    return fig
