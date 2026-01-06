import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from .panel_solver import PanelSolver
from .thwaites_solver import ThwaitesSolver


class VIISolver:
    """
    Iterative Viscous-Inviscid Interaction (VII) Solver.
    
    Couples Panel Method (inviscid) with Thwaites (viscous boundary layer)
    using the displacement thickness (delta*) transpiration model.
    
    The effective body geometry is updated iteratively:
        y_effective = y_base + delta_star (for upper surface)
        y_effective = y_base - delta_star (for lower surface)
    
    until convergence is achieved.
    """
    
    def __init__(self, naca_code, alpha_deg, n_panels=100, 
                 max_iter=20, tol=1e-5, relax=0.3):
        """
        Initialize the VII solver.
        
        Parameters
        ----------
        naca_code : str
            NACA 4-digit airfoil code (e.g., '0009', '2412')
        alpha_deg : float
            Angle of attack in degrees
        n_panels : int
            Number of panels (must be even)
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance (relative change in delta_star)
        relax : float
            Under-relaxation factor (0 < relax <= 1). 
            Lower values = more stable but slower convergence.
        """
        self.naca_code = naca_code
        self.alpha_deg = alpha_deg
        self.n_panels = n_panels if n_panels % 2 == 0 else n_panels + 1
        self.max_iter = max_iter
        self.tol = tol
        self.relax = relax
        
        # Initialize base panel solver to get original geometry
        self._base_panel = PanelSolver(naca_code, alpha_deg, n_panels)
        self.x_base = self._base_panel.x.copy()
        self.y_base = self._base_panel.y.copy()
        
        # Store node count
        self.n_nodes = len(self.x_base)
        self.n_half = self.n_nodes // 2
        
        # Results storage
        self.results = None
        self.history = None
        
    def solve(self, nu, rho, U_inf):
        """
        Run the iterative VII coupling solver.
        
        Parameters
        ----------
        nu : float
            Kinematic viscosity [m^2/s]
        rho : float
            Air density [kg/m^3]
        U_inf : float
            Freestream velocity [m/s]
            
        Returns
        -------
        dict
            Results dictionary containing:
            - 'converged': bool
            - 'iterations': int
            - 'Cl': Lift coefficient
            - 'Cd': Drag coefficient (viscous only)
            - 'upper': dict with upper surface BL properties
            - 'lower': dict with lower surface BL properties
            - 'history': dict with convergence history
        """
        # Initialize geometry displacement (delta_star at each node)
        # Node ordering: TE(lower) -> LE -> TE(upper)
        # Lower surface nodes: 0 to n_half (inclusive)
        # Upper surface nodes: n_half to n_nodes-1
        delta_star_nodes = np.zeros(self.n_nodes)
        
        # History tracking
        history = {
            'Cl': [],
            'Cd': [],
            'Cf_max_upper': [],
            'Cf_max_lower': [],
            'Tw_max_upper': [],
            'Tw_max_lower': [],
            'theta_te_upper': [],
            'theta_te_lower': [],
            'delta_star_te_upper': [],
            'delta_star_te_lower': [],
            'delta_star_residual': [],
        }
        
        converged = False
        iteration = 0
        Cl = 0.0
        Cd = 0.0
        bl_upper = None
        bl_lower = None
        upper_data = None
        lower_data = None
        Tw_upper = np.array([0.0])
        Tw_lower = np.array([0.0])
        
        for iteration in range(1, self.max_iter + 1):
            # Store old delta_star for convergence check
            delta_star_nodes_old = delta_star_nodes.copy()
            
            # --- Step 1: Compute effective geometry ---
            y_eff = self._apply_displacement(delta_star_nodes)
            
            # --- Step 2: Solve Panel Method on effective geometry ---
            panel = PanelSolver(self.naca_code, self.alpha_deg, self.n_panels)
            panel.set_geometry(self.x_base, y_eff)
            panel.solve()
            panel_res = panel.get_surface_properties()
            
            # --- Step 3: Extract upper and lower surfaces ---
            upper_data = self._extract_surface(panel_res, 'upper')
            lower_data = self._extract_surface(panel_res, 'lower')
            
            # --- Step 4: Run Thwaites on each surface ---
            thwaites = ThwaitesSolver()
            
            # Upper surface
            Ue_upper_dim = upper_data['Ue'] * U_inf
            bl_upper = thwaites.calculate_properties(
                upper_data['s'], Ue_upper_dim, nu, rho
            )
            
            # Lower surface  
            Ue_lower_dim = lower_data['Ue'] * U_inf
            bl_lower = thwaites.calculate_properties(
                lower_data['s'], Ue_lower_dim, nu, rho
            )
            
            # --- Step 5: Map delta_star back to nodes ---
            delta_star_nodes_new = self._map_delta_star_to_nodes(
                upper_data, lower_data, 
                bl_upper['delta_star'], bl_lower['delta_star']
            )
            
            # Under-relaxation
            delta_star_nodes = (1 - self.relax) * delta_star_nodes_old + self.relax * delta_star_nodes_new
            
            # --- Step 6: Calculate aerodynamic coefficients ---
            Cl = self._calculate_Cl(panel)
            Cd = self._calculate_Cd(bl_upper, bl_lower)
            
            # Calculate wall shear stress Tw = Cf * 0.5 * rho * Ue^2
            Tw_upper = bl_upper['Cf'] * 0.5 * rho * Ue_upper_dim**2
            Tw_lower = bl_lower['Cf'] * 0.5 * rho * Ue_lower_dim**2
            
            # --- Step 7: Record history ---
            history['Cl'].append(Cl)
            history['Cd'].append(Cd)
            history['Cf_max_upper'].append(np.max(bl_upper['Cf']))
            history['Cf_max_lower'].append(np.max(bl_lower['Cf']))
            history['Tw_max_upper'].append(np.max(Tw_upper))
            history['Tw_max_lower'].append(np.max(Tw_lower))
            history['theta_te_upper'].append(bl_upper['theta'][-1])
            history['theta_te_lower'].append(bl_lower['theta'][-1])
            history['delta_star_te_upper'].append(bl_upper['delta_star'][-1])
            history['delta_star_te_lower'].append(bl_lower['delta_star'][-1])
            
            # --- Step 8: Check convergence ---
            residual = self._compute_residual(delta_star_nodes_old, delta_star_nodes)
            history['delta_star_residual'].append(residual)
            
            if residual < self.tol:
                converged = True
                break
        
        # Store final results
        self.history = history
        self.results = {
            'converged': converged,
            'iterations': iteration,
            'Cl': Cl,
            'Cd': Cd,
            'upper': {
                'x': upper_data['x'],
                's': upper_data['s'],
                'Ue': upper_data['Ue'],
                'theta': bl_upper['theta'],
                'delta_star': bl_upper['delta_star'],
                'H': bl_upper['H'],
                'Cf': bl_upper['Cf'],
                'Tw': Tw_upper,
                'lambda': bl_upper['lambda'],
            },
            'lower': {
                'x': lower_data['x'],
                's': lower_data['s'],
                'Ue': lower_data['Ue'],
                'theta': bl_lower['theta'],
                'delta_star': bl_lower['delta_star'],
                'H': bl_lower['H'],
                'Cf': bl_lower['Cf'],
                'Tw': Tw_lower,
                'lambda': bl_lower['lambda'],
            },
            'history': history,
            'x_final': self.x_base,
            'y_final': self._apply_displacement(delta_star_nodes),
        }
        
        return self.results
    
    def _apply_displacement(self, delta_star_nodes):
        """
        Apply displacement thickness to base geometry.
        Upper surface: y += delta_star
        Lower surface: y -= delta_star
        """
        y_eff = self.y_base.copy()
        
        # Node ordering: TE(lower) -> LE -> TE(upper)
        # Lower surface: indices 0 to n_half (y decreases with delta_star)
        # Upper surface: indices n_half to n_nodes-1 (y increases with delta_star)
        
        for i in range(self.n_nodes):
            if i <= self.n_half:
                # Lower surface - displacement outward means y decreases
                y_eff[i] = self.y_base[i] - delta_star_nodes[i]
            else:
                # Upper surface - displacement outward means y increases
                y_eff[i] = self.y_base[i] + delta_star_nodes[i]
        
        return y_eff
    
    def _extract_surface(self, panel_res, surface):
        """
        Extract upper or lower surface data from panel results.
        
        Panel midpoints are organized: 
        - Indices 0 to N/2-1: Lower surface panels (TE -> LE direction)
        - Indices N/2 to N-1: Upper surface panels (LE -> TE direction)
        
        For boundary layer calculation, we need s starting from LE (stagnation).
        """
        x = panel_res['x']    # Panel midpoint x
        y = panel_res['y']    # Panel midpoint y
        Ue = panel_res['Ue']  # Tangential velocity at midpoints
        
        n_panels = len(x)
        split_idx = n_panels // 2
        
        if surface == 'upper':
            # Upper surface: indices split_idx to end, already LE -> TE
            node_indices = np.arange(self.n_half, self.n_nodes)
            x_surf = x[split_idx:]
            y_surf = y[split_idx:]
            ue_surf = Ue[split_idx:]
        else:
            # Lower surface: indices 0 to split_idx-1, need to reverse (TE -> LE to LE -> TE)
            node_indices = np.arange(self.n_half, -1, -1)  # n_half down to 0
            x_surf = x[:split_idx][::-1]
            y_surf = y[:split_idx][::-1]
            ue_surf = Ue[:split_idx][::-1]
        
        # Calculate arc length from first point
        dx = np.diff(x_surf)
        dy = np.diff(y_surf)
        ds = np.sqrt(dx**2 + dy**2)
        s_cumsum = np.concatenate(([0], np.cumsum(ds)))
        
        # Add stagnation point at s=0, Ue=0
        # First panel midpoint is at some distance from geometric LE
        dist_to_le = np.sqrt(x_surf[0]**2 + y_surf[0]**2)
        s_final = np.concatenate(([0.0], s_cumsum + dist_to_le))
        ue_final = np.concatenate(([0.0], np.abs(ue_surf)))
        x_final = np.concatenate(([0.0], x_surf))
        
        # Smooth Ue to remove panel method discretization noise
        if len(ue_final) > 15:
            ue_final = savgol_filter(ue_final, window_length=9, polyorder=3)
            ue_final[0] = 0.0
            ue_final = np.abs(ue_final)
        
        return {
            'x': x_final,
            's': s_final,
            'Ue': ue_final,
            'node_indices': node_indices,
        }
    
    def _map_delta_star_to_nodes(self, upper_data, lower_data, ds_upper, ds_lower):
        """
        Map delta_star from BL calculation (indexed by s) back to geometry nodes.
        Uses interpolation since BL points don't exactly match nodes.
        """
        delta_star_nodes = np.zeros(self.n_nodes)
        
        # Upper surface: interpolate ds_upper onto node x-coordinates
        # ds_upper[0] is at x=0 (stagnation), ds_upper[1:] at x_final[1:]
        upper_x = upper_data['x']  # includes stagnation at 0
        upper_indices = upper_data['node_indices']
        
        if len(upper_x) > 1 and len(ds_upper) > 1:
            # Create interpolator (s -> delta_star)
            # But we want to interpolate by x-coordinate
            interp_upper = interp1d(upper_x, ds_upper, kind='linear', 
                                   bounds_error=False, fill_value=(0, ds_upper[-1]))
            for node_idx in upper_indices:
                node_x = self.x_base[node_idx]
                delta_star_nodes[node_idx] = interp_upper(node_x)
        
        # Lower surface: same approach
        lower_x = lower_data['x']
        lower_indices = lower_data['node_indices']
        
        if len(lower_x) > 1 and len(ds_lower) > 1:
            interp_lower = interp1d(lower_x, ds_lower, kind='linear',
                                   bounds_error=False, fill_value=(0, ds_lower[-1]))
            for node_idx in lower_indices:
                node_x = self.x_base[node_idx]
                delta_star_nodes[node_idx] = interp_lower(node_x)
        
        return delta_star_nodes
    
    def _calculate_Cl(self, panel):
        """
        Calculate lift coefficient from panel method circulation.
        """
        # Kutta-Joukowski: Cl = 2 * Gamma / (U_inf * c)
        # Panel solver normalizes by U_inf=1 and c=1
        return 2 * panel.gamma
    
    def _calculate_Cd(self, bl_upper, bl_lower):
        """
        Calculate viscous drag coefficient from momentum thickness at TE.
        Using Squire-Young formula for laminar boundary layer.
        """
        theta_te_upper = bl_upper['theta'][-1]
        theta_te_lower = bl_lower['theta'][-1]
        
        # Chord = 1 (normalized)
        c = 1.0
        
        # Cd = 2 * (theta_upper + theta_lower) / c
        return 2 * (theta_te_upper + theta_te_lower) / c
    
    def _compute_residual(self, ds_old, ds_new):
        """
        Compute relative residual in delta_star.
        """
        norm_old = np.linalg.norm(ds_old)
        if norm_old < 1e-12:
            return np.linalg.norm(ds_new)
        
        return np.linalg.norm(ds_new - ds_old) / norm_old


def plot_convergence_history(history, save_path=None):
    """
    Plot convergence history of all tracked quantities.
    """
    import matplotlib.pyplot as plt
    
    iterations = np.arange(1, len(history['Cl']) + 1)
    
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle('VII Solver Convergence History', fontsize=14)
    
    # Cl
    ax = axs[0, 0]
    ax.plot(iterations, history['Cl'], 'b-o', markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$C_l$')
    ax.set_title('Lift Coefficient')
    ax.grid(True, alpha=0.3)
    
    # Cd
    ax = axs[0, 1]
    ax.plot(iterations, history['Cd'], 'r-o', markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$C_d$')
    ax.set_title('Drag Coefficient (Viscous)')
    ax.grid(True, alpha=0.3)
    
    # Cf max
    ax = axs[1, 0]
    ax.plot(iterations, history['Cf_max_upper'], 'b-o', markersize=4, label='Upper')
    ax.plot(iterations, history['Cf_max_lower'], 'r-s', markersize=4, label='Lower')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$C_f$ max')
    ax.set_title('Max Skin Friction Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Tw max
    ax = axs[1, 1]
    ax.plot(iterations, history['Tw_max_upper'], 'b-o', markersize=4, label='Upper')
    ax.plot(iterations, history['Tw_max_lower'], 'r-s', markersize=4, label='Lower')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$\tau_w$ max [Pa]')
    ax.set_title('Max Wall Shear Stress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Theta at TE
    ax = axs[2, 0]
    ax.plot(iterations, np.array(history['theta_te_upper']) * 1000, 'b-o', markersize=4, label='Upper')
    ax.plot(iterations, np.array(history['theta_te_lower']) * 1000, 'r-s', markersize=4, label='Lower')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$\theta_{TE}$ [mm]')
    ax.set_title('Momentum Thickness at TE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Delta star at TE
    ax = axs[2, 1]
    ax.plot(iterations, np.array(history['delta_star_te_upper']) * 1000, 'b-o', markersize=4, label='Upper')
    ax.plot(iterations, np.array(history['delta_star_te_lower']) * 1000, 'r-s', markersize=4, label='Lower')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$\delta^*_{TE}$ [mm]')
    ax.set_title('Displacement Thickness at TE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Convergence history saved to {save_path}")
    
    return fig


def plot_residual_history(history, save_path=None):
    """
    Plot residual convergence history.
    """
    import matplotlib.pyplot as plt
    
    iterations = np.arange(1, len(history['delta_star_residual']) + 1)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(iterations, history['delta_star_residual'], 'k-o', markersize=5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'Relative $\delta^*$ Change')
    ax.set_title('VII Solver Residual Convergence')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1e-5, color='r', linestyle='--', label='Tolerance (1e-5)')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Residual history saved to {save_path}")
    
    return fig
