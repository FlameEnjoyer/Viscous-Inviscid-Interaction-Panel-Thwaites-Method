import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# GEOMETRY GENERATION
# ============================================================================
class NACA4Airfoil:
    """Generate NACA 4-digit airfoil coordinates."""

    def __init__(self, series, num_points=200, cosine_spacing=True, closed_te=True):
        self.series = series
        self.num_points = num_points
        self.cosine_spacing = cosine_spacing
        self.closed_te = closed_te

        self.m = int(series[0]) / 100
        self.p = int(series[1]) / 10
        self.t = int(series[2:4]) / 100

    def generate(self):
        if self.cosine_spacing:
            beta = np.linspace(0, np.pi, self.num_points + 1)
            x = (1 - np.cos(beta)) / 2
        else:
            x = np.linspace(0, 1, self.num_points + 1)

        yc = np.zeros_like(x)
        dycdx = np.zeros_like(x)
        yt = np.zeros_like(x)

        for i in range(len(x)):
            if x[i] < self.p:
                yc[i] = self.m / self.p**2 * (2 * self.p * x[i] - x[i] ** 2)
                dycdx[i] = 2 * self.m / self.p**2 * (self.p - x[i])
            else:
                yc[i] = self.m / (1 - self.p) ** 2 * ((1 - 2 * self.p) + 2 * self.p * x[i] - x[i] ** 2)
                dycdx[i] = 2 * self.m / (1 - self.p) ** 2 * (self.p - x[i])

            if self.closed_te:
                yt[i] = self.t / 0.2 * (
                    0.2969 * np.sqrt(x[i])
                    - 0.1260 * x[i]
                    - 0.3516 * x[i] ** 2
                    + 0.2843 * x[i] ** 3
                    - 0.1036 * x[i] ** 4
                )
            else:
                yt[i] = self.t / 0.2 * (
                    0.2969 * np.sqrt(x[i])
                    - 0.1260 * x[i]
                    - 0.3516 * x[i] ** 2
                    + 0.2843 * x[i] ** 3
                    - 0.1015 * x[i] ** 4
                )

        theta = np.arctan(dycdx)
        xU = x - yt * np.sin(theta)
        xL = x + yt * np.sin(theta)
        yU = yc + yt * np.cos(theta)
        yL = yc - yt * np.cos(theta)

        airfoil_x = np.concatenate([np.flip(xL), xU[1:]])
        airfoil_y = np.concatenate([np.flip(yL), yU[1:]])
        return airfoil_x, airfoil_y

# ============================================================================
# INVISCID FLOW SOLVER (VORTEX PANEL METHOD)
# ============================================================================
class VortexPanelMethod:
    """Vortex panel method for inviscid flow."""

    def __init__(self, XB, YB, AoA):
        self.XB = XB
        self.YB = YB
        self.AoA = AoA
        self.M = len(XB) - 1

    def solve(self):
        M = self.M
        X = np.zeros(M)
        Y = np.zeros(M)
        S = np.zeros(M)
        Phi = np.zeros(M)
        RHS = np.zeros(M)
        RHSC = np.zeros(M)

        for i in range(M):
            X[i] = 0.5 * (self.XB[i] + self.XB[i + 1])
            Y[i] = 0.5 * (self.YB[i] + self.YB[i + 1])
            S[i] = np.sqrt((self.XB[i + 1] - self.XB[i]) ** 2 + (self.YB[i + 1] - self.YB[i]) ** 2)
            Phi[i] = np.arctan2(self.YB[i + 1] - self.YB[i], self.XB[i + 1] - self.XB[i])
            RHS[i] = np.sin(Phi[i] - self.AoA)
            RHSC[i] = np.cos(Phi[i] - self.AoA)

        CN1, CN2, CT1, CT2 = self._compute_integrals(X, Y, S, Phi, M)
        AN, AT = self._build_influence_matrix(CN1, CN2, CT1, CT2, M)

        AN[M, 0] = 1.0
        AN[M, M] = 1.0
        for j in range(1, M):
            AN[M, j] = 0.0
        RHS_ext = np.append(RHS, 0.0)

        Gamma = np.linalg.solve(AN, RHS_ext)
        VpVinf = RHSC + AT @ Gamma
        CP = 1 - VpVinf**2
        return X, Y, Phi, S, Gamma, VpVinf, CP

    def _compute_integrals(self, X, Y, S, Phi, M):
        CN1 = np.zeros((M, M))
        CN2 = np.zeros((M, M))
        CT1 = np.zeros((M, M))
        CT2 = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                if i == j:
                    CN1[i, j] = -1.0
                    CN2[i, j] = 1.0
                    CT1[i, j] = 0.5 * np.pi
                    CT2[i, j] = 0.5 * np.pi
                else:
                    A = -(X[i] - self.XB[j]) * np.cos(Phi[j]) - (Y[i] - self.YB[j]) * np.sin(Phi[j])
                    B = (X[i] - self.XB[j]) ** 2 + (Y[i] - self.YB[j]) ** 2
                    C = np.sin(Phi[i] - Phi[j])
                    D = np.cos(Phi[i] - Phi[j])
                    E = (X[i] - self.XB[j]) * np.sin(Phi[j]) - (Y[i] - self.YB[j]) * np.cos(Phi[j])
                    F = np.log(1.0 + S[j] * (S[j] + 2 * A) / B)
                    G = np.arctan2(E * S[j], B + A * S[j])
                    P = (X[i] - self.XB[j]) * np.sin(Phi[i] - 2 * Phi[j]) + (Y[i] - self.YB[j]) * np.cos(Phi[i] - 2 * Phi[j])
                    Q = (X[i] - self.XB[j]) * np.cos(Phi[i] - 2 * Phi[j]) - (Y[i] - self.YB[j]) * np.sin(Phi[i] - 2 * Phi[j])
                    CN2[i, j] = D + 0.5 * Q * F / S[j] - (A * C + D * E) * G / S[j]
                    CN1[i, j] = 0.5 * D * F + C * G - CN2[i, j]
                    CT2[i, j] = C + 0.5 * P * F / S[j] + (A * D - C * E) * G / S[j]
                    CT1[i, j] = 0.5 * C * F - D * G - CT2[i, j]
        return CN1, CN2, CT1, CT2

    def _build_influence_matrix(self, CN1, CN2, CT1, CT2, M):
        AN = np.zeros((M + 1, M + 1))
        AT = np.zeros((M, M + 1))
        for i in range(M):
            AN[i, 0] = CN1[i, 0]
            AN[i, M] = CN2[i, M - 1]
            AT[i, 0] = CT1[i, 0]
            AT[i, M] = CT2[i, M - 1]
            for j in range(1, M):
                AN[i, j] = CN1[i, j] + CN2[i, j - 1]
                AT[i, j] = CT1[i, j] + CT2[i, j - 1]
        return AN, AT

# ============================================================================
# FORCE INTEGRATION
# ============================================================================

def cl_from_cp(X, Cp):
    M = len(X)
    half = M // 2
    Cpl = np.flip(Cp[:half])
    Cpu = Cp[half:]
    dCp = Cpl - Cpu
    dX = X[half:]
    return np.trapezoid(dCp, dX)


def integrate_forces(X, XB, YB, CP, tau_w, delta, AoA, rho, Vinf):
    M = len(X)
    half = M // 2
    dX = X[half:]

    XB_L = np.flip(XB[: half + 1])
    YB_L = -np.flip(YB[: half + 1])
    XB_U = XB[half : M + 1]
    YB_U = YB[half : M + 1]

    dydx_L = np.diff(YB_L) / np.diff(XB_L)
    dydx_U = np.diff(YB_U) / np.diff(XB_U)

    cf = tau_w / (0.5 * rho * Vinf**2)
    cfl = np.flip(cf[:half])
    cfu = cf[half:]

    Cpl = np.flip(CP[:half])
    Cpu = CP[half:]

    dCp = Cpl - Cpu
    dCpdydx = Cpu * dydx_U - Cpl * dydx_L

    new_dydx_L = dydx_L
    new_dydx_U = dydx_U
    dcfdydx = cfu * new_dydx_U + cfl * new_dydx_L
    dcf = cfu + cfl

    cn = np.trapezoid(dCp, dX) + np.trapezoid(dcfdydx, dX)
    ca = np.trapezoid(dCpdydx, dX) + np.trapezoid(dcf, dX)

    cl = cn * np.cos(AoA) - ca * np.sin(AoA)
    cd = cn * np.sin(AoA) + ca * np.cos(AoA)
    return cl, cd

# ============================================================================
# BOUNDARY LAYER SOLVER
# ============================================================================

def thwaites_laminar(Ue, S, nu):
    """March Thwaites laminar boundary layer solver."""
    N = len(Ue)
    theta = np.zeros(N)
    H = np.zeros(N)
    Cf = np.zeros(N)
    tau_w = np.zeros(N)
    delta = np.zeros(N)
    delta_star = np.zeros(N)
    momentum_thick = np.zeros(N)

    # Accumulate integral I = \int Ue^5 ds
    I = 0.0
    eps = 1e-9
    for i in range(N):
        if i == 0:
            ds = max(S[1] - S[0], eps) if N > 1 else 1.0
            U_mid = Ue[0]
        else:
            ds = max(S[i] - S[i - 1], eps)
            U_mid = 0.5 * (Ue[i] + Ue[i - 1])
            I += U_mid**5 * ds

        Uloc = max(Ue[i], eps)
        theta_sq = 0.45 * nu * I / (Uloc**6 + eps)
        theta[i] = np.sqrt(max(theta_sq, 0.0))

        dUds = (Ue[i] - Ue[i - 1]) / ds if i > 0 else 0.0
        lam = (theta[i] ** 2 / nu) * dUds

        # Shape factor correlation (typical Thwaites fit)
        H[i] = 2.088 + 0.0731 / (lam + 0.14)

        # Skin friction coefficient correlation; force Cf -> 0 near laminar separation (~lam <= -0.09)
        F_lam = np.clip(1.0 + 11.5 * lam, 0.0, 2.0)
        Re_theta = max(Uloc * theta[i] / nu, eps)
        Cf[i] = 0.45 / Re_theta * F_lam
        Cf[i] = min(Cf[i], 0.01)

        # Derived quantities
        delta_star[i] = H[i] * theta[i]
        momentum_thick[i] = theta[i]
        delta[i] = 7.0 * theta[i]  
        tau_w[i] = 0.5 * Cf[i] * Uloc**2  

    return theta, H, Cf, tau_w, delta, delta_star, momentum_thick

# ============================================================================
# VISCOUS-INVISCID INTERACTION
# ============================================================================

def thwaites_single_sweep(XB, YB, Np, nu, rho, Vinf, AoA):
    """Single-sweep coupling of inviscid panels and Thwaites laminar BL."""
    M = 2 * Np
    solver = VortexPanelMethod(XB, YB, AoA)
    X, Y, Phi, DeltaS, _, VpVinf, CP = solver.solve()
    U = VpVinf * Vinf

    # Build curvilinear S 
    S = np.zeros(M)
    S[Np - 1] = -DeltaS[Np - 1] / 2.0
    for i in range(Np - 2, -1, -1):
        S[i] = S[i + 1] - 0.5 * (DeltaS[i] + DeltaS[i + 1])
    S[Np] = DeltaS[Np] / 2.0
    for i in range(Np + 1, M):
        S[i] = S[i - 1] + 0.5 * (DeltaS[i] + DeltaS[i - 1])

    # Stagnation reference and split sides
    lowest_idx = np.argmin(np.abs(U))
    if 0 < lowest_idx < M - 1 and U[lowest_idx] * U[lowest_idx + 1] < 0:
        stagnationS = S[lowest_idx] + (-U[lowest_idx]) / (U[lowest_idx + 1] - U[lowest_idx]) * (S[lowest_idx + 1] - S[lowest_idx])
    elif 0 < lowest_idx and U[lowest_idx] * U[lowest_idx - 1] < 0:
        stagnationS = S[lowest_idx] + (-U[lowest_idx]) / (U[lowest_idx - 1] - U[lowest_idx]) * (S[lowest_idx - 1] - S[lowest_idx])
    else:
        stagnationS = S[lowest_idx]

    subS = S - stagnationS
    upper_mask = subS >= 0
    lower_mask = ~upper_mask

    upperS = subS[upper_mask]
    lowerS = np.flip(np.abs(subS[lower_mask]))
    upperU = U[upper_mask]
    lowerU = np.flip(np.abs(U[lower_mask]))

    # March Thwaites on each side
    up_theta, up_H, up_Cf, up_tau_w, up_delta, up_delta_star, up_theta_m = thwaites_laminar(upperU, upperS, nu)
    lo_theta, lo_H, lo_Cf, lo_tau_w, lo_delta, lo_delta_star, lo_theta_m = thwaites_laminar(lowerU, lowerS, nu)

    # Recombine in original order
    delta = np.concatenate([np.flip(lo_delta), up_delta])
    tau_w_local = np.concatenate([np.flip(lo_tau_w), up_tau_w])
    disp_thick = np.concatenate([np.flip(lo_delta_star), up_delta_star])
    momentum_thick = np.concatenate([np.flip(lo_theta_m), up_theta_m])

    bl_thick = delta

    # Offset geometry by bl_thick normal to surface 
    n = np.zeros((2, M))
    for i in range(M):
        n[:, i] = [-(YB[i + 1] - YB[i]), (XB[i + 1] - XB[i])]
        n[:, i] = n[:, i] / np.linalg.norm(n[:, i])

    DeltaBL = n * bl_thick
    X_BL = X + DeltaBL[0, :]
    Y_BL = Y + DeltaBL[1, :]

    XB_new = np.zeros(M + 1)
    YB_new = np.zeros(M + 1)
    for i in range(1, M):
        XB_new[i] = 0.5 * (X_BL[i] + X_BL[i - 1])
        YB_new[i] = 0.5 * (Y_BL[i] + Y_BL[i - 1])
    XB_new[0] = 2 * X_BL[0] - XB_new[1]
    YB_new[0] = 2 * Y_BL[0] - YB_new[1]
    XB_new[M] = 2 * X_BL[M - 1] - XB_new[M - 1]
    YB_new[M] = 2 * Y_BL[M - 1] - YB_new[M - 1]

    solver_new = VortexPanelMethod(XB_new, YB_new, AoA)
    X_new, Y_new, Phi_new, DeltaS_new, _, VpVinf_new, CP_new = solver_new.solve()

    return (
        X_BL,
        Y_BL,
        VpVinf_new,
        CP_new,
        bl_thick,
        disp_thick,
        momentum_thick,
        tau_w_local,
        bl_thick,
        VpVinf_new * Vinf,
        XB_new,
        YB_new,
    )


def thwaites_iterative_vii(XB, YB, Np, nu, rho, Vinf, AoA, max_iter=20, tol=1e-4, relax=0.3, verbose=True):
    """Iterative VII coupling of inviscid panels and Thwaites laminar BL."""
    M = 2 * Np
    
    def smooth_field(arr, n_passes=2):
        """Apply simple moving average smoothing."""
        result = arr.copy()
        for _ in range(n_passes):
            smoothed = result.copy()
            for i in range(1, len(result) - 1):
                smoothed[i] = 0.25 * result[i-1] + 0.5 * result[i] + 0.25 * result[i+1]
            result = smoothed
        return result
    
    # Inviscid solution on original geometry (solve once)
    solver = VortexPanelMethod(XB, YB, AoA)
    X0, Y0, Phi, DeltaS, _, VpVinf_inv, CP_inv = solver.solve()
    Ue_inv = VpVinf_inv * Vinf  # Inviscid edge velocity
    
    # Compute panel normals (for final BL plot)
    n = np.zeros((2, M))
    for i in range(M):
        n[:, i] = [-(YB[i + 1] - YB[i]), (XB[i + 1] - XB[i])]
        norm = np.linalg.norm(n[:, i])
        if norm > 1e-12:
            n[:, i] = n[:, i] / norm

    # Build curvilinear coordinate
    S = np.zeros(M)
    S[Np - 1] = -DeltaS[Np - 1] / 2.0
    for i in range(Np - 2, -1, -1):
        S[i] = S[i + 1] - 0.5 * (DeltaS[i] + DeltaS[i + 1])
    S[Np] = DeltaS[Np] / 2.0
    for i in range(Np + 1, M):
        S[i] = S[i - 1] + 0.5 * (DeltaS[i] + DeltaS[i - 1])

    # Initialize velocity field
    Ue = Ue_inv.copy()
    Ue_old = Ue_inv.copy()
    disp_thick = np.zeros(M)
    disp_thick_old = np.zeros(M)
    
    convergence_history = []

    for iteration in range(max_iter):
        # Split upper/lower surfaces
        lowest_idx = np.argmin(np.abs(Ue))
        if 0 < lowest_idx < M - 1 and Ue[lowest_idx] * Ue[lowest_idx + 1] < 0:
            stagnationS = S[lowest_idx] + (-Ue[lowest_idx]) / (Ue[lowest_idx + 1] - Ue[lowest_idx]) * (S[lowest_idx + 1] - S[lowest_idx])
        elif 0 < lowest_idx and Ue[lowest_idx] * Ue[lowest_idx - 1] < 0:
            stagnationS = S[lowest_idx] + (-Ue[lowest_idx]) / (Ue[lowest_idx - 1] - Ue[lowest_idx]) * (S[lowest_idx - 1] - S[lowest_idx])
        else:
            stagnationS = S[lowest_idx]
        
        subS = S - stagnationS
        upper_mask = subS >= 0
        lower_mask = ~upper_mask
        
        upperS = subS[upper_mask]
        lowerS = np.flip(np.abs(subS[lower_mask]))
        upperU = np.abs(Ue[upper_mask])
        lowerU = np.flip(np.abs(Ue[lower_mask]))

        # Compute boundary layer properties
        up_theta, up_H, up_Cf, up_tau_w, up_delta, up_delta_star, up_theta_m = thwaites_laminar(upperU, upperS, nu)
        lo_theta, lo_H, lo_Cf, lo_tau_w, lo_delta, lo_delta_star, lo_theta_m = thwaites_laminar(lowerU, lowerS, nu)

        # Recombine results
        delta = np.concatenate([np.flip(lo_delta), up_delta])
        tau_w_local = np.concatenate([np.flip(lo_tau_w), up_tau_w])
        disp_thick_raw = np.concatenate([np.flip(lo_delta_star), up_delta_star])
        momentum_thick = np.concatenate([np.flip(lo_theta_m), up_theta_m])
        
        # Limit maximum displacement thickness to prevent runaway
        max_disp = 0.05  # 5% of chord
        disp_thick_raw = np.clip(disp_thick_raw, 0, max_disp)

        # Apply smoothing
        disp_thick_raw = smooth_field(disp_thick_raw, n_passes=3)

        # Apply under-relaxation
        disp_thick_new = relax * disp_thick_raw + (1 - relax) * disp_thick_old
        
        # Check convergence
        max_change = np.max(np.abs(disp_thick_new - disp_thick_old))
        convergence_history.append(max_change)
        
        if verbose:
            print(f"  Iteration {iteration + 1}: max delta* change = {max_change:.2e}")
        
        if iteration > 0 and max_change < tol:
            if verbose:
                print(f"  VII converged at iteration {iteration + 1}")
            disp_thick = disp_thick_new
            break
        
        disp_thick_old = disp_thick_new.copy()
        disp_thick = disp_thick_new

        # Velocity correction
        K = 0.3
        Ue_target = Ue_inv * (1 - K * disp_thick)

        # Relaxation on velocity
        Ue_new = 0.3 * Ue_target + 0.7 * Ue_old

        # Smooth velocity field
        Ue_new = smooth_field(Ue_new, n_passes=2)

        # Soft velocity limiting
        min_frac = 0.7
        Ue_min = min_frac * np.abs(Ue_inv)

        # Blend towards minimum when needed
        for i in range(M):
            if abs(Ue_inv[i]) > 1e-6:
                current_frac = abs(Ue_new[i]) / abs(Ue_inv[i])
                if current_frac < min_frac:
                    blend = 0.5 * (1 + current_frac / min_frac)
                    if Ue_inv[i] > 0:
                        Ue_new[i] = blend * Ue_new[i] + (1 - blend) * Ue_min[i]
                    else:
                        Ue_new[i] = blend * Ue_new[i] - (1 - blend) * Ue_min[i]
        
        Ue = Ue_new
        Ue_old = Ue.copy()

    else:
        # Check convergence status
        if len(convergence_history) >= 10:
            recent = convergence_history[-10:]
            if max(recent) < 0.05:
                if verbose:
                    print(f"  VII reached stable oscillation after {max_iter} iterations")
            else:
                if verbose:
                    print(f"  VII did not converge after {max_iter} iterations, max delta* change = {max_change:.2e}")

    # Final boundary layer computation
    Ue_final = smooth_field(Ue, n_passes=3)
    
    lowest_idx = np.argmin(np.abs(Ue_final))
    if 0 < lowest_idx < M - 1 and Ue_final[lowest_idx] * Ue_final[lowest_idx + 1] < 0:
        stagnationS = S[lowest_idx] + (-Ue_final[lowest_idx]) / (Ue_final[lowest_idx + 1] - Ue_final[lowest_idx]) * (S[lowest_idx + 1] - S[lowest_idx])
    elif 0 < lowest_idx and Ue_final[lowest_idx] * Ue_final[lowest_idx - 1] < 0:
        stagnationS = S[lowest_idx] + (-Ue_final[lowest_idx]) / (Ue_final[lowest_idx - 1] - Ue_final[lowest_idx]) * (S[lowest_idx - 1] - S[lowest_idx])
    else:
        stagnationS = S[lowest_idx]
    
    subS = S - stagnationS
    upper_mask = subS >= 0
    lower_mask = ~upper_mask
    
    upperS = subS[upper_mask]
    lowerS = np.flip(np.abs(subS[lower_mask]))
    upperU = np.abs(Ue_final[upper_mask])
    lowerU = np.flip(np.abs(Ue_final[lower_mask]))
    
    up_theta, up_H, up_Cf, up_tau_w, up_delta, up_delta_star, up_theta_m = thwaites_laminar(upperU, upperS, nu)
    lo_theta, lo_H, lo_Cf, lo_tau_w, lo_delta, lo_delta_star, lo_theta_m = thwaites_laminar(lowerU, lowerS, nu)
    
    delta = np.concatenate([np.flip(lo_delta), up_delta])
    tau_w_local = np.concatenate([np.flip(lo_tau_w), up_tau_w])
    disp_thick_final = np.concatenate([np.flip(lo_delta_star), up_delta_star])
    momentum_thick = np.concatenate([np.flip(lo_theta_m), up_theta_m])

    # Smooth final outputs
    delta = smooth_field(delta, n_passes=5)
    tau_w_local = smooth_field(tau_w_local, n_passes=5)
    disp_thick_final = smooth_field(disp_thick_final, n_passes=5)
    momentum_thick = smooth_field(momentum_thick, n_passes=5)

    # Select smoother displacement thickness
    te_region = slice(int(0.7 * M), M)
    var_iter = np.var(np.diff(disp_thick[te_region]))
    var_final = np.var(np.diff(disp_thick_final[te_region]))

    if var_iter < var_final:
        disp_thick_out = smooth_field(disp_thick, n_passes=5)
    else:
        disp_thick_out = disp_thick_final

    # Prepare output variables
    VpVinf = Ue_final / Vinf
    CP = 1 - VpVinf**2

    # Boundary layer thickness
    bl_thick = 7.0 * momentum_thick

    # Boundary layer surface
    DeltaBL = n * bl_thick
    X_BL = X0 + DeltaBL[0, :]
    Y_BL = Y0 + DeltaBL[1, :]

    # Offset boundary points
    XB_new = np.zeros(M + 1)
    YB_new = np.zeros(M + 1)
    for i in range(1, M):
        XB_new[i] = 0.5 * (X_BL[i] + X_BL[i - 1])
        YB_new[i] = 0.5 * (Y_BL[i] + Y_BL[i - 1])
    XB_new[0] = 2 * X_BL[0] - XB_new[1]
    YB_new[0] = 2 * Y_BL[0] - YB_new[1]
    XB_new[M] = 2 * X_BL[M - 1] - XB_new[M - 1]
    YB_new[M] = 2 * Y_BL[M - 1] - YB_new[M - 1]
    
    return (
        X_BL,
        Y_BL,
        VpVinf,
        CP,
        bl_thick,
        disp_thick_out,
        momentum_thick,
        tau_w_local,
        delta,
        Ue_final,
        XB_new,
        YB_new,
        convergence_history,
    )

# ============================================================================
# CASE EXECUTION
# ============================================================================

def run_case(naca_series="0009", Np=200, AoA_deg=0.0, Re=100000.0):
    miu = 18.37e-6
    rho = 1.225
    nu = miu / rho
    Vinf = Re * miu / rho
    AoA = np.deg2rad(AoA_deg)

    airfoil = NACA4Airfoil(naca_series, num_points=Np, cosine_spacing=True, closed_te=True)
    XB, YB = airfoil.generate()

    solver = VortexPanelMethod(XB, YB, AoA)
    X, Y, Phi, S, Gamma, VpVinf, CP = solver.solve()
    Cl_inviscid = cl_from_cp(X, CP)

    (
        XBL,
        YBL,
        VpVinf_bl,
        CP_bl,
        bl_thick,
        disp_thick,
        momentum_thick,
        tau_w,
        delta,
        U,
        new_XB,
        new_YB,
    ) = thwaites_single_sweep(XB, YB, Np, nu, rho, Vinf, AoA)

    cl, cd = integrate_forces(X, new_XB, new_YB, CP_bl, tau_w, delta, AoA, rho, Vinf)

    cf = tau_w / (0.5 * rho * Vinf**2)

    results = {
        "X": X,
        "VpVinf": VpVinf_bl,
        "CP": CP_bl,
        "Cl_inviscid": Cl_inviscid,
        "cl": cl,
        "cd": cd,
        "XB": XB,
        "YB": YB,
        "XBL": XBL,
        "YBL": YBL,
        "bl_thick": bl_thick,
        "disp_thick": disp_thick,
        "momentum_thick": momentum_thick,
        "cf": cf,
        "tau_w": tau_w,
    }
    return results


def run_case_iterative(naca_series="0009", Np=200, AoA_deg=0.0, Re=100000.0, max_iter=20, tol=1e-4, relax=0.5, verbose=True):
    miu = 18.37e-6
    rho = 1.225
    nu = miu / rho
    Vinf = Re * miu / rho
    AoA = np.deg2rad(AoA_deg)

    airfoil = NACA4Airfoil(naca_series, num_points=Np, cosine_spacing=True, closed_te=True)
    XB, YB = airfoil.generate()

    solver = VortexPanelMethod(XB, YB, AoA)
    X, Y, Phi, S, Gamma, VpVinf, CP = solver.solve()
    Cl_inviscid = cl_from_cp(X, CP)

    (
        XBL,
        YBL,
        VpVinf_bl,
        CP_bl,
        bl_thick,
        disp_thick,
        momentum_thick,
        tau_w,
        delta,
        U,
        new_XB,
        new_YB,
        convergence_history,
    ) = thwaites_iterative_vii(XB, YB, Np, nu, rho, Vinf, AoA, max_iter=max_iter, tol=tol, relax=relax, verbose=verbose)

    cl, cd = integrate_forces(X, new_XB, new_YB, CP_bl, tau_w, delta, AoA, rho, Vinf)

    cf = tau_w / (0.5 * rho * Vinf**2)

    results = {
        "X": X,
        "VpVinf": VpVinf_bl,
        "CP": CP_bl,
        "Cl_inviscid": Cl_inviscid,
        "cl": cl,
        "cd": cd,
        "XB": XB,
        "YB": YB,
        "XBL": XBL,
        "YBL": YBL,
        "bl_thick": bl_thick,
        "disp_thick": disp_thick,
        "momentum_thick": momentum_thick,
        "cf": cf,
        "tau_w": tau_w,
        "convergence_history": convergence_history,
    }
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(res, naca_series, AoA_deg, xfoil_data=None, output_dir=None):
    X = res["X"]
    CP = res["CP"]
    VpVinf = res["VpVinf"]
    XB = res["XB"]
    YB = res["YB"]
    XBL = res["XBL"]
    YBL = res["YBL"]
    disp = res["disp_thick"]
    mom = res["momentum_thick"]
    bl = res["bl_thick"]
    cf = res.get("cf")

    half = len(X) // 2

    # Determine plot configuration
    plot_both_surfaces = (AoA_deg != 0.0)

    # Upper surface data
    x_upper = X[half:]
    cp_upper = CP[half:]
    v_upper = np.abs(VpVinf[half:])
    disp_upper = disp[half:]
    mom_upper = mom[half:]
    bl_upper = bl[half:]
    cf_upper = cf[half:] if cf is not None else None

    # Sort by x coordinate
    upper_idx = np.argsort(x_upper)
    x_upper = x_upper[upper_idx]
    cp_upper = cp_upper[upper_idx]
    v_upper = v_upper[upper_idx]
    disp_upper = disp_upper[upper_idx]
    mom_upper = mom_upper[upper_idx]
    bl_upper = bl_upper[upper_idx]
    if cf_upper is not None:
        cf_upper = cf_upper[upper_idx]

    # Lower surface data
    x_lower = np.flip(X[:half])
    cp_lower = np.flip(CP[:half])
    v_lower = np.abs(np.flip(VpVinf[:half]))
    disp_lower = np.flip(disp[:half])
    mom_lower = np.flip(mom[:half])
    bl_lower = np.flip(bl[:half])
    cf_lower = np.flip(cf[:half]) if cf is not None else None

    # Sort lower surface
    lower_idx = np.argsort(x_lower)
    x_lower = x_lower[lower_idx]
    cp_lower = cp_lower[lower_idx]
    v_lower = v_lower[lower_idx]
    disp_lower = disp_lower[lower_idx]
    mom_lower = mom_lower[lower_idx]
    bl_lower = bl_lower[lower_idx]
    if cf_lower is not None:
        cf_lower = cf_lower[lower_idx]

    ref_upper = None
    ref_lower = None
    if xfoil_data and "x" in xfoil_data:
        y_arr = xfoil_data.get("y")
        if y_arr is not None:
            y_arr = np.array(y_arr)
            mask_upper = y_arr >= 0
            mask_lower = y_arr < 0
        else:
            mask_upper = np.zeros_like(xfoil_data["x"], dtype=bool)
            mask_upper[len(mask_upper) // 2 :] = True
            mask_lower = ~mask_upper

        # Reference upper surface
        ref_upper = {
            "x": np.array(xfoil_data["x"])[mask_upper],
            "cp": np.array(xfoil_data.get("cp", []))[mask_upper] if "cp" in xfoil_data else None,
            "v": np.abs(np.array(xfoil_data.get("v", []))[mask_upper]) if "v" in xfoil_data else None,
            "disp": np.array(xfoil_data.get("disp", []))[mask_upper] if "disp" in xfoil_data else None,
            "mom": np.array(xfoil_data.get("mom", []))[mask_upper] if "mom" in xfoil_data else None,
            "delta": np.array(xfoil_data.get("delta", []))[mask_upper] if "delta" in xfoil_data else None,
            "cf": np.array(xfoil_data.get("cf", []))[mask_upper] if "cf" in xfoil_data else None,
        }
        ref_upper_idx = np.argsort(ref_upper["x"])
        ref_upper["x"] = ref_upper["x"][ref_upper_idx]
        for key in ["cp", "v", "disp", "mom", "delta", "cf"]:
            if ref_upper.get(key) is not None:
                ref_upper[key] = ref_upper[key][ref_upper_idx]

        # Reference lower surface
        if plot_both_surfaces:
            ref_lower = {
                "x": np.array(xfoil_data["x"])[mask_lower],
                "cp": np.array(xfoil_data.get("cp", []))[mask_lower] if "cp" in xfoil_data else None,
                "v": np.abs(np.array(xfoil_data.get("v", []))[mask_lower]) if "v" in xfoil_data else None,
                "disp": np.array(xfoil_data.get("disp", []))[mask_lower] if "disp" in xfoil_data else None,
                "mom": np.array(xfoil_data.get("mom", []))[mask_lower] if "mom" in xfoil_data else None,
                "delta": np.array(xfoil_data.get("delta", []))[mask_lower] if "delta" in xfoil_data else None,
                "cf": np.array(xfoil_data.get("cf", []))[mask_lower] if "cf" in xfoil_data else None,
            }
            ref_lower_idx = np.argsort(ref_lower["x"])
            ref_lower["x"] = ref_lower["x"][ref_lower_idx]
            for key in ["cp", "v", "disp", "mom", "delta", "cf"]:
                if ref_lower.get(key) is not None:
                    ref_lower[key] = ref_lower[key][ref_lower_idx]

    # Backward compatibility
    ref = ref_upper

    figs = []

    fig = plt.figure()
    figs.append(fig)
    plt.plot(XB, YB, label="Thwaites airfoil", linestyle="-", color="#1f77b4")
    plt.plot(XBL, YBL, label="Thwaites BL surface", linestyle="-", color="#ff7f0e")
    plt.xlim(0, 1)
    plt.ylim(-0.15, 0.15)
    plt.title(f"NACA-{naca_series} Airfoil and Boundary Layer")
    plt.xlabel(r"$x/c$")
    plt.ylabel(r"$y/c$")
    plt.legend()
    plt.grid(True)

    # Plot: Velocity Distribution
    fig = plt.figure()
    figs.append(fig)
    plt.plot(x_upper, v_upper, label="Thwaites (upper)", linestyle="-", color="#1f77b4")
    if ref_upper and ref_upper.get("v") is not None:
        plt.plot(ref_upper["x"], ref_upper["v"], label="JavaFoil (upper)", linestyle="--", color="#1f77b4")
    if plot_both_surfaces:
        plt.plot(x_lower, v_lower, label="Thwaites (lower)", linestyle="-", color="#ff7f0e")
        if ref_lower and ref_lower.get("v") is not None:
            plt.plot(ref_lower["x"], ref_lower["v"], label="JavaFoil (lower)", linestyle="--", color="#ff7f0e")
    plt.title(r"Velocity Distribution, $V/V_\infty$")
    plt.xlabel(r"$x/c$")
    plt.ylabel(r"$|V/V_\infty|$")
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True)

    # Plot: Pressure Coefficient
    fig = plt.figure()
    figs.append(fig)
    plt.plot(x_upper, cp_upper, label="Thwaites (upper)", linestyle="-", color="#1f77b4")
    if ref_upper and ref_upper.get("cp") is not None:
        plt.plot(ref_upper["x"], ref_upper["cp"], label="JavaFoil (upper)", linestyle="--", color="#1f77b4")
    if plot_both_surfaces:
        plt.plot(x_lower, cp_lower, label="Thwaites (lower)", linestyle="-", color="#ff7f0e")
        if ref_lower and ref_lower.get("cp") is not None:
            plt.plot(ref_lower["x"], ref_lower["cp"], label="JavaFoil (lower)", linestyle="--", color="#ff7f0e")
    plt.gca().invert_yaxis()
    plt.title(r"Pressure Coefficient, $C_p$")
    plt.xlabel(r"$x/c$")
    plt.ylabel(r"$C_p$")
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True)

    # Plot: Boundary-Layer Thickness
    fig = plt.figure()
    figs.append(fig)
    if plot_both_surfaces:
        plt.plot(x_upper, bl_upper, label=r"$\delta$ Thwaites (upper)", linestyle="-", color="#1f77b4")
        if ref_upper and ref_upper.get("delta") is not None:
            plt.plot(ref_upper["x"], ref_upper["delta"], linestyle="--", color="#1f77b4", label=r"$\delta$ JavaFoil (upper)")
        plt.plot(x_lower, bl_lower, label=r"$\delta$ Thwaites (lower)", linestyle="-", color="#ff7f0e")
        if ref_lower and ref_lower.get("delta") is not None:
            plt.plot(ref_lower["x"], ref_lower["delta"], linestyle="--", color="#ff7f0e", label=r"$\delta$ JavaFoil (lower)")
        plt.title(r"Boundary-Layer Thickness, $\delta$")
    else:
        plt.plot(x_upper, bl_upper, label=r"$\delta$ (Thwaites)", linestyle="-", color="#1f77b4")
        if ref_upper and ref_upper.get("delta") is not None:
            plt.plot(ref_upper["x"], ref_upper["delta"], linestyle="--", color="#ff7f0e", label=r"$\delta$ (JavaFoil)")
        plt.title(r"Boundary-Layer Thickness, $\delta$ (upper)")
    plt.xlabel(r"$x/c$")
    plt.ylabel(r"$\delta$")
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True)

    # Plot: Displacement Thickness
    fig = plt.figure()
    figs.append(fig)
    if plot_both_surfaces:
        plt.plot(x_upper, disp_upper, label=r"$\delta^*$ Thwaites (upper)", linestyle="-", color="#1f77b4")
        if ref_upper and ref_upper.get("disp") is not None:
            plt.plot(ref_upper["x"], ref_upper["disp"], linestyle="--", color="#1f77b4", label=r"$\delta^*$ JavaFoil (upper)")
        plt.plot(x_lower, disp_lower, label=r"$\delta^*$ Thwaites (lower)", linestyle="-", color="#ff7f0e")
        if ref_lower and ref_lower.get("disp") is not None:
            plt.plot(ref_lower["x"], ref_lower["disp"], linestyle="--", color="#ff7f0e", label=r"$\delta^*$ JavaFoil (lower)")
        plt.title(r"Displacement Thickness, $\delta^*$")
    else:
        plt.plot(x_upper, disp_upper, label=r"$\delta^*$ (Thwaites)", linestyle="-", color="#1f77b4")
        if ref_upper and ref_upper.get("disp") is not None:
            plt.plot(ref_upper["x"], ref_upper["disp"], linestyle="--", color="#ff7f0e", label=r"$\delta^*$ (JavaFoil)")
        plt.title(r"Displacement Thickness, $\delta^*$ (upper)")
    plt.xlabel(r"$x/c$")
    plt.ylabel(r"$\delta^*$")
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True)

    # Plot: Momentum Thickness
    fig = plt.figure()
    figs.append(fig)
    if plot_both_surfaces:
        plt.plot(x_upper, mom_upper, label=r"$\theta$ Thwaites (upper)", linestyle="-", color="#1f77b4")
        if ref_upper and ref_upper.get("mom") is not None:
            plt.plot(ref_upper["x"], ref_upper["mom"], linestyle="--", color="#1f77b4", label=r"$\theta$ JavaFoil (upper)")
        plt.plot(x_lower, mom_lower, label=r"$\theta$ Thwaites (lower)", linestyle="-", color="#ff7f0e")
        if ref_lower and ref_lower.get("mom") is not None:
            plt.plot(ref_lower["x"], ref_lower["mom"], linestyle="--", color="#ff7f0e", label=r"$\theta$ JavaFoil (lower)")
        plt.title(r"Momentum Thickness, $\theta$")
    else:
        plt.plot(x_upper, mom_upper, label=r"$\theta$ (Thwaites)", linestyle="-", color="#1f77b4")
        if ref_upper and ref_upper.get("mom") is not None:
            plt.plot(ref_upper["x"], ref_upper["mom"], linestyle="--", color="#ff7f0e", label=r"$\theta$ (JavaFoil)")
        plt.title(r"Momentum Thickness, $\theta$ (upper)")
    plt.xlabel(r"$x/c$")
    plt.ylabel(r"$\theta$")
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True)

    # Plot: Skin-Friction Coefficient
    if cf_upper is not None:
        fig = plt.figure()
        figs.append(fig)
        if plot_both_surfaces:
            plt.plot(x_upper, cf_upper, label=r"$C_f$ Thwaites (upper)", linestyle="-", color="#1f77b4")
            if ref_upper and ref_upper.get("cf") is not None:
                plt.plot(ref_upper["x"], ref_upper["cf"], label=r"$C_f$ JavaFoil (upper)", linestyle="--", color="#1f77b4")
            if cf_lower is not None:
                plt.plot(x_lower, cf_lower, label=r"$C_f$ Thwaites (lower)", linestyle="-", color="#ff7f0e")
            if ref_lower and ref_lower.get("cf") is not None:
                plt.plot(ref_lower["x"], ref_lower["cf"], label=r"$C_f$ JavaFoil (lower)", linestyle="--", color="#ff7f0e")
            plt.title(r"Skin-Friction Coefficient, $C_f$")
        else:
            plt.plot(x_upper, cf_upper, label=r"$C_f$ (Thwaites)", linestyle="-", color="#1f77b4")
            if ref_upper and ref_upper.get("cf") is not None:
                plt.plot(ref_upper["x"], ref_upper["cf"], label=r"$C_f$ (JavaFoil)", linestyle="--", marker="s", markersize=3, markevery=10, color="#2ca02c")
            plt.title(r"Skin-Friction Coefficient, $C_f$ (upper)")
        plt.xlabel(r"$x/c$")
        plt.ylabel(r"$C_f$")
        plt.xlim(0, 1)
        plt.legend()
        plt.grid(True)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = ["geometry", "velocity", "cp", "delta", "delta_star", "theta", "cf"]
        for fig, name in zip(figs, names):
            fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    return figs

# ============================================================================
# DATA LOADING
# ============================================================================

def load_xfoil_cpwr(cp_path):
    xs, ys, cps = [], [], []
    with open(cp_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("naca") or line.lower().startswith("alfa"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            x, y, cp = map(float, parts[:3])
            xs.append(x)
            ys.append(y)
            cps.append(cp)
    xs = np.array(xs)
    ys = np.array(ys)
    cps = np.array(cps)
    v = np.sqrt(np.maximum(1.0 - cps, 0.0))
    return {"x": xs, "y": ys, "cp": cps, "v": v}


def load_javafoil_polar(path):
    aoa_list = []
    cl_list = []
    cd_list = []
    with open(path, "r") as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip header lines until we hit the data
            if line.startswith("[") or line.startswith("α"):
                in_data = True
                continue
            if not in_data:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    aoa = float(parts[0])
                    cl = float(parts[1])
                    cd = float(parts[2])
                    aoa_list.append(aoa)
                    cl_list.append(cl)
                    cd_list.append(cd)
                except ValueError:
                    continue
    return {"aoa": np.array(aoa_list), "cl": np.array(cl_list), "cd": np.array(cd_list)}


def load_javafoil_surface(path):
    xs, ys, vv, d1, d2, d3, cfs = [], [], [], [], [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if any(line.lower().startswith(prefix) for prefix in ["naca", "angle", "transition", "upper", "lower", "mach", "surface", "stall", "transition model", "re =", "x/c"]):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                x, y, v, delta1, delta2, delta3 = map(float, parts[:6])
                cf = float(parts[7])  # C_f is at index 7
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)
            vv.append(v)
            d1.append(delta1)
            d2.append(delta2)
            d3.append(delta3)
            cfs.append(cf)
    xs = np.array(xs)
    ys = np.array(ys)
    vv = np.array(vv)
    d1 = np.array(d1)
    d2 = np.array(d2)
    d3 = np.array(d3)
    cfs = np.array(cfs)
    cp = 1.0 - vv**2
    return {"x": xs, "y": ys, "v": vv, "cp": cp, "disp": d1, "mom": d2, "delta": d3, "cf": cfs}

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def run_aoa_sweep(naca_series="0009", Np=200, Re=100000.0, aoa_range=None):
    if aoa_range is None:
        aoa_range = np.arange(-4, 4.5, 0.5)
    
    cl_list = []
    cd_list = []
    aoa_list = []
    
    for aoa_deg in aoa_range:
        try:
            res = run_case_iterative(
                naca_series=naca_series,
                Np=Np,
                AoA_deg=aoa_deg,
                Re=Re,
                max_iter=50,
                tol=5e-3,
                relax=0.3,
                verbose=False
            )
            aoa_list.append(aoa_deg)
            cl_list.append(res["cl"])
            cd_list.append(res["cd"])
            print(f"  AoA = {aoa_deg:+.1f}°: Cl = {res['cl']:.4f}, Cd = {res['cd']:.5f}")
        except Exception as e:
            print(f"  AoA = {aoa_deg:+.1f}°: Failed ({e})")
            continue
    
    return {"aoa": np.array(aoa_list), "cl": np.array(cl_list), "cd": np.array(cd_list)}


def plot_polar(thwaites_polar, javafoil_polar=None, naca_series="0009", Re=100000, output_dir=None):
    figs = []

    # Plot: Lift coefficient
    fig = plt.figure()
    figs.append(fig)
    plt.plot(thwaites_polar["aoa"], thwaites_polar["cl"], 
             label="Thwaites Method", linestyle="-", color="#1f77b4", linewidth=1.5)
    if javafoil_polar is not None:
        plt.plot(javafoil_polar["aoa"], javafoil_polar["cl"], 
                 label="JavaFoil", linestyle="--", color="#1f77b4", linewidth=1.5)
    plt.title(f"Lift Coefficient ($c_l$)\nRe = {Re}\nNACA {naca_series}")
    plt.xlabel("AoA [°]")
    plt.ylabel(r"$c_l$")
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)

    # Plot: Drag coefficient
    fig = plt.figure()
    figs.append(fig)
    plt.plot(thwaites_polar["aoa"], thwaites_polar["cd"], 
             label="Thwaites Method", linestyle="-", color="#1f77b4", linewidth=1.5)
    if javafoil_polar is not None:
        plt.plot(javafoil_polar["aoa"], javafoil_polar["cd"], 
                 label="JavaFoil", linestyle="--", color="#1f77b4", linewidth=1.5)
    plt.title(f"Drag Coefficient ($c_d$)\nRe = {Re}\nNACA {naca_series}")
    plt.xlabel("AoA [°]")
    plt.ylabel(r"$c_d$")
    plt.legend()
    plt.grid(True)
    
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        figs[0].savefig(out / "cl_vs_aoa.png", dpi=200, bbox_inches="tight")
        figs[1].savefig(out / "cd_vs_aoa.png", dpi=200, bbox_inches="tight")
    
    return figs


def plot_combined_subplots(all_results, all_ref_data, aoa_list, naca_series, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Extract surface data
    def extract_surface_data(res, surface='upper'):
        X = res["X"]
        CP = res["CP"]
        VpVinf = res["VpVinf"]
        disp = res["disp_thick"]
        mom = res["momentum_thick"]
        bl = res["bl_thick"]
        cf = res.get("cf")
        
        half = len(X) // 2
        
        if surface == 'upper':
            x = X[half:]
            cp = CP[half:]
            v = np.abs(VpVinf[half:])
            disp_t = disp[half:]
            mom_t = mom[half:]
            bl_t = bl[half:]
            cf_t = cf[half:] if cf is not None else None
        else:
            x = np.flip(X[:half])
            cp = np.flip(CP[:half])
            v = np.abs(np.flip(VpVinf[:half]))
            disp_t = np.flip(disp[:half])
            mom_t = np.flip(mom[:half])
            bl_t = np.flip(bl[:half])
            cf_t = np.flip(cf[:half]) if cf is not None else None

        # Sort data
        idx = np.argsort(x)
        x = x[idx]
        cp = cp[idx]
        v = v[idx]
        disp_t = disp_t[idx]
        mom_t = mom_t[idx]
        bl_t = bl_t[idx]
        if cf_t is not None:
            cf_t = cf_t[idx]
        
        return {"x": x, "cp": cp, "v": v, "disp": disp_t, "mom": mom_t, "delta": bl_t, "cf": cf_t}
    
    def extract_ref_surface(ref_data, surface='upper'):
        if ref_data is None or "x" not in ref_data:
            return None
        
        y_arr = ref_data.get("y")
        if y_arr is not None:
            y_arr = np.array(y_arr)
            if surface == 'upper':
                mask = y_arr >= 0
            else:
                mask = y_arr < 0
        else:
            n = len(ref_data["x"])
            mask = np.zeros(n, dtype=bool)
            if surface == 'upper':
                mask[n // 2:] = True
            else:
                mask[:n // 2] = True
        
        ref = {
            "x": np.array(ref_data["x"])[mask],
            "cp": np.array(ref_data.get("cp", []))[mask] if "cp" in ref_data else None,
            "v": np.abs(np.array(ref_data.get("v", []))[mask]) if "v" in ref_data else None,
            "disp": np.array(ref_data.get("disp", []))[mask] if "disp" in ref_data else None,
            "mom": np.array(ref_data.get("mom", []))[mask] if "mom" in ref_data else None,
            "delta": np.array(ref_data.get("delta", []))[mask] if "delta" in ref_data else None,
            "cf": np.array(ref_data.get("cf", []))[mask] if "cf" in ref_data else None,
        }
        idx = np.argsort(ref["x"])
        ref["x"] = ref["x"][idx]
        for key in ["cp", "v", "disp", "mom", "delta", "cf"]:
            if ref.get(key) is not None:
                ref[key] = ref[key][idx]
        return ref

    # Variables configuration
    variables = [
        {"key": "v", "ylabel": r"$|V/V_\infty|$", "title": "Velocity Distribution", "filename": "velocity_combined", "invert_y": False},
        {"key": "cp", "ylabel": r"$C_p$", "title": "Pressure Coefficient", "filename": "cp_combined", "invert_y": True},
        {"key": "delta", "ylabel": r"$\delta$", "title": "Boundary-Layer Thickness", "filename": "delta_combined", "invert_y": False},
        {"key": "disp", "ylabel": r"$\delta^*$", "title": "Displacement Thickness", "filename": "delta_star_combined", "invert_y": False},
        {"key": "mom", "ylabel": r"$\theta$", "title": "Momentum Thickness", "filename": "theta_combined", "invert_y": False},
        {"key": "cf", "ylabel": r"$C_f$", "title": "Skin-Friction Coefficient", "filename": "cf_combined", "invert_y": False},
    ]

    # Generate plots
    for var in variables:
        fig = plt.figure(figsize=(14, 10))

        # Create subplot grid
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.25)

        # Subplot positions
        subplot_positions = [
            gs[0, 0],
            gs[0, 1],
            gs[1, 0],
            gs[1, 1]
        ]
        
        for i, aoa in enumerate(aoa_list[:4]):
            ax = fig.add_subplot(subplot_positions[i])
            
            res = all_results[aoa]
            ref_data = all_ref_data.get(aoa)
            plot_both = (aoa != 0.0)

            # Data extraction
            upper = extract_surface_data(res, 'upper')
            ref_upper = extract_ref_surface(ref_data, 'upper')

            # Upper surface plot
            if upper[var["key"]] is not None:
                ax.plot(upper["x"], upper[var["key"]], label="Thwaites (upper)", linestyle="-", color="#1f77b4")
            if ref_upper and ref_upper.get(var["key"]) is not None:
                ax.plot(ref_upper["x"], ref_upper[var["key"]], label="JavaFoil (upper)", linestyle="--", color="#1f77b4")

            # Lower surface plot
            if plot_both:
                lower = extract_surface_data(res, 'lower')
                ref_lower = extract_ref_surface(ref_data, 'lower')
                
                if lower[var["key"]] is not None:
                    ax.plot(lower["x"], lower[var["key"]], label="Thwaites (lower)", linestyle="-", color="#ff7f0e")
                if ref_lower and ref_lower.get(var["key"]) is not None:
                    ax.plot(ref_lower["x"], ref_lower[var["key"]], label="JavaFoil (lower)", linestyle="--", color="#ff7f0e")
            
            if var["invert_y"]:
                ax.invert_yaxis()
            
            ax.set_xlim(0, 1)
            ax.set_xlabel(r"$x/c$")
            ax.set_ylabel(var["ylabel"])
            ax.set_title(f"AoA = {aoa}°")
            ax.legend(fontsize=7, loc='best')
            ax.grid(True)

        # Fifth subplot
        ax5 = fig.add_subplot(gs[2, :])
        pos = ax5.get_position()
        new_width = pos.width * 0.5
        new_x0 = pos.x0 + (pos.width - new_width) / 2
        ax5.set_position([new_x0, pos.y0, new_width, pos.height])

        aoa = aoa_list[4]
        res = all_results[aoa]
        ref_data = all_ref_data.get(aoa)
        plot_both = (aoa != 0.0)
        
        upper = extract_surface_data(res, 'upper')
        ref_upper = extract_ref_surface(ref_data, 'upper')
        
        if upper[var["key"]] is not None:
            ax5.plot(upper["x"], upper[var["key"]], label="Thwaites (upper)", linestyle="-", color="#1f77b4")
        if ref_upper and ref_upper.get(var["key"]) is not None:
            ax5.plot(ref_upper["x"], ref_upper[var["key"]], label="JavaFoil (upper)", linestyle="--", color="#1f77b4")
        
        if plot_both:
            lower = extract_surface_data(res, 'lower')
            ref_lower = extract_ref_surface(ref_data, 'lower')
            
            if lower[var["key"]] is not None:
                ax5.plot(lower["x"], lower[var["key"]], label="Thwaites (lower)", linestyle="-", color="#ff7f0e")
            if ref_lower and ref_lower.get(var["key"]) is not None:
                ax5.plot(ref_lower["x"], ref_lower[var["key"]], label="JavaFoil (lower)", linestyle="--", color="#ff7f0e")
        
        if var["invert_y"]:
            ax5.invert_yaxis()
        
        ax5.set_xlim(0, 1)
        ax5.set_xlabel(r"$x/c$")
        ax5.set_ylabel(var["ylabel"])
        ax5.set_title(f"AoA = {aoa}°")
        ax5.legend(fontsize=7, loc='best')
        ax5.grid(True)

        # Figure title
        fig.suptitle(f"{var['title']}\nNACA {naca_series}", fontsize=14, fontweight='bold')

        # Save output
        fig.savefig(out / f"{var['filename']}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {var['filename']}.png")

    # Geometry combined plot
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.25)
    
    subplot_positions = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]
    
    for i, aoa in enumerate(aoa_list[:4]):
        ax = fig.add_subplot(subplot_positions[i])
        res = all_results[aoa]
        
        ax.plot(res["XB"], res["YB"], label="Airfoil", linestyle="-", color="#1f77b4")
        ax.plot(res["XBL"], res["YBL"], label="BL surface", linestyle="-", color="#ff7f0e")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.15, 0.15)
        ax.set_xlabel(r"$x/c$")
        ax.set_ylabel(r"$y/c$")
        ax.set_title(f"AoA = {aoa}°")
        ax.legend(fontsize=7, loc='best')
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')

    # Fifth subplot
    ax5 = fig.add_subplot(gs[2, :])
    pos = ax5.get_position()
    new_width = pos.width * 0.5
    new_x0 = pos.x0 + (pos.width - new_width) / 2
    ax5.set_position([new_x0, pos.y0, new_width, pos.height])
    
    aoa = aoa_list[4]
    res = all_results[aoa]
    ax5.plot(res["XB"], res["YB"], label="Airfoil", linestyle="-", color="#1f77b4")
    ax5.plot(res["XBL"], res["YBL"], label="BL surface", linestyle="-", color="#ff7f0e")
    ax5.set_xlim(0, 1)
    ax5.set_ylim(-0.15, 0.15)
    ax5.set_xlabel(r"$x/c$")
    ax5.set_ylabel(r"$y/c$")
    ax5.set_title(f"AoA = {aoa}°")
    ax5.legend(fontsize=7, loc='best')
    ax5.grid(True)
    ax5.set_aspect('equal', adjustable='box')
    
    fig.suptitle(f"Airfoil and Boundary Layer Geometry\nNACA {naca_series}", fontsize=14, fontweight='bold')
    fig.savefig(out / "geometry_combined.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved geometry_combined.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    naca_series = "0009"
    Np = 100
    AoA_deg = 0.0
    Re = 100000.0

    # Setup output directory
    results_base = Path("results")
    results_base.mkdir(parents=True, exist_ok=True)

    # Multiple angle of attack cases
    aoa_list = [-10, -5, 0, 5, 10]

    # Storage for combined plots
    all_results = {}
    all_ref_data = {}
    
    for AoA_deg in aoa_list:
        print(f"\n{'='*60}")
        print(f"Running for AoA = {AoA_deg}°")
        print(f"{'='*60}")

        # Load reference data
        aoa_int = int(AoA_deg)
        if aoa_int >= 0:
            javafoil_path = Path(f"javafoil_data/alpha_{aoa_int}/java_foil_data.txt")
        else:
            javafoil_path = Path(f"javafoil_data/alpha_{aoa_int}/java_foil_data.txt")
        
        ref_data = load_javafoil_surface(javafoil_path) if javafoil_path.exists() else None
        if ref_data:
            print(f"Loaded JavaFoil data from {javafoil_path}")
        else:
            print(f"JavaFoil data not found at {javafoil_path}")

        # Execute solver
        print("Running iterative Viscous-Inviscid Interaction...")
        res = run_case_iterative(
            naca_series=naca_series,
            Np=Np,
            AoA_deg=AoA_deg,
            Re=Re,
            max_iter=50,
            tol=5e-3,
            relax=0.3,
        )
        print(f"Cl (inviscid): {res['Cl_inviscid']:.4f}")
        print(f"Cl (Thwaites VII): {res['cl']:.4f}")
        print(f"Cd (Thwaites VII): {res['cd']:.5f}")

        # Store results
        all_results[AoA_deg] = res
        all_ref_data[AoA_deg] = ref_data

        # Output directory setup
        aoa_str = f"aoa_{aoa_int:+d}".replace("+", "pos").replace("-", "neg")
        out_dir = results_base / f"thwaites_{naca_series}_{aoa_str}"

        figs = plot_results(res, naca_series, AoA_deg, xfoil_data=ref_data, output_dir=out_dir)

        # Convergence plot
        if "convergence_history" in res and res["convergence_history"] and len(res["convergence_history"]) > 1:
            hist = np.array(res["convergence_history"])
            hist = np.maximum(hist, 1e-12)
            fig_conv = plt.figure()
            plt.semilogy(range(1, len(hist) + 1), hist, "o-")
            plt.xlabel("Iteration")
            plt.ylabel(r"Max $\Delta \delta^*$")
            plt.title(f"VII Convergence History (AoA = {AoA_deg}°)")
            plt.grid(True)
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            fig_conv.savefig(out / "convergence.png", dpi=200, bbox_inches="tight")
            figs.append(fig_conv)
            plt.close(fig_conv)

        # Cleanup
        for fig in figs:
            plt.close(fig)

        print(f"Saved plots to {out_dir.resolve()}")
    
    print(f"\n{'='*60}")
    print("All individual AoA cases completed!")
    print(f"{'='*60}")
    
    # Generate combined subplot figures
    print(f"\n{'='*60}")
    print("Generating combined subplot figures...")
    print(f"{'='*60}")
    
    combined_output_dir = results_base / "combined_plots"
    plot_combined_subplots(all_results, all_ref_data, aoa_list, naca_series, combined_output_dir)
    print(f"Combined plots saved to {combined_output_dir.resolve()}")

    # Polar curves generation
    print(f"\n{'='*60}")
    print("Generating Cl and Cd vs AoA polar curves...")
    print(f"{'='*60}")
    
    aoa_sweep_range = np.arange(-4, 4.5, 0.5)  # -4 to 4 degrees
    thwaites_polar = run_aoa_sweep(naca_series=naca_series, Np=50, Re=Re, aoa_range=aoa_sweep_range)

    # Load reference polar data
    javafoil_polar_path = Path("javafoil_data/cl_result_javafoil.txt")
    javafoil_polar = load_javafoil_polar(javafoil_polar_path) if javafoil_polar_path.exists() else None
    
    if javafoil_polar is not None:
        print(f"JavaFoil polar data loaded from {javafoil_polar_path}")

    # Plot and save polar comparison
    polar_figs = plot_polar(thwaites_polar, javafoil_polar, naca_series=naca_series, Re=int(Re), output_dir=results_base)

    # Cleanup
    for fig in polar_figs:
        plt.close(fig)
    
    print(f"Cl and Cd polar plots saved to {results_base.resolve()}")
    
    print(f"\n{'='*60}")
    print("All plots completed!")
    print(f"Results saved to: {results_base.resolve()}")
    print(f"{'='*60}")

