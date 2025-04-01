#!/usr/bin/env python3
"""
Shear Building Analysis with Rayleigh Damping + Pandas + Plotly
Developer: Mohammad Talebi-Kalaleh <talebika@ualberta.ca>

Key Features:
-------------
1) User provides:
   - number of DOFs (stories)
   - mass array
   - stiffness array
   - two mode numbers with a single damping ratio zeta
   - time-step, total time
   - load functions as LaTeX strings (for each story or a single string)
2) We:
   - build M, K
   - do eigenvalue analysis => frequencies, mode shapes
   - produce a Pandas DataFrame of frequencies, periods, etc.
   - compute a, b for Rayleigh damping
   - parse latex load functions -> sympy -> callable
   - build continuous & discrete state-space, time-step solution
   - produce interactive mode-shape plots (Plotly) in subplots
   - produce an interactive time-history plot of displacements vs. time
"""

import numpy as np
import math
import sympy
from sympy.parsing.latex import parse_latex
from scipy.linalg import expm
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

###############################################################################
# 1) Matrix-Building and Damping Computation
###############################################################################

def shear_building_stiffness_matrix(num_stories, stiffnesses):
    """
    Generate the banded stiffness matrix for a shear building.
    """
    K = np.zeros((num_stories, num_stories))
    for i in range(num_stories):
        K[i, i] = stiffnesses[i]
        if i < num_stories - 1:
            K[i, i] += stiffnesses[i+1]
        if i > 0:
            K[i, i-1] = -stiffnesses[i]
            K[i-1, i] = -stiffnesses[i]
    return K

def eigen_analysis(M, K):
    """
    Perform eigenvalue analysis on (M^-1 K).

    Returns
    -------
    omegas : (n,) array  (sorted ascending)
    Phi    : (n x n) array (mode shapes in columns)
    """
    evals, evecs = np.linalg.eig(np.linalg.inv(M) @ K)
    idx_sort = np.argsort(evals)
    evals_sorted = evals[idx_sort]
    evecs_sorted = evecs[:, idx_sort]
    # Frequencies
    omegas = np.sqrt(evals_sorted)
    # Normalize
    Phi = np.zeros_like(evecs_sorted)
    for i in range(evecs_sorted.shape[1]):
        mode_shape = evecs_sorted[:, i]
        max_val = np.max(np.abs(mode_shape))
        if max_val < 1e-12:
            Phi[:, i] = mode_shape
        else:
            Phi[:, i] = mode_shape / max_val
    return omegas, Phi

def compute_rayleigh_damping_coeffs(omega1, omega2, zeta):
    """
    For two modes with the same damping ratio zeta => solve:
        2*zeta*omega1 = a + b*omega1^2
        2*zeta*omega2 = a + b*omega2^2
    """
    A_mat = np.array([
        [1.0,        omega1**2],
        [1.0,        omega2**2]
    ])
    b_vec = np.array([
        [2.0*zeta*omega1],
        [2.0*zeta*omega2]
    ])
    sol = np.linalg.solve(A_mat, b_vec)
    a_coeff = sol[0,0]
    b_coeff = sol[1,0]
    return a_coeff, b_coeff

###############################################################################
# 2) State-Space & Discretization
###############################################################################

def build_state_space_matrices(M, C, K):
    """
    Build continuous-time state-space (A, B) for
      M x'' + C x' + K x = F(t)
    """
    n = M.shape[0]
    M_inv = np.linalg.inv(M)

    A_top = np.hstack((np.zeros((n,n)), np.eye(n)))
    A_bot = np.hstack((-M_inv @ K, -M_inv @ C))
    A = np.vstack((A_top, A_bot))

    B_top = np.zeros((n, n))
    B_bot = M_inv
    B = np.vstack((B_top, B_bot))
    return A, B

def discretize_state_space(A, B, dt):
    """
    Discretize (A, B) -> (Ad, Bd) using matrix exponentials.
    """
    num_states = A.shape[0]
    I = np.eye(num_states)
    Ad = expm(A * dt)
    try:
        A_inv = np.linalg.inv(A)
        Bd = A_inv @ ((Ad - I) @ B)
    except np.linalg.LinAlgError:
        Bd = dt * B
    return Ad, Bd

###############################################################################
# 3) LaTeX to Sympy to Callable
###############################################################################

def latex_to_callable(latex_str, num_stories):
    t_sym = sympy.Symbol('t', real=True)
    expr = parse_latex(latex_str)
    # Use a custom dictionary mapping for pi and sin
    func = sympy.lambdify(t_sym, expr, modules=[{"pi": math.pi, "sin": math.sin}, "math"])
    def load_at_time(t):
        t_val = float(t)  # ensure t is a Python float
        val = func(t_val)
        # If still symbolic, force evaluation
        if hasattr(val, "evalf"):
            val = val.evalf()
        return np.full(num_stories, float(val), dtype=float)
    return load_at_time

def latex_to_callable_list(latex_str_list, num_stories):
    t_sym = sympy.Symbol('t', real=True)
    exprs = [parse_latex(s) for s in latex_str_list]
    funcs = [sympy.lambdify(t_sym, e, modules=[{"pi": math.pi, "sin": math.sin}, "math"]) for e in exprs]
    def load_at_time(t):
        t_val = float(t)
        vals = []
        for f in funcs:
            val = f(t_val)
            if hasattr(val, "evalf"):
                val = val.evalf()
            vals.append(float(val))
        return np.array(vals, dtype=float)
    return load_at_time


###############################################################################
# 4) Main Analysis Function
###############################################################################

def shear_building_analysis_with_rayleigh(
    num_stories,
    masses,
    stiffnesses,
    mode_pair,   # e.g. (1, 2)
    zeta,        # damping ratio for both chosen modes
    dt,
    T,
    load_latex_list=None,
    x0=None,
    dx0=None
):
    """
    Perform a time-domain analysis of a shear building with Rayleigh damping
    from the user-chosen modes.

    Returns
    -------
    results_dict : dict
       {
         "time_array": ...,
         "displacement": ...,
         "velocity": ...,
         "acceleration": ...,
         "drift": ...,
         "omega_table": pd.DataFrame(...),
         "Phi": (n x n),
         ...
       }
    """
    # 1) Build M, K
    M = np.diag(masses)
    K = shear_building_stiffness_matrix(num_stories, stiffnesses)

    # 2) Eigen Analysis
    omegas, Phi = eigen_analysis(M, K)
    freq_hz = omegas/(2*np.pi)
    periods = 2*np.pi/omegas

    # Pandas table of modal properties
    mode_nums = np.arange(1, num_stories+1, dtype=int)
    df_modes = pd.DataFrame({
        "Mode": mode_nums,
        r"omega (rad/s)": omegas,
        r"f (Hz)": freq_hz,
        r"T (s)": periods
    })

    # 3) Rayleigh damping from chosen modes
    mode1 = mode_pair[0] - 1  # zero-based
    mode2 = mode_pair[1] - 1  # zero-based
    omega1 = omegas[mode1]
    if num_stories > 1:
        omega2 = omegas[mode2]
        a_coeff, b_coeff = compute_rayleigh_damping_coeffs(omega1, omega2, zeta)
    else:
        a_coeff, b_coeff = 2 * omegas[mode1] * zeta, 0
    C = a_coeff * M + b_coeff * K

    # 4) State-space
    A, B = build_state_space_matrices(M, C, K)
    Ad, Bd = discretize_state_space(A, B, dt)

    # 5) Load function
    if load_latex_list is None:
        def load_function(t):
            return np.zeros(num_stories)
    elif isinstance(load_latex_list, str):
        load_function = latex_to_callable(load_latex_list, num_stories)
    elif isinstance(load_latex_list, (list, tuple)):
        load_function = latex_to_callable_list(load_latex_list, num_stories)
    else:
        raise ValueError("load_latex_list must be None, str, or list/tuple of str.")

    # 6) Time stepping
    num_steps = int(np.floor(T/dt))
    time_array = np.arange(num_steps)*dt
    x_state = np.zeros((num_steps, 2*num_stories))

    if x0 is None:
        x0 = np.zeros(num_stories)
    if dx0 is None:
        dx0 = np.zeros(num_stories)
    x_state[0, :num_stories] = x0
    x_state[0, num_stories:] = dx0

    for k_step in range(num_steps-1):
        xk = x_state[k_step, :]
        f_current = load_function(time_array[k_step])
        x_next = Ad @ xk + Bd @ f_current
        x_state[k_step+1, :] = x_next

    n = M.shape[0]
    disps = x_state[:, :n]   # shape (num_steps, n)
    vels  = x_state[:, n:]   # shape (num_steps, n)

    # Arrays to store results
    accs = np.zeros((num_steps, n))
    drifts = np.zeros((num_steps, n))  # We'll store drift(i) = x(i+1) - x(i) for i=0..n-2

    # Precompute M^-1 if needed
    M_inv = np.linalg.inv(M)

    for k in range(num_steps):
        # acceleration: M^-1( f - C*vel - K*disp )
        disp_k = disps[k, :]
        vel_k  = vels[k, :]
        f_k    = load_function(time_array[k])  # shape (n,)

        acc_k = M_inv @ (f_k - C @ vel_k - K @ disp_k)
        accs[k, :] = acc_k

        # Interstory drift
        # For story i, drift_i = x_i+1 - x_i. 
        # We'll define drift_n = 0 or just keep last one as 0
        for i_story in range(n-1):
            drifts[k, i_story] = disp_k[i_story+1] - disp_k[i_story]
        # Last entry remains 0 for convenience.
    # Collect results in a dictionary
    results_dict = {
        "time_array": time_array,
        "displacement": disps,
        "velocity": vels,
        "acceleration": accs,
        "drift": drifts,       
        "omega_table": df_modes,
        "Phi": Phi,
        "a_coeff": a_coeff,
        "b_coeff": b_coeff
    }
    return results_dict

###############################################################################
# 5) Example Usage (if run as main)
###############################################################################

if __name__ == "__main__":
    # Example
    import math

    num_stories = 3
    masses = [1000.0, 1200.0, 1500.0]
    stiffnesses = [2e6, 2e6, 2e6]

    mode_pair = (1, 2)  # 1-based
    zeta = 0.05

    dt = 0.001
    T = 1.0

    # Example load for each DOF as LaTeX strings:
    #  1) "1e5 sin(2 pi * 5 t)"
    #  2) "0"
    #  3) "1e5 * (t<0.05)"
    load_latex_list = [
        r"100 \sin(5 t)",
        r"0",
        r"100 * (t<0.05)"
    ]

    # Solve
    results = shear_building_analysis_with_rayleigh(
        num_stories=num_stories,
        masses=masses,
        stiffnesses=stiffnesses,
        mode_pair=mode_pair,
        zeta=zeta,
        dt=dt,
        T=T,
        load_latex_list=load_latex_list,
        x0=None,
        dx0=None
    )

    # 1) Show the mode table (Pandas)
    print("\n--- Modal Properties Table ---")
    print(results["omega_table"])

    # 2) Show Rayleigh damping coefficients
    print(f"\nRayleigh damping: a = {results['a_coeff']:.4e}, b = {results['b_coeff']:.4e}")
