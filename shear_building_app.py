#!/usr/bin/env python3

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import copy

# Import your existing functions from the base module:
from shear_building_utiles import (
    shear_building_analysis_with_rayleigh,
    latex_to_callable, latex_to_callable_list
)

###############################################################################
# Cached Simulation Function
###############################################################################
@st.cache_data
def cached_simulation(num_stories, masses, stiffnesses, mode_pair, zeta, dt, T, load_latex_list, x0, dx0):
    # This function wraps our heavy simulation function
    return shear_building_analysis_with_rayleigh(
        num_stories=num_stories,
        masses=masses,
        stiffnesses=stiffnesses,
        mode_pair=mode_pair,
        zeta=zeta,
        dt=dt,
        T=T,
        load_latex_list=load_latex_list,
        x0=x0,
        dx0=dx0
    )

###############################################################################
# Streamlit Layout
###############################################################################

def main():
    st.set_page_config(page_title="Shear Building App", layout="wide")

    # ---------------
    # Sidebar Inputs
    # ---------------
    st.sidebar.header("System Parameters")

    # Masses
    mass_str = st.sidebar.text_area(
        "Mass values per story (kg)",
        value="1000, 1200, 1500",
        help="Comma-separated, e.g. 1000, 1200, 1500"
    )
    masses = [float(x.strip()) for x in mass_str.split(",") if x.strip()]

    # Stiffness
    stiff_str = st.sidebar.text_area(
        "Stiffness values per story (N/m)",
        value="2e6, 2e6, 2e6",
        help="Comma-separated, must match length of mass list."
    )
    stiffnesses = [float(x.strip()) for x in stiff_str.split(",") if x.strip()]

    # Check consistency
    if len(masses) != len(stiffnesses):
        st.sidebar.error("Length of mass list must match length of stiffness list!")
        return

    num_stories = len(masses)

    # Damping
    st.sidebar.header("Damping Settings")
    zeta = st.sidebar.number_input(
        "Rayleigh Damping Ratio",
        min_value=0.0, max_value=1.0, value=0.05, step=0.01
    )

    if num_stories > 1:
        st.sidebar.write("Select Two Mode Numbers for Rayleigh Damping")
        mode1 = st.sidebar.selectbox("Mode #1", list(range(1, num_stories+1)), index=0)
        mode2 = st.sidebar.selectbox("Mode #2", list(range(1, num_stories+1)), index=1)
    else:
        mode1, mode2 = 1, 1

    # Load Function
    st.sidebar.header("Load Function")
    load_type = st.sidebar.radio("Loading Type", ["Same Load for All Stories", "Different Load for Each Story"])

    if load_type == "Same Load for All Stories":
        single_load_latex = st.sidebar.text_input(
            "LaTeX for Load (applied equally)",
            value=r"100 \sin(5*t)"
        )
        load_latex_list = single_load_latex  # just a string
    else:
        # Different load per story
        load_latex_list = []
        for i_story in range(num_stories):
            default_expr = r"0"
            load_i = st.sidebar.text_input(
                f"LaTeX for story #{i_story+1}",
                value=default_expr
            )
            load_latex_list.append(load_i)

    # Time Settings
    st.sidebar.header("Time Settings")
    T_str = st.sidebar.text_input("Total Simulation Time (s)", value="2.0")
    dt_str = st.sidebar.text_input("Time Step (s)", value="0.01")
    try:
        T = float(T_str)
        dt = float(dt_str)
    except:
        st.sidebar.error("Invalid time settings.")
        return

    # Initial Conditions
    st.sidebar.header("Initial Conditions (Optional)")
    x0_str = st.sidebar.text_area(
        "Initial Displacement per Story (m)",
        value="",
        help="Comma-separated, leave empty for zeros"
    )
    dx0_str = st.sidebar.text_area(
        "Initial Velocity per Story (m/s)",
        value="",
        help="Comma-separated, leave empty for zeros"
    )
    x0 = []
    dx0 = []
    if x0_str.strip():
        x0 = [float(x.strip()) for x in x0_str.split(",") if x.strip()]
    if dx0_str.strip():
        dx0 = [float(x.strip()) for x in dx0_str.split(",") if x.strip()]

    # Ensure correct lengths or fallback to zero
    if len(x0) != num_stories:
        x0 = [0.0]*num_stories
    if len(dx0) != num_stories:
        dx0 = [0.0]*num_stories

    # Buttons
    st.sidebar.write("---")
    run_button = st.sidebar.button("Run Simulation")
    reset_button = st.sidebar.button("Reset Inputs")

    if reset_button:
        # This just reruns the script clearing the fields to defaults
        st.experimental_rerun()

    # ---------------
    # Main Panel
    # ---------------
    st.title("Shear Building Analysis App")

   # If the simulation has been run previously, keep the results in session_state.
    if run_button:
        with st.spinner("Running simulation..."):
            results = cached_simulation(
                num_stories=num_stories,
                masses=masses,
                stiffnesses=stiffnesses,
                mode_pair=(mode1, mode2),
                zeta=zeta,
                dt=dt,
                T=T,
                load_latex_list=load_latex_list,
                x0=x0,
                dx0=dx0
            )
            st.session_state["results"] = results
    elif "results" in st.session_state:
        results = st.session_state["results"]
    else:
        st.info("Set your inputs in the sidebar and click 'Run Simulation' to see results.")
        return
    
    # Unpack results
    time_array = results["time_array"]
    df_modes = results["omega_table"]
    Phi = results["Phi"]
    a_coeff = results["a_coeff"]
    b_coeff = results["b_coeff"]

    # For acceleration or drift, we need M, K from the base code.
    # If not directly available in results, we can quickly rebuild them or store them.
    # Alternatively, your base code might return M, K as well. Suppose we do that here:
    # ...
    # For demonstration, let's call it again or adapt as needed:
    # We'll just do a quick "M = diag(masses), K=..." approach here:
    M = np.diag(masses)
    # For a shear building, you can replicate the logic from your base code if not stored:
    K = np.zeros((num_stories, num_stories))
    for i in range(num_stories):
        K[i, i] = stiffnesses[i]
        if i > 0:
            K[i, i] += stiffnesses[i-1]
            K[i, i-1] = -stiffnesses[i-1]
            K[i-1, i] = -stiffnesses[i-1]

    # We'll build a load_function for computing acceleration offline:
    if isinstance(load_latex_list, str):
        load_func = latex_to_callable(load_latex_list, num_stories)
    else:
        load_func = latex_to_callable_list(load_latex_list, num_stories)

    # Make tabs
    tab1, tab2 = st.tabs(["Modal Analysis", "Response Histories"])

    # ------------------------
    # Tab 1: Modal Analysis
    # ------------------------
    with tab1:
        st.subheader("Modal Properties")
        st.dataframe(df_modes.style.format(precision=4))


        # Multi-select: which modes to display
        mode_options = list(range(1, num_stories+1))
        selected_modes = st.multiselect(
            "Select Modes to Display",
            mode_options,
            default=mode_options[:3]  # default first 3
        )
        if selected_modes:
            # Create a custom Phi that only has the selected modes
            # Phi is (n x n). The i-th mode is the i-th column (0-based).
            # We'll extract columns for the selected modes
            # We can create a sub-matrix with the selected columns
            # but the existing plot_mode_shapes_plotly shows *all* modes by default
            # in subplots. Let's just create a new function or patch the original.
            # For simplicity, let's do a small snippet:

            selected_cols = [m-1 for m in selected_modes]  # zero-based
            Phi_subset = Phi[:, selected_cols]

            # We'll create a single figure with subplots:
            fig_modes = make_subplots(
                rows=1, cols=len(selected_cols),
                subplot_titles=[f"Mode {m}" for m in selected_modes],
                horizontal_spacing=0.15
            )
            for i_col, col_index in enumerate(selected_cols):
                shape = Phi[:, col_index]
                shape_with_base = np.concatenate(([0.0], shape))
                levels = np.arange(num_stories+1)
                fig_modes.add_trace(
                    go.Scatter(
                        x=shape_with_base,
                        y=levels,
                        mode='lines+markers',
                        name=f"Mode {col_index+1}",
                    ),
                    row=1, col=i_col+1
                )
                # Apply to all subplots
                fig_modes.update_xaxes(
                    range=(-1.05, 1.05),
                    dtick=1,
                    showgrid=True,  # Explicitly enable grid
                    gridcolor='lightgray',
                    gridwidth=0.5,
                    griddash='solid',
                    minor_showgrid=False,
                    row=1, col=i_col+1  # Specify which subplot
                )                    
                fig_modes.update_yaxes(
                    dtick=1,  # Set grid lines exactly 1 unit apart
                    gridcolor='lightgray',
                    gridwidth=0.5,
                    griddash='solid',
                    minor_showgrid=False,
                    row=1, 
                    col=i_col+1
                    )
            fig_modes.update_layout(
                title="Selected Mode Shapes",
                showlegend=False,
                height=500,
                width=300*len(selected_cols)
            )
            fig_modes.update_xaxes(title="Normalized Amplitude")
            fig_modes.update_yaxes(title="Story Level")

            st.plotly_chart(fig_modes, use_container_width=True)

    # ------------------------
    # Tab 2: Response Histories
    # ------------------------
    with tab2:
        st.subheader("Response Histories")

        # Choose response type
        resp_type = st.selectbox(
            "Select Response Quantity",
            ["Displacement", "Velocity", "Acceleration", "Interstory Drift"]
        )

        # Story selector
        story_options = list(range(1, num_stories+1))
        selected_stories = st.multiselect(
            "Select Stories to Plot",
            story_options,
            default=story_options[-3:] if num_stories >= 3 else story_options
        )
        show_all = st.checkbox("Show All Stories", value=False)
        if show_all:
            selected_stories = story_options

        # Build the time history figure
        fig_resp = go.Figure()
        # Extract desired data
        if resp_type == "Displacement":
            data_array = results["displacement"]  # shape (num_steps, n)
            ylabel = "Displacement (m)"
        elif resp_type == "Velocity":
            data_array = results["velocity"]
            ylabel = "Velocity (m/s)"
        elif resp_type == "Acceleration":
            data_array = results["acceleration"]
            ylabel = "Acceleration (m/s²)"
        else:  # Interstory Drift
            data_array = results["drift"]
            ylabel = "Interstory Drift (m)"

        for s in selected_stories:
            idx = s - 1  # zero-based
            if resp_type == "Interstory Drift" and idx == num_stories - 1:
                # The top story's drift is 0 by definition in our array
                # (because there's no story above it). We'll still plot for completeness.
                pass
            fig_resp.add_trace(
                go.Scatter(
                    x=time_array,
                    y=data_array[:, idx],
                    mode='lines',
                    name=f"Story {s}"
                )
            )
        fig_resp.update_layout(
            title=f"{resp_type} Time History",
            xaxis_title="Time (s)",
            yaxis_title=ylabel,
            template="plotly_white",
            height=500, width=900
        )
        st.plotly_chart(fig_resp, use_container_width=True)

# End main()

if __name__ == "__main__":
    main()
