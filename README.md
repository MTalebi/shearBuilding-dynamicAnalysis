# Shear Building Analysis App

Author: **Mohammad Talebi-Kalaleh**  
Email: *<talebika@ualberta.ca>*

## Overview

The **Shear Building Analysis App** is an interactive web application built with **Streamlit** and **Plotly**. It provides a user-friendly interface to:

1. **Define** the dynamic characteristics of a multi-story shear building model (mass, stiffness, damping).  
2. **Specify** Rayleigh damping parameters by choosing two modes that share the same damping ratio.  
3. **Input** custom load functions in LaTeX form, either applied uniformly to all stories or individually to each story.  
4. **Simulate** the system response (displacements, velocities, accelerations, and interstory drifts) over a user-defined time span.  
5. **Visualize** results with interactive **Plotly** plots (mode shapes, time histories).  
6. **Explore** modal properties in a detailed Pandas table.

This application leverages **Python**’s scientific stack—`numpy`, `scipy`, `sympy`, `pandas`, and more—under the hood to perform robust eigenvalue analysis, matrix exponentials for state-space discretization, and dynamic response computations.

---

## Key Features

1. **Shear Building Model**  
   - Banded stiffness matrix automatically generated from user-provided floor stiffnesses.  
   - Diagonal mass matrix from user-supplied floor masses.  
   - Rayleigh damping matrix derived from two selected modes.

2. **Modal Analysis**  
   - Eigenvalue analysis of \(\mathbf{M}^{-1}\mathbf{K}\).  
   - Natural frequencies (\(\omega\)) and mode shapes (\(\phi\)) displayed.  
   - Users can **select** which mode shapes to visualize in interactive Plotly subplots.

3. **Rayleigh Damping**  
   - Single damping ratio \(\zeta\) across two chosen modes (e.g., Mode #1 and Mode #2).  
   - Automatic calculation of \(\alpha\) and \(\beta\) for \(\mathbf{C} = \alpha \mathbf{M} + \beta \mathbf{K}\).

4. **Load Functions in LaTeX**  
   - **Sympy** parses LaTeX input strings (e.g., \(\sin(2 \pi \cdot 5 \, t)\), step loads, etc.).  
   - Option to apply **one** load function across **all** stories or **individual** loads per story.

5. **Dynamic Response Computation**  
   - Discrete-time state-space approach using **matrix exponentials**.  
   - Time histories of **displacements**, **velocities**, **accelerations**, and **interstory drifts**.  
   - Configurable time step (\(\Delta t\)) and total simulation time \(T\).

6. **Interactive Visualization**  
   - **Modal Properties**: A Pandas table with \(\omega\), frequency in Hz, and period \(T\).  
   - **Mode Shapes**: Plotly subplots with normalized amplitudes.  
   - **Response Histories**: Plotly time-series charts of selected DOFs (e.g., top floors).

7. **User-Friendly Streamlit Interface**  
   - **Sidebar** for all input parameters (masses, stiffnesses, damping, load functions, initial conditions).  
   - **Tabs** for switching between **Modal Analysis** and **Response Histories**.  
   - Option to **reset** inputs or **run** the simulation on demand.

---

## Installation

1. **Clone or download** the repository containing:
   - `base_shear_code.py` (the core computational module)
   - `shear_building_app.py` (the Streamlit front-end)
   - `requirements.txt` (list of dependencies)
   - `INSTALLATION_INSTRUCTIONS.txt` (detailed setup steps)

2. **Install dependencies** (preferably inside a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
   or with conda:
   ```bash
   conda create -n shear_app python=3.9
   conda activate shear_app
   conda install --file requirements.txt
   ```

3. **Launch** the application:
   ```bash
   streamlit run shear_building_app.py
   ```
   The browser will open automatically at [http://localhost:8501](http://localhost:8501).

---

## Usage

1. **Open the sidebar** to configure model parameters:
   - **Mass & Stiffness** per story (comma-separated).  
   - **Damping Ratio** (\(\zeta\)) and **Mode Numbers** (for Rayleigh damping).  
   - **Load Function(s)** in LaTeX (choose either “Same Load” or “Different Load”).  
   - **Time Settings** (\(T\), \(\Delta t\)).  
   - (Optional) **Initial Conditions** (displacement and velocity).

2. **Run Simulation**:
   - Click **“Run Simulation”**. The results will be computed and displayed in the right panel.

3. **View Results**:
   - **Modal Analysis** tab:
     - A table of modes (Mode #, \(\omega\) in rad/s, \(f\) in Hz, \(T\) in s).  
     - Rayleigh damping coefficients (\(\alpha, \beta\)).  
     - Interactive **mode shape** plots for selected modes.
   - **Response Histories** tab:
     - Select **displacement**, **velocity**, **total acceleration**, or **interstory drift**.  
     - Choose which stories to plot.  
     - An interactive time-series chart is generated using Plotly.

4. **Reset Inputs** if needed, or adjust parameters and re-run.

---

## Example

1. **Three-Story Example**  
   - Masses: `1000, 1200, 1500`  
   - Stiffnesses: `2e6, 2e6, 2e6`  
   - \(\zeta = 0.05\), Mode Pair = (1, 2)  
   - **Load**: \(F_1(t) = 10^5 \sin(2\pi \cdot 5t)\), \(F_2(t) = 0\), \(F_3(t) = 10^5 (t<0.05)\)  
   - **Time**: \(T=1.0\), \(\Delta t=0.001\)  
   - **Results**:  
     - Mode #1 freq ~ 18.7 rad/s, etc.  
     - Plots of displacements, velocities, or accelerations for each floor.

---

## Contributing

Contributions are welcome—feel free to open a pull request or issue on the repository if you discover a bug or have a feature request.

---

## Contact

**Developer**: *Mohammad Talebi-Kalaleh (<talebika@ualberta.ca>)*

If you have questions, suggestions, or comments, feel free to reach out via email.

---

*(End of README.md)*
