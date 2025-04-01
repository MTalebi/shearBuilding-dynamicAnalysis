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
