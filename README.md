# Viscous-Inviscid Interaction Panel Thwaites Method

This repository contains a Python implementation of a Viscous-Inviscid Interaction (VII) solver for NACA 4-digit airfoils. It couples a **Vortex Panel Method** (for inviscid flow) with **Thwaites' Laminar Boundary Layer Method** (for viscous flow) to estimate aerodynamic characteristics.

## Features

- **Geometry Generation**: Generates coordinates for NACA 4-digit airfoils with cosine spacing.
- **Inviscid Solver**: Uses a linear strength Vortex Panel Method.
- **Viscous Solver**: Implements Thwaites' method to solve integral boundary layer equations.
- **VII Coupling**: Iteratively couples the viscous and inviscid solutions using an under-relaxation scheme on the displacement thickness.
- **Aerodynamic Coefficients**: Calculates $C_l$, $C_d$, and $C_p$.
- **Validation**: Includes plotting utilities to compare results against JavaFoil or XFOIL data.

## Installation

Ensure you have Python installed along with the following dependencies:

```bash
pip install numpy matplotlib
```

## Usage

You can run the main solver script directly. It is configured to run a sweep of Angle of Attacks (AoA) for a NACA 0009 airfoil by default.

```bash
python panel_thwaites_solver.py
```

The script will:
1. Generate geometry.
2. Run the VII solver for defined AoAs (default: -10, -5, 0, 5, 10).
3. Generate and save plots in the `results/` directory, including:
   - Pressure coefficient ($C_p$) distribution.
   - Boundary layer properties ($\delta$, $\delta^*$, $\theta$, $C_f$).
   - Convergence history.
   - Combined geometry and property plots.
4. Calculate and plot $C_l$ and $C_d$ polar curves.

## Structure

- `panel_thwaites_solver.py`: Main source code containing the solver classes and execution logic.
- `javafoil_data/`: Directory containing reference data for validation (if available).
- `results/`: Output directory where plots and data are saved.

## Methodology

The solver follows this process:
1. **Inviscid Solution**: Initial potential flow solution.
2. **Boundary Layer**: Marching solution using Thwaites' correlations for laminar flow.
3. **Viscous-Inviscid Coupling**:  Modifies the airfoil geometry (or transpiration velocity) based on displacement thickness and re-solves the potential flow. This is repeated until convergence.
4. **Force Integration**: Lift is obtained from pressure integration, and drag is calculated using the Squire-Young formula or integral momentum balance at the trailing edge.

## References

- Computations of Viscous-Inviscid Interactions using Panel Method and Thwaites' Method.
- NACA Report No. 460.
