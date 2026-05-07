# EN.553.481.Final_Project

# Physics-Informed Neural Networks for Solving Differential Equations

This repository contains the source code for my EN.553.481/681 Numerical Analysis final project on Physics-Informed Neural Networks (PINNs) for solving differential equations.

The project compares classical numerical methods with PINN-based approaches using automatic differentiation (AD) and finite-difference derivative approximations (FDM).

## Project Contents

The code addresses the following problems:

1. **First-Order ODE**
   - Forward Euler method
   - Classical RK4 method
   - AD-PINN
   - FDM-PINN
   - Comparison of errors, training times, and finite-difference step size effects

2. **1D Heat Equation**
   - Explicit Forward Euler finite-difference method
   - AD-PINN
   - FDM-PINN
   - CFL stability experiment
   - Comparison of errors, training times, and finite-difference step size effects

3. **Analysis and Discussion**
   - Error comparison across methods
   - Effect of collocation points
   - Effect of network size
   - Discussion connecting the project to numerical analysis concepts

4. **Bonus**
   - Inverse problem for recovering the unknown diffusion coefficient \(\nu\) in the heat equation using noisy data

## Files

The main Python script contains all implementations, including:

- Classical ODE solvers
- Heat equation finite-difference solver
- PINN model class
- AD and FDM loss functions
- Training routines
- Plotting functions
- Comparison tables
- Bonus inverse problem code

Generated figures are saved as `.png` files and used in the final PDF report.

## Requirements

The code was run using Python 3 with the following packages:

bash
numpy
matplotlib
torch

To install the required packages, run:

pip install numpy matplotlib torch torchvision torchaudio

If using Anaconda, make sure PyTorch is installed in the same Python environment used to run the script.

How to Run

From the command line, run:

python final_project.py

or from Spyder/Jupyter, open the script and run the file directly.

The script prints numerical tables to the console and saves figures such as loss curves, solution plots, heatmaps, and comparison plots.

Notes on Runtime

Some PINN experiments, especially the heat equation AD-PINN and FDM-PINN, can take several minutes because they require many collocation points and thousands of training epochs.

To make testing faster, the script includes adjustable settings such as:

FAST_TEST = True

When FAST_TEST = True, the code uses fewer epochs and fewer collocation points for debugging. For final reported results, this should be changed back to:

FAST_TEST = False

The full runs use the project-specified settings, such as:
	•	(N_r = 500) for the ODE PINN residual
	•	(N_r = 10000) for the heat equation PINN residual
	•	10,000 epochs for the ODE PINNs
	•	20,000 epochs for the heat equation PINNs

Some later analysis experiments use shorter training runs to keep the computational cost reasonable while comparing trends across collocation points and network sizes.

Output

Running the script produces:
	•	Training loss curves
	•	ODE solution comparisons
	•	Heat equation prediction and error heatmaps
	•	Error-versus-step-size plots
	•	Error-versus-collocation-point plots
	•	Tables comparing error and training time
	•	Bonus inverse problem summary table

The main numerical quantities reported include:
	•	Maximum absolute error for the ODE
	•	Relative (L^2) error for the heat equation
	•	(L^2) error at (t=0.5) for the finite-difference heat solver
	•	Wall-clock training times for PINN methods
	•	Recovered diffusion coefficient (\nu) for the bonus inverse problem

Author

Ming Zhong
EN.553.481/681 Numerical Analysis
Johns Hopkins University
April 2026

