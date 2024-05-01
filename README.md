# WAVE : Methods for Solving the Wave Equation
### Comparing Numerical Methods on Cylindrical Wave Equation Solver Performance
<p align="left">
<img src="https://github.com/mw6136/WAVE/assets/144184708/d6cf7824-2713-4758-9624-795f0be29dc5" width="60%">
</p>

## A Model of Surface Deformation with the Cylindrical Wave Equation
<p align="left">
<img src="https://github.com/mw6136/WAVE/assets/144184708/b3378325-87dc-4669-991d-cb421a41f216" width="60%">
</p>

## Running the code
Base script
``run_cases.py``

Numerical methods
``iterative_methods.py``
``fft_solver.py``
``fwd_euler.py``
``bwd_euler.py``

Analytical solution
``getting_analytical_data.py``

### Slurm Submission
There are also varying example slurm scripts that can be run by submitting to the queue.
```
#!/bin/bash
#SBATCH --job-name=       
#SBATCH --nodes=1              
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=1        
#SBATCH --mem-per-cpu=8G         
#SBATCH --time=03:00:00 

####################################################################################

RUNNAME=""


####################################################################################
```

## Solver information (and citation)

The solvers implemented in `WAVE` solve the scalar wave equation in cylindrical coordinates using forward and backward Euler methods, Fourier spectral methods, and various iterative methods.

[1] .

[2] .
