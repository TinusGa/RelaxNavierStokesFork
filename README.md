# A Parallel-in-Time Solution for Parabolic PDEs with Multigrid Waveform Relaxation

Code and implementations supplement to my master's thesis 'A Parallel-in-Time Solution for Parabolic PDEs with Multigrid Waveform Relaxation' (not yet public). 
This repository contains all code relevant to experiments and results therein. 

This repository is a fork from the RelaxNavierStokes repository found here: https://github.com/JamesJackaman/RelaxNavierStokes.git. 

## Installation
This repository depends on Firedrake and asQ. An installation guide for Firedrake can be found here: https://www.firedrakeproject.org/install.html. The asQ GitHub repository can be found here: https://github.com/firedrakeproject/asQ.git

## Repository Structure
The repository structure is not organized, and filenames can be misleading. In development, both the RelaxNavierStokes and asQ repositories were cloned into the current repository for efficient testing and can be found in the folders \texttt{asQ} and \texttt{RelaxNavierStokes}. Accompanying test and example files from these repositories are also included in this repo. 
No changes have been made to the RelaxNavierStokes repo, but changes have been made to the asQ repo, namely in the \texttt{AllAtOnceJacobian} for explicit assembly of Jacobians required for the methodologies of the implementation.

The core functionality of this repo is found under the folder \texttt{CyclicReduction}, with the most recent changes and updated functionality in \texttt{preconditioner3.py}. 

Remaining files are either test files for code development or experiments relevant to the master's thesis.

## Code
### Heat Equation
For the homogenous heat equation, the main test for the application of the preconditioner in \texttt{preconditioner3.py} can be run with the file \texttt{heat_test.py} for various configurations of processors and problem parameters. This will run the \texttt{heat_run.py} file for the provided configurations. 

Comparison between implementations made here and the one from RelaxNavierStokes, and asQ can be done with the files \texttt{heatcr.py}, \texttt{heatrn.py} and \texttt{heatasQ.py} repsectively. 

### Fisher equation
For the nonlinear KPP-Fisher equation, comparisons between the implementations here and the one from RelaxNavierStokes can be done with the files \texttt{fishercr.py} and \texttt{fisherrn.py} respectively

### Multigrid
Comparisons of multigrid with block Jacobi preconditioners can be done with the file \texttt{multigrid_test.py}

### Plots
Most files can be run with setting a plot parameter to True. This will save the simulation under the \texttt{ParaView} folder and can be visualized with the ParaView software. 

