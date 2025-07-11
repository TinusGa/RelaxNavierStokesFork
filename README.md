# A Parallel-in-Time Solution for Parabolic PDEs with Multigrid Waveform Relaxation

This repository contains the source code and implementation for the master's thesis, *"A Parallel-in-Time Solution for Parabolic PDEs with Multigrid Waveform Relaxation."*

This project is a fork of the **RelaxNavierStokes** repository, which can be found [here](https://github.com/JamesJackaman/RelaxNavierStokes.git).

---

## Dependencies

This project relies on **Firedrake** and **asQ**.

* **Firedrake:** An installation guide is available at the [Firedrake project website](https://www.firedrakeproject.org/install.html).
* **asQ:** The official GitHub repository can be found [here](https://github.com/firedrakeproject/asQ.git).

---

## 📂 Repository Structure

This repository contains several key directories.

* `CyclicReduction/`: This directory holds the core functionality of the project. The most recent and updated implementation of the preconditioner can be found in `preconditioner3.py`.
* `asQ/`: This is a cloned version of the asQ repository. Modifications have been made to the `AllAtOnceJacobian` for explicit assembly of Jacobians required by the implemented methodologies.
* `RelaxNavierStokes/`: A cloned version of the original repository, with no modifications.

The remaining files are either scripts used for development testing or for running the experiments detailed in the thesis.

---

## 🚀 Running Experiments

This section outlines how to run the experiments for different partial differential equations.

### Heat Equation

To test the preconditioner from `preconditioner3.py` on the homogeneous heat equation, run the `heat_test.py` script. This script will execute `heat_run.py` with various processor configurations and problem parameters.

For a direct comparison between different implementations, you can use the following scripts:
* **This implementation:** `heatcr.py`
* **RelaxNavierStokes:** `heatrn.py`
* **asQ:** `heatasQ.py`

### Fisher Equation

For the nonlinear KPP-Fisher equation, you can compare this project's implementation with the RelaxNavierStokes version using these files:
* **This implementation:** `fishercr.py`
* **RelaxNavierStokes:** `fisherrn.py`

### Multigrid

To compare the performance of multigrid with block Jacobi preconditioners, run the `multigrid_test.py` script.

---

## 📊 Visualization

Most experiment files can be run with a plot parameter set to `True`. This will save the simulation output in the `ParaView/` directory. The results can then be visualized using the **ParaView** software.
