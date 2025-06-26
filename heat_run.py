from firedrake import *
from firedrake.petsc import PETSc
from asQ import (
    create_ensemble,
    AllAtOnceFunction,
    AllAtOnceForm,
    AllAtOnceSolver,
    LinearSolver
)
from time import time
import warnings
warnings.simplefilter("ignore", FutureWarning)
import os 
import csv
import argparse

class ProblemParameters:
    def __init__(self):
        self.Nslice = 4
        self.Pt = 4
        self.M = 9
        self.dt = 0.001
        self.theta = 1.0 # Time-stepping parameter (1.0 for Backward Euler)
        self.space_degree = 2
        self.plot = False  # Whether to output results to ParaView
        self.write_metrics = False  # Whether to write metrics to file

def heat_run(parameters = ProblemParameters()):

    setup_start = time()

    # 1. ENSEMBLE and MESH setup
    time_partition = [parameters.Nslice] * parameters.Pt
    ensemble = create_ensemble(time_partition, comm=COMM_WORLD)

    distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
    mesh = UnitSquareMesh(nx=parameters.M, ny=parameters.M,
                                distribution_parameters=distribution_parameters,
                                comm=ensemble.comm)

    # 2. FUNCTION SPACES and BOUNDARY CONDITIONS
    space_element = FiniteElement("CG", triangle, parameters.space_degree)
    U = FunctionSpace(mesh, space_element)

    # I.C and B.C
    x, y = SpatialCoordinate(U.mesh())
    u0 = Function(U)

    # u0.interpolate(cos(pi*x)*cos(2*pi*y))
    # bcs = []

    u0.project(sin(0.25*pi*x)*cos(2*pi*y))
    bcs = [DirichletBC(U, 0, sub_domain=1)]

    # 3. VARIATIONAL FORMS
    def form_mass(u, v):
        return u*v*dx

    def form_function(u, v, t):
        return inner(grad(u), grad(v))*dx

    # 4. asQ SETUP
    aaofunc = AllAtOnceFunction(ensemble, time_partition, U)
    aaofunc.initial_condition.assign(u0)
    aaofunc.zero(zero_ics=False)

    aaoform = AllAtOnceForm(aaofunc, 
                            parameters.dt, 
                            parameters.theta, 
                            form_mass,
                            form_function, 
                            bcs=bcs)

    # 5. SOLVER SETUP FOR PRECONDITIONER

    solver_parameters = {
        'snes_type': 'ksponly',
        'mat_type': 'aij',
        'ksp_type': 'fgmres',
        # "ksp_monitor_true_residual": None,
        "ksp_max_it": 25,
        "ksp_atol": 1e-18,
        "ksp_rtol": 1e-18,
        'pc_type': 'python',
        'pc_python_type': 'CyclicReduction.CyclicReductionPC3',}

    solver = AllAtOnceSolver(aaoform, 
                            aaofunc, 
                            solver_parameters)

    setup_time = time()-setup_start


    processes = COMM_WORLD.size
    A,_ = solver.snes.ksp.getOperators()
    space_dofs = U.dim()
    time_dofs = sum(time_partition)
    total_dofs = time_dofs*space_dofs

    solve_start = time()
    solver.solve()
    solve_time = time() - solve_start
    iterations = solver.snes.getLinearSolveIterations()

    header = ["NT", "Nt", "Nx", 
              "P", "Pt", "Px", 
              "Setup Time (s)", "Solve Time (s)", "Iterations"]
    data_out = [total_dofs, time_dofs, space_dofs, 
                processes, parameters.Pt, processes//len(time_partition),
                setup_time, solve_time, iterations]


    return data_out

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parallel-in-time heat equation solver.")
    parser.add_argument('--Nslice', type=int, default=4, help='Number of timesteps per time-slice.')
    parser.add_argument('--Pt', type=int, default=4, help='Number of time-slices (temporal processors).')
    parser.add_argument('--M', type=int, default=10, help='Mesh size.')
    parser.add_argument('--dt', type=float, default=0.001, help='Timestep size.')
    parser.add_argument('--theta', type=float, default=1.0, help='Theta-method parameter (1.0 for BE).')
    parser.add_argument('--space_degree', type=int, default=1, help='Spatial polynomial degree.')
    parameters,_ = parser.parse_known_args()

    # Run simulation
    results = heat_run(parameters)

    PETSc.Sys.Print("Results:")
    PETSc.Sys.Print("N:",results[0])
    PETSc.Sys.Print("Nt:",results[1])
    PETSc.Sys.Print("Nx:",results[2])
    PETSc.Sys.Print("P:",results[3])
    PETSc.Sys.Print("Pt:",results[4])
    PETSc.Sys.Print("Px:",results[5])
    PETSc.Sys.Print("Setup time (s):",results[6])
    PETSc.Sys.Print("Solve time (s):",results[7])
    PETSc.Sys.Print("Iterations:",results[8])
