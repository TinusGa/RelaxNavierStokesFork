"""
Orchestrator script for running multiple heat equation simulations.

This script loops through a defined parameter space, launches each simulation
using `mpiexec` and `subprocess`, waits for it to complete, then gathers
the results from both the simulation's output file and the PETSc log file.
All aggregated data is written to a master CSV file.
"""

import sys
import subprocess
from CyclicReduction import analyze_profiling_data
import subprocess
import os
import csv
import shutil

def run_and_collect():
    """
    Manages the process of running simulations and collecting data.
    """
    # Run for only temporal processors
    ranks_list = [1,2,3,4,6,8,12,16,24] # 48, 96, 144, 192, 240 are all divisible by the ranks list
    Pt_list = [1,2,3,4,6,8,12,16,24]
    # Pt_list = [1]
    Nslice_list = [48,24,16,12,8,6,4,3,2]
    # Nslice_list = [48]
    M_list = [31]
    space_degree_list = [1]

    dt = 0.001
    theta = 1.0

    output_csv_file = "metrics_folder/49152_heat_run_results.csv"
    log_file_path = "metrics_folder/heat_run_log.txt"

    def executable(ranks, Pt, M, Nslice, space_degree):
        if ranks % Pt != 0:
            raise ValueError(f"Number of ranks {ranks} must be divisible by Pt {Pt}.")
        
        python_executable = sys.executable
        mpiexec_path = shutil.which("mpiexec")

        # Build the command for subprocess
        command = [
            mpiexec_path,"-n", str(ranks),
            python_executable, "heat_run.py",
            "--Nslice", str(Nslice),
            "--Pt", str(Pt),
            "--M", str(M),
            "--dt", str(dt),
            "--theta", str(theta),
            "--space_degree", str(space_degree),
            "-log_view", f":{log_file_path}:ascii_flamegraph"
            # "-log_view", f":foo.txt:ascii_flamegraph"
        ]
        
        try:
            # Execute the simulation command
            command_str = " ".join(command)
            print(f"Executing: {command_str}")
            current_env = os.environ.copy()
            process = subprocess.Popen(command_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True,env=current_env)
            lines = process.stdout.readlines()
            stdout, stderr = process.communicate()

            headers = ["N:", "Nt:", "Nx:",
                        "P:", "Pt:", "Px:",
                        "Setup time (s):", "Solve time (s):", "Iterations:"
                        ]
            results = [0]*len(headers)
            
            
            for line in lines:
                for i, header in enumerate(headers):
                    if line.startswith(header):
                        value = line.split(': ')[1].strip()
                        results[i] = float(value)

            flamegraph_search_terms = ["startup", "forward_reduction", "interface_solve",
                            "forward_substitution", "solution_filling",]
            
            flamegraph_timings = analyze_profiling_data(log_file_path, flamegraph_search_terms)

            header = ["N","Nt","Nx",
                        "P","Pt","Px",
                        "Setup time (s)","Solve time (s)","Iterations",
                        "startup","forward_reduction","interface_solve",
                        "forward_substitution","solution_filling"]
            
            
            with open(output_csv_file, "a", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([*results,*flamegraph_timings])

        except subprocess.CalledProcessError as e:
            print(f"!!! ERROR executing run: {e}")
            print("--- STDOUT ---")
            print(e.stdout)
            print("--- STDERR ---")
            print(e.stderr)
        except FileNotFoundError as e:
            # print(f"Error: Could not find the file(s) at {output_csv_file} or {log_file_path}.")
            print(e)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


    for (ranks,Pt,Nslice) in zip(ranks_list, Pt_list, Nslice_list):
        for M in M_list:    
            for space_degree in space_degree_list:
                executable(ranks, Pt, M, Nslice, space_degree)

    # for ranks in ranks_list:
    #     for Pt in Pt_list:
    #         for Nslice in Nslice_list:
    #             for M in M_list:    
    #                 for space_degree in space_degree_list:
    #                     executable(ranks, Pt, M, Nslice, space_degree)
                    
                
    print("\nAll simulations finished.")

if __name__ == "__main__":
    run_and_collect()