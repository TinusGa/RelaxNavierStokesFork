import matplotlib.pyplot as plt
headers = ["N","Nt","Nx","P","Pt","Px","Setup time (s)","Solve time (s)","Iterations","startup","forward_reduction","interface_solve","forward_substitution","solution_filling"]

filepath = "metrics_folder/1728_heat_run_results.csv"
solve_times = []
pts = []
# Split csv column wise
with open(filepath, 'r') as file:
    lines = file.readlines()
    data = [line.strip().split(',') for line in lines]
    data = [list(map(float, row)) for row in data]  # Skip header and convert to float
    solve_times = [row[7] for row in data]
    pts = [row[4] for row in data]
    startup_times = [row[9] for row in data]
    startup_times = [startup_time*solve_time*0.01 for startup_time, solve_time in zip(startup_times, solve_times)]
    forward_reduction_times = [row[10] for row in data]
    forward_reduction_times = [forward_reduction_time*solve_time*0.01 for forward_reduction_time, solve_time in zip(forward_reduction_times, solve_times)]
    interface_solve_times = [row[11] for row in data]
    interface_solve_times = [interface_solve_time*solve_time*0.01 for interface_solve_time, solve_time in zip(interface_solve_times, solve_times)]
    forward_substitution_times = [row[12] for row in data]
    forward_substitution_times = [forward_substitution_time*solve_time*0.01 for forward_substitution_time, solve_time in zip(forward_substitution_times, solve_times)]
    solution_filling_times = [row[13] for row in data]
    solution_filling_times = [solution_filling_time*solve_time*0.01 for solution_filling_time, solve_time in zip(solution_filling_times, solve_times)]

plt.figure(figsize=(10, 6))
plt.plot(pts,solve_times, label='o', marker='o')
plt.ylabel('Wall clock time (s)')
plt.xlabel('Number of time-slices (Pt)')
plt.xticks(pts)
plt.title('Strong scaling for N = 1728, Px = 1')
plt.legend()
plt.grid(True)
plt.savefig(f'metrics_folder/1728strongscaling.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(pts, startup_times, label='Startup', marker='o')
plt.plot(pts, forward_reduction_times, label='Forward Reduction', marker='o')
plt.plot(pts, interface_solve_times, label='Interface Solve', marker='o')
plt.plot(pts, forward_substitution_times, label='Forward Substitution', marker='o')
plt.plot(pts, solution_filling_times, label='Solution Filling', marker='o')
plt.ylabel('Wall clock time (s)')
plt.xlabel('Number of time-slices (Pt)')
plt.xticks(pts)
plt.title('Strong scaling breakdown for N = 1728, Px = 1')
plt.legend()
plt.grid(True)
plt.savefig(f'metrics_folder/1728strongscalingbreakdown.png', dpi=300)
plt.close()



filepath = "metrics_folder/1728_px_heat_run_results.csv"
solve_times = []
pxs = []
# Split csv column wise
with open(filepath, 'r') as file:
    lines = file.readlines()
    data = [line.strip().split(',') for line in lines]
    data = [list(map(float, row)) for row in data]  # Skip header and convert to float
    solve_times = [row[7] for row in data]
    pxs = [row[5] for row in data]
    startup_times = [row[9] for row in data]
    startup_times = [startup_time*solve_time*0.01 for startup_time, solve_time in zip(startup_times, solve_times)]
    forward_reduction_times = [row[10] for row in data]
    forward_reduction_times = [forward_reduction_time*solve_time*0.01 for forward_reduction_time, solve_time in zip(forward_reduction_times, solve_times)]
    interface_solve_times = [row[11] for row in data]
    interface_solve_times = [interface_solve_time*solve_time*0.01 for interface_solve_time, solve_time in zip(interface_solve_times, solve_times)]
    forward_substitution_times = [row[12] for row in data]
    forward_substitution_times = [forward_substitution_time*solve_time*0.01 for forward_substitution_time, solve_time in zip(forward_substitution_times, solve_times)]
    solution_filling_times = [row[13] for row in data]
    solution_filling_times = [solution_filling_time*solve_time*0.01 for solution_filling_time, solve_time in zip(solution_filling_times, solve_times)]

plt.figure(figsize=(10, 6))
plt.plot(pxs,solve_times, label='o', marker='o')
plt.ylabel('Wall clock time (s)')
plt.xlabel('Number of spatial processors (Px)')
plt.xticks(pxs)
plt.title('Strong scaling for N = 1728, Pt = 1')
plt.legend()
plt.grid(True)
plt.savefig(f'metrics_folder/1728pxstrongscaling.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(pts, startup_times, label='Startup', marker='o')
plt.plot(pts, forward_reduction_times, label='Forward Reduction', marker='o')
plt.plot(pts, interface_solve_times, label='Interface Solve', marker='o')
plt.plot(pts, forward_substitution_times, label='Forward Substitution', marker='o')
plt.plot(pts, solution_filling_times, label='Solution Filling', marker='o')
plt.ylabel('Wall clock time (s)')
plt.xlabel('Number of time-slices (Pt)')
plt.xticks(pts)
plt.title('Strong scaling breakdown for N = 1728, Px = 1')
plt.legend()
plt.grid(True)
plt.savefig(f'metrics_folder/1728pxstrongscalingbreakdown.png', dpi=300)
plt.close()

filepath = "metrics_folder/49152_heat_run_results.csv"
solve_times = []
pxs = []
# Split csv column wise
with open(filepath, 'r') as file:
    lines = file.readlines()
    data = [line.strip().split(',') for line in lines]
    data = [list(map(float, row)) for row in data]  # Skip header and convert to float
    solve_times = [row[7] for row in data]
    pxs = [row[5] for row in data]
    startup_times = [row[9] for row in data]
    startup_times = [startup_time*solve_time*0.01 for startup_time, solve_time in zip(startup_times, solve_times)]
    forward_reduction_times = [row[10] for row in data]
    forward_reduction_times = [forward_reduction_time*solve_time*0.01 for forward_reduction_time, solve_time in zip(forward_reduction_times, solve_times)]
    interface_solve_times = [row[11] for row in data]
    interface_solve_times = [interface_solve_time*solve_time*0.01 for interface_solve_time, solve_time in zip(interface_solve_times, solve_times)]
    forward_substitution_times = [row[12] for row in data]
    forward_substitution_times = [forward_substitution_time*solve_time*0.01 for forward_substitution_time, solve_time in zip(forward_substitution_times, solve_times)]
    solution_filling_times = [row[13] for row in data]
    solution_filling_times = [solution_filling_time*solve_time*0.01 for solution_filling_time, solve_time in zip(solution_filling_times, solve_times)]

plt.figure(figsize=(10, 6))
plt.plot(pxs,solve_times, label='o', marker='o')
plt.ylabel('Wall clock time (s)')
plt.xlabel('Number of spatial processors (Px)')
plt.xticks(pxs)
plt.title('Strong scaling for N = 1728, Pt = 1')
plt.legend()
plt.grid(True)
plt.savefig(f'metrics_folder/49152strongscaling.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(pts, startup_times, label='Startup', marker='o')
plt.plot(pts, forward_reduction_times, label='Forward Reduction', marker='o')
plt.plot(pts, interface_solve_times, label='Interface Solve', marker='o')
plt.plot(pts, forward_substitution_times, label='Forward Substitution', marker='o')
plt.plot(pts, solution_filling_times, label='Solution Filling', marker='o')
plt.ylabel('Wall clock time (s)')
plt.xlabel('Number of time-slices (Pt)')
plt.xticks(pts)
plt.title('Strong scaling breakdown for N = 1728, Px = 1')
plt.legend()
plt.grid(True)
plt.savefig(f'metrics_folder/49152strongscalingbreakdown.png', dpi=300)
plt.close()