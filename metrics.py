import matplotlib.pyplot as plt
metrics_folder = 'metrics_folder'
# Plot for iteration discrepacy heatcr and heatrn. Homogenous heat equation

# HEATCR
# u0.interpolate(cos(pi*x)*cos(2*pi*y))
# Running with 1 MPI processes. 1 in time, each with 1 in space
# DOF's space: 289, DOF's time: 8, DOF's total: 2312 
iterations_heatcr = [3.384274358890e+01, 8.615874001601e+00, 2.228401505753e+00, 2.352928938850e-01, 5.137065255786e-02, 1.207484855744e-02, 1.838000249421e-03, 2.740688564933e-04, 3.308491099079e-05, 2.714462870439e-06, 2.889102775772e-07, 2.899206464803e-08, 3.073932680088e-09, 2.211437011077e-10, 1.651623120048e-11]
iterations_heatrn = [1.692137179445e-02, 1.265052357290e-03, 4.258890878192e-05, 1.749097269470e-06, 4.859501816399e-08, 1.003723945605e-09, 1.844878813494e-11, 4.150687037120e-13]
# Add linear scaling
iterations_linear = [3.3e+01,3.3e+00,3.3e-01,3.3e-02,3.3e-03,3.3e-04,3.3e-05,3.3e-06,3.3e-07,3.3e-08,3.3e-09,3.3e-10,3.3e-11,3.3e-12]
iterations_linear_rn = [1.7e-02,1.7e-03,1.7e-04,1.7e-05,1.7e-06,1.7e-07,1.7e-08,1.7e-09,1.7e-10,1.7e-11,1.7e-12]

plt.figure(figsize=(10, 6))
plt.plot(iterations_heatcr, label='HeatCR', marker='o')
plt.plot(iterations_heatrn, label='HeatRN', marker='x')
plt.plot(iterations_linear, label='Linear Scaling', linestyle='--', color='gray')
plt.plot(iterations_linear_rn, label='Linear Scaling RN', linestyle='--', color='gray')
plt.xlabel('Iteration')
plt.ylabel('Residual Norm')
plt.title('Iteration Discrepancy between HeatCR and HeatRN. N_x = 8, Mref=1, N_t = 8. Space degree = 2')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig(f'{metrics_folder}/iteration_discrepancy_heatcr_heatrn.png', dpi=300)
plt.close()

######################################################
# WITHOUT MULTIGRID
no_mg_iterations_heatcr = [3.384274358890e+01, 1.341538937933e+01, 7.991286754886e+00, 4.860170720874e+00, 2.128465856305e+00, 7.418582609304e-01, 2.014208187073e-01, 5.113069526031e-02, 1.826727542476e-02, 5.351403789540e-03, 1.911052048752e-03, 4.818317960468e-04, 1.511859590995e-04, 2.952762171532e-05, 1.104054084101e-05, 1.803917471251e-06, 5.476508789794e-07, 1.362171911236e-07, 2.561310991637e-08, 8.819007838064e-09, 1.425448507142e-09, 4.791541580037e-10, 7.394629168839e-11, 1.717593939230e-11]
no_mg_iterations_heatrn = [1.692692132002e-02, 9.730647777122e-03, 6.677576022177e-03, 4.498401534060e-03, 2.075274586156e-03, 7.778490016051e-04, 2.166431976716e-04, 5.357585417464e-05, 1.921469395003e-05, 5.579525055385e-06, 2.130178106935e-06, 5.257384847684e-07, 1.696430288343e-07, 3.741942683668e-08, 1.357054050353e-08, 2.441463760489e-09, 7.761609140289e-10, 2.001715285104e-10, 4.661071711431e-11, 1.651282342034e-11, 3.648924877715e-12, 1.059564030107e-12, 2.385051052983e-13]

plt.figure(figsize=(10, 6))
plt.plot(no_mg_iterations_heatcr, label='HeatCR without MG', marker='o')
plt.plot(no_mg_iterations_heatrn, label='HeatRN without MG', marker='x')
plt.xlabel('Iteration')
plt.ylabel('Residual Norm')
plt.title('Iteration Discrepancy without Multigrid for HeatCR and HeatRN')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig(f'{metrics_folder}/iteration_discrepancy_no_mg_heatcr_heatrn.png', dpi=300)
plt.close()

shifted_no_mg_iterations_heatcr = [x * 1e-03 for x in no_mg_iterations_heatcr]

plt.figure(figsize=(10, 6))
plt.plot(shifted_no_mg_iterations_heatcr, label='HeatCR without MG (shifted)', marker='o')
plt.plot(no_mg_iterations_heatrn, label='HeatRN without MG', marker='x')
plt.xlabel('Iteration')
plt.ylabel('Residual Norm')
plt.title('Iteration Discrepancy without Multigrid for HeatCR and HeatRN (Shifted). Both one processor')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig(f'{metrics_folder}/iteration_discrepancy_no_mg_heatcr_heatrn_shifted.png', dpi=300)
plt.close()

############################ SAME AS ABOVE BUT 8 PROCS in space
eight_procs_iterations_heatcr = [3.384274358890e+01, 1.341538937933e+01, 7.991286754886e+00, 4.860170720874e+00, 2.128465856305e+00, 7.418582609304e-01, 2.014208187073e-01, 5.113069526031e-02, 1.826727542476e-02, 5.351403789540e-03, 1.911052048751e-03, 4.818317960468e-04, 1.511859590999e-04, 2.952762171542e-05, 1.104054084106e-05, 1.803917471188e-06, 5.476508788553e-07, 1.362171912324e-07, 2.561310994710e-08, 8.819007761017e-09, 1.425448577604e-09, 4.791541808827e-10, 7.394625543044e-11, 1.717600396658e-11]
eight_procs_iterations_heatrn = [1.692692132218e-02, 9.730647777305e-03, 6.677576022357e-03, 4.498401534494e-03, 2.075274586554e-03, 7.778490018097e-04, 2.166431977216e-04, 5.357585419297e-05, 1.921469395478e-05, 5.579525054522e-06, 2.130178105934e-06, 5.257384846237e-07, 1.696430287621e-07, 3.741942679659e-08, 1.357054048049e-08, 2.441463756026e-09, 7.761609115510e-10, 2.001715282454e-10, 4.661071710285e-11, 1.651282344326e-11, 3.648924850017e-12, 1.059564038099e-12, 2.385051170109e-13]

plt.figure(figsize=(10, 6))
plt.plot(eight_procs_iterations_heatcr, label='HeatCR with 8 Procs', marker='o')
plt.plot(eight_procs_iterations_heatrn, label='HeatRN with 8 Procs', marker='x')
plt.xlabel('Iteration')
plt.ylabel('Residual Norm')
plt.title('Iteration Discrepancy for HeatCR and HeatRN with 8 Procs in Space')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig(f'{metrics_folder}/iteration_discrepancy_heatcr_heatrn_8_procs.png', dpi=300)
plt.close()

shifted_eight_procs_iterations_heatcr = [x * 1e-03 for x in eight_procs_iterations_heatcr]

plt.figure(figsize=(10, 6))
plt.plot(shifted_eight_procs_iterations_heatcr, label='HeatCR with 8 Procs (shifted)', marker='o')
plt.plot(eight_procs_iterations_heatrn, label='HeatRN with 8 Procs', marker='x')
plt.xlabel('Iteration')
plt.ylabel('Residual Norm')
plt.title('Iteration Discrepancy for HeatCR and HeatRN with 8 Procs in Space (Shifted)')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig(f'{metrics_folder}/iteration_discrepancy_heatcr_heatrn_8_procs_shifted.png', dpi=300)
plt.close()

# SAME AS ABOVE BUT CR HAS 4 in time 4 each with 2 in space
#     0 KSP unpreconditioned resid norm 3.384274358890e+01 true resid norm 3.384274358890e+01 ||r(i)||/||b|| 1.000000000000e+00
#     1 KSP unpreconditioned resid norm 1.341538937933e+01 true resid norm 1.341538937933e+01 ||r(i)||/||b|| 3.964037178041e-01
#     2 KSP unpreconditioned resid norm 7.991286754886e+00 true resid norm 7.991286754886e+00 ||r(i)||/||b|| 2.361299914675e-01
#     3 KSP unpreconditioned resid norm 4.860170720874e+00 true resid norm 4.860170720874e+00 ||r(i)||/||b|| 1.436104229584e-01
#     4 KSP unpreconditioned resid norm 2.128465856305e+00 true resid norm 2.128465856305e+00 ||r(i)||/||b|| 6.289282813947e-02
#     5 KSP unpreconditioned resid norm 7.418582609304e-01 true resid norm 7.418582609304e-01 ||r(i)||/||b|| 2.192074820948e-02
#     6 KSP unpreconditioned resid norm 2.014208187073e-01 true resid norm 2.014208187073e-01 ||r(i)||/||b|| 5.951669319546e-03
#     7 KSP unpreconditioned resid norm 5.113069526031e-02 true resid norm 5.113069526031e-02 ||r(i)||/||b|| 1.510831860484e-03
#     8 KSP unpreconditioned resid norm 1.826727542476e-02 true resid norm 1.826727542476e-02 ||r(i)||/||b|| 5.397693415953e-04
#     9 KSP unpreconditioned resid norm 5.351403789540e-03 true resid norm 5.351403789540e-03 ||r(i)||/||b|| 1.581255897732e-04
#    10 KSP unpreconditioned resid norm 1.911052048751e-03 true resid norm 1.911052048751e-03 ||r(i)||/||b|| 5.646859108012e-05
#    11 KSP unpreconditioned resid norm 4.818317960468e-04 true resid norm 4.818317960467e-04 ||r(i)||/||b|| 1.423737395229e-05
#    12 KSP unpreconditioned resid norm 1.511859591001e-04 true resid norm 1.511859591003e-04 ||r(i)||/||b|| 4.467307997745e-06
#    13 KSP unpreconditioned resid norm 2.952762171544e-05 true resid norm 2.952762171508e-05 ||r(i)||/||b|| 8.724949157126e-07
#    14 KSP unpreconditioned resid norm 1.104054084091e-05 true resid norm 1.104054084118e-05 ||r(i)||/||b|| 3.262306677996e-07
#    15 KSP unpreconditioned resid norm 1.803917471122e-06 true resid norm 1.803917471497e-06 ||r(i)||/||b|| 5.330293233343e-08
#    16 KSP unpreconditioned resid norm 5.476508788381e-07 true resid norm 5.476508785286e-07 ||r(i)||/||b|| 1.618222462047e-08
#    17 KSP unpreconditioned resid norm 1.362171912887e-07 true resid norm 1.362171908862e-07 ||r(i)||/||b|| 4.025004371421e-09
#    18 KSP unpreconditioned resid norm 2.561310997672e-08 true resid norm 2.561311016349e-08 ||r(i)||/||b|| 7.568272381997e-10
#    19 KSP unpreconditioned resid norm 8.819007685775e-09 true resid norm 8.819008569850e-09 ||r(i)||/||b|| 2.605878730453e-10
#    20 KSP unpreconditioned resid norm 1.425448553941e-09 true resid norm 1.425449246086e-09 ||r(i)||/||b|| 4.211978979605e-11
#    21 KSP unpreconditioned resid norm 4.791542074324e-10 true resid norm 4.791542517938e-10 ||r(i)||/||b|| 1.415825671861e-11
#    22 KSP unpreconditioned resid norm 7.394625378876e-11 true resid norm 7.394649248414e-11 ||r(i)||/||b|| 2.185002888134e-12
#    23 KSP unpreconditioned resid norm 1.717599908920e-11 true resid norm 1.717601860071e-11 ||r(i)||/||b|| 5.075244137813e-13

four_time_two_space_iterations_heatcr = [
    3.384274358890e+01, 1.341538937933e+01, 7.991286754886e+00, 4.860170720874e+00,
    2.128465856305e+00, 7.418582609304e-01, 2.014208187073e-01, 5.113069526031e-02,
    1.826727542476e-02, 5.351403789540e-03, 1.911052048751e-03, 4.818317960468e-04,
    1.511859591001e-04, 2.952762171544e-05, 1.104054084091e-05, 1.803917471122e-06,
    5.476508788381e-07, 1.362171912887e-07, 2.561310997672e-08, 8.819007685775e-09,
    1.425448553941e-09, 4.791542074324e-10, 7.394625378876e-11, 1.717599908920e-11
]

plt.figure(figsize=(10, 6))
plt.plot(four_time_two_space_iterations_heatcr, label='HeatCR with 4 in time, 2 in space', marker='o')
plt.plot(eight_procs_iterations_heatrn, label='HeatRN with 8 Procs', marker='x')
plt.xlabel('Iteration')
plt.ylabel('Residual Norm')
plt.title('Iteration Discrepancy for HeatCR with 4 in time, 2 in space and HeatRN with 8 Procs')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig(f'{metrics_folder}/iteration_discrepancy_heatcr_4_time_2_space_heatrn_8_procs.png', dpi=300)
plt.close()