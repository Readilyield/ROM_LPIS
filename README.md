# ROM_LPIS
## A fast ROM for Linear Parabolic Inverse Source problems

For comparision purpose, we implemented FEM+CG and ROM+CG pipelines.
- FEM/ROM is used to perform forward operations (computing the solution to the heat equations)
- And CG is used to perform backward operations (computing the reconstructed source term)

*Created by: Yuxuan Huang*

Required python packages:
numpy, scipy, pandas, matplotlib, **NGSolve**(for mesh initialization and FEM solver)\\
To download NGSolve: https://ngsolve.org/downloads

### - Data folder

This folder contains test data that stores the FEM and ROM evaluation results for different parameter sets and organized into csv files.
(data includes 1/mesh_stepsize, runtime, iteration number, etc.)

### - Letter-images folder

This folder contains images for true source terms, final time observations, and reconstructed final results via FEM/ROM.
The source terms are all letters.
Naming rule: <1/mesh_stepsize>_<letter or pattern name>_<type(source or observation or (reconstuciton)method>

### - Utils folder

This folder contains all the utility functions for the numerical methods, input (image) conversion, and plotting.

### - animations folder

This folder contains some animated processes (forward and backward).

### - non-Letters folder

This folder contains images for true source terms, final time observations, and reconstructed final results via FEM/ROM.
The source terms are all non-letter patterns.
Naming rule: <pattern name>_<type(source or observation or (reconstuciton)method><1/mesh_stepsize>