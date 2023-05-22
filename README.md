# ROM_LPIS
## A fast ROM for Linear Parabolic Inverse Source problems

For comparision purpose, we implemented FEM+CG and ROM+CG pipelines.
- FEM/ROM is used to perform forward operations (computing the solution to the heat equations)
- And CG is used to perform backward operations (computing the reconstructed source term)

*Created by: Yuxuan Huang*

Required python packages:
numpy, scipy, pandas, matplotlib, **NGSolve**(for mesh initialization and FEM solver)
<br>
To download NGSolve: https://ngsolve.org/downloads

### - Data folder
This folder contains test data that stores the FEM and ROM evaluation results for different parameter sets and organized into csv files.
(data includes 1/mesh_stepsize, runtime, iteration number, etc.)

### - Letter-images folder
This folder contains images for true source terms, final time observations, and reconstructed final results via FEM/ROM.
<br>
The source terms are all letters.
<br>
*Naming rule: [1/mesh\_stepsize]\_[letter or pattern name]\_[type(source or observation or (reconstuciton)method]*

### - Letters folder
This folder contains primitive square images with basic Arial font letters of different (pixel)sizes.
<br>
They are used as source terms.
<br>
*Naming rule: [letter name]\_[image side length:number of pixels]*

### - Utils folder
This folder contains all the utility functions for the numerical methods, input (image) conversion, and plotting.

### - animations folder
This folder contains some animated processes (forward and backward).

### - non-Letters folder
This folder contains primitive and processed images for true source terms, final time observations, and reconstructed final results via FEM/ROM.
<br>
The source terms are all non-letter patterns.
<br>
*Naming rule: [pattern name]\_[type(source or observation or (reconstuciton)method][1/mesh_stepsize]*

## Notebooks:
#### Single_Run
- this is the main testing notebook.
- conducts a single reconstruction process for the final observation data of a "unknown" source pattern using FEM+CG and/or ROM+CG.
#### Runtime_plot
- plots the runtime and/or iteration comparison for the FEM+CG and ROM+CG method.
#### Comparision_Error_plots
- plots L-curve graphs for FEM and ROM results and their differences in L_2 errors in terms of different hyper parameters.
#### Gen_Data_tocsv
- conducts a series of reconstruction trials over *one pattern* under different hyper parameter combination and organized the results into one csv file.
#### Prototype_[something]
- prototype notebooks to verify pipeline feasibility and conduct miscellaneous tests.


