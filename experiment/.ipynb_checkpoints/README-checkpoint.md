# Experiments Overview

This repository contains all experiments corresponding to Section 6 (Evaluation) of the paper. Each subsection has its own directory, and each directory contains:

* Required scripts
* Configuration files
* A shell script to run the experiment

To run each experiment, navigate to the corresponding directory and execute the script as shown below.

6.1 – Baselines under Runtime Workload Variations
cd 6.1/
./run_experiment.sh

6.2 – Different Set Points
cd 6.2/
./run.sh

6.3
cd 6.3/
./experiments.sh

6.4
cd 6.4/
./run.sh

# To run all experiments automatically
./run_allexperiments.sh

All generated figures are stored in the Figures/ directory.

Figure mapping to the paper:

* 6_1.pdf corresponds to Figure 3 in the paper
* 6_2.pdf corresponds to Figure 4 in the paper
* 6_3.pdf corresponds to Figure 5 in the paper
* 6_4.pdf corresponds to Figure 6 in the paper