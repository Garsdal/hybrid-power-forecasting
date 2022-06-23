#!/bin/sh
### General options

### -- specify queue --
#BSUB -q hpc

### -- set the job Name --
#BSUB -J swp_HPP2

### -- ask for number of cores (default: 1) --
#BSUB -n 1
#BSUB -R "span[hosts=1]"

### -- Select the resources: 1 gpu in exclusive process mode -- 
##BSUB -cpu "num=8"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 48:00

# request 5GB of system-memory
##BSUB -R "select[gpu32gb]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "select[model == XeonGold6226R]"

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u garsdal@live.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --

#BSUB -o hpc/sweep_models-%J.out
#BSUB -e hpc/sweep_models-%J.err
# -- end of LSF options --

# Load the cuda module
module load python3/3.8.2

# Go to directory
cd /zhome/20/b/127753/forecasting/Hybrid_Forecasts/hybrid_forecasts # MARCUS GARSDAL

# Load venv
source env_msc/bin/activate

# (1.) start a sweep in the cmd
#wandb sweep --entity [entity-name] --project [project-name] [test.yaml]

# (2.) change the line below to the correct agent and submit
#bsub < src/sweeps/submit-sweep.sh