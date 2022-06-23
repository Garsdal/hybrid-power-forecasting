#!/bin/sh
### General options

### -- specify queue --
#BSUB -q hpc

### -- set the job Name --
#BSUB -J Naz_LGB

### -- ask for number of cores (default: 1) --
#BSUB -n 1
#BSUB -R "span[hosts=1]"

### -- Select the resources: 1 gpu in exclusive process mode -- 
##BSUB -cpu "num=8"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 72:00

# request 5GB of system-memory
##BSUB -R "select[gpu32gb]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[model == XeonGold6226R]"

### -- set the email address --
### please uncomment the following line and put in your e-mail address,
### if you want to receive e-mail notifications on a non-default address
###BSUB -u garsdal@live.dk
### -- send notification at start --
###BSUB -B
### -- send notification at completion--
###BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --

#BSUB -o hpc/uncertainty-%J.out
#BSUB -e hpc/uncertainty-%J.err
# -- end of LSF options --

# Load the cuda module
module load python3/3.8.2

# Go to directory
cd /zhome/20/b/127753/forecasting/Hybrid_Forecasts/hybrid_forecasts # MARCUS GARSDAL

# Load venv
source env_msc/bin/activate

# Run uncertainty scripts

#python src/models/probabilistic/simulation_length_check.py # check how many simulations (n_sims) are required for the mean std across all horizons to converge (was tested with RF for HPP3) 
#python src/models/probabilistic/match_forecast_distribution_optimization.py --plant Nazeerabad --model LGB --n_sims 500 # optimize the max uncertainty for wind / solar

#python src/models/probabilistic/uncertainty_propagation.py --plant HPP1 --model LSTM --n_sims 200  --max_uncertainty_wind 0.295 --max_uncertainty_solar 0.35  # runs the uncertainty propagation for determined uncertainties
python src/models/probabilistic/uncertainty_propagation.py --plant Nazeerabad --model LGB --n_sims 200 --max_uncertainty_wind 0.295 --max_uncertainty_solar 0.35 --bias_corrected True
#python src/models/probabilistic/uncertainty_propagation.py --plant Nazeerabad --model LSTM --n_sims 2 --max_uncertainty_wind 0 --max_uncertainty_solar 0