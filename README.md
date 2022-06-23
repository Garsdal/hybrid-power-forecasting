# Hybrid_Forecasts
==============================

This project contain all work and results related to my Master Thesis named 'Hybrid Power Forecasting'. 
In this repository the data/ folder has not been included due to a NDA. This results in majority of the scripts not working. I will provide public test data in the near future, such that models and methods implemented can be tested.

The repository still contains predictions made for all models implemented. These predictions are included in different notebooks, but 'aggregate_model_results.ipyndb' can be run without any data available. Create a new environment, install the requirements.txt and you should be able to run the notebook to see the forecasts.

# Notebook overview

### Deterministic
All notebooks are located in the /notebooks folder.
- aggregate_model_results.ipyndb: Notebook which produces lead time plots for all deterministic baseline models (included).
- comparison_pre_post_tuning.ipyndb: Notebook which produces a skillscore table again the LR model comparing a final models before and after tuning (included).
- create_result_tables.ipyndb: Notebook which produces a table with a skillscore for all final models compared to the baseline. Also produces a table with information about all plants in the study, and correlation of their data (included).
- create_params_tables.ipyndb: Notebook which produces a table with all final parameters from hyperparameter tuning(included).
- data_cleaning.ipyndb: Notebook which produces power curves before and after data processing for all plants (included). Will require the conwx-anomaly-detection package to run.
- export_deterministic_results.ipyndb: Notebook which produces time series plots of available data, pairplots of meteo features, lead time statistics and lead time plots (included). Also produces ridge-plots and violin plots although these might not be included in the report.
- reconcilation_analysis.ipyndb: Notebook which produces a lead time plot for different aggregation methods and a table with nRMSE metric for comparison (not included in the final analysis).

### Probabilistic
All notebooks are located in the /notebooks/probabilistic folder.
/Old_notebooks and /Test_notebooks are not a part of the final results.
- create_PICP_tables.ipyndb: Notebook which was used to create PICP tables (included).
- create_uncertain_inputs.ipyndb: Notebook which was used to create all final results for uncertain inputs (included).
- forecast_errors_monte_carlo.ipyndb: Does not produce any final results.
- simulation_length_check.ipyndb: Notebook which produces plots showing the number of simulations before average standard deviation across all horizons converges when monte carlo simulating the forecasts (included).
- uncertainty_propagation.ipyndb: Used to produce copula and noise generation plots. The remaining results in here are exported from /src (this notework was primarily used for code development). 

# /src code overview
All scripts should be run remotely on a HPC due to long computational time. The code will have to be modified to include a correct output path since it currently points to storage on the DTU HPC! 

The data/ folders has been included in this private repository. This means that the code can be run if the output paths are changed to either local or a different DTU HPC location.  

## Main scripts for model training/prediction 
- train_models.py: The main script used to train models. The script will require data in 'data/processed/...'. It will try to save model files to 'work3/s174440/...'.
- predict_models.py: The main script used to make predictions for the trained models. The script will require data in 'data/processed/...'. The script will try to point to model files in 'work3/s174440/...'. The script will save predictions as .csv in 'reports/results/{plant}/deterministic/predictions/...'.

### Helper scripts for model training/prediction
- utils.py: A script containing many utility functions required to run training and predictions.
- models/deterministic/models.py: A script containing all model classes. Also includes methods for saving/loading/predictions for each model implemented.
- models/predict_functions.py: A script containing functions for step-ahead predictions, recursive updated predictions, full sequence predictions. All of these are used in the models.py file .
- features/build_features.py: Script which takes data in 'data/processed/...' and builds input features depending on the specified technology, model and targets. This script is essential to almost all steps of training/prediction.

### Helper scripts for HPC scheduling
- submit-predict.sh: Bash script for scheduling predictions on the HPC.
- submit-train.sh: Bash script for scheduling training on the HPC.
- submit-uncertainty.sh: Bash script for scheduling Monte Carlo simulation on the HPC.
- sweeps/submit-sweep.sh: Bash script for scheduling WanDB sweeps on the HPC.

### Helper scripts for handling raw data
- make_conwx_dataframe.py: A script which combines wind/solar raw data from ConWX to hybrid format.
- make_hybrid_dataframe.py: A script which combines wind/solar raw data from Nazeerabad to hybrid format.

## Main scripts for hyperparameter tuning
NB. these scripts requires a WanDB key 'wandb_key.json' containing an API key used to connect to a WanDB project to log results.
- sweeps/LGB_sweep.py: Script which starts a WanDB sweep for LGB models based on parameters in 'LGB_sweep.yaml'.
- sweeps/LSTM_sweep.py: Script which starts a WanDB sweep for LSTM models based on parameters in 'LSTM_sweep.yaml'.
- sweeps/RF_sweeps.py: Script which starts a WanDB sweep for RF models based on parameters in 'RF_sweep.yaml'.

## Main scripts for uncertainty propagation
- models/probabilistic/uncertainty_propagation.py: The main script used for running Monte Carlo simulations. The script uses the deterministic code described above, but also important functions 'create_pertubed_inputs()' and 'gen_gauss_copula()' in utils.py.
- models/probabilistic/simulation_length_check.py: The script which was used to carry out checks for simulation length.
- models/probabilistic/match_forecast_distribution_optimization.py: A script which was used to try and match the uncertainty level on inputs. The results did not end up in the final analysis.

### Helper scripts for visualization/evaluation of results
- visualization/visualize.py: Script containing all plots which are generated through scripts run on the HPC. Some of the functions are also used in the notebooks described above.
- visualization/evaluate.py: Script containing all evaluation metrics used when comparing predictions to true values.

# Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>