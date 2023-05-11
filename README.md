# Reproducibility report

This is the code for the reproducibility of parts of the paper:
[_Pre-Train Your Loss: Easy Bayesian Transfer Learning with Informative Priors_](https://openreview.net/forum?id=ao30zaT3YL).

The code inside folder `BayesianTransferLearning` is copied from the authors GitHub, but adjusted for replicating the results.

## Installation

```bash
# using pip in a virtual environment
pip install -r requirements.txt

# using Conda
conda create --name <env_name> --file requirements.txt
conda activate <env_name>
```

## Downloading the data

1. Run `./BayesianTransferLearning/Prapare Data/oxford-102-flowers.py` to download the Oxford-102-Flowers data.
2. Run `downsampled_data/create_downsampled_folders.py` from the `downsampled_data` directory to create the subfolders with smaller data set sizes.
3. [Download priors](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/hsouri1_jh_edu/Et6cosMhV39CqI_k_rqpKMoBSJmJSXKkk_w3dn_VgEfr7w?e=RdNcKn) from the original authors and put them in to `priors` folder.

## Run experiments

There are three experiments we ran corresponding to python scripts:

1. The influence of low-dimensional rank on performance (`run_experiment_rank.py`)

2. The influence of prior scaling on performance (`run_experiment_scale.py`)

3. Comparison of Bayesian and non-Bayesian learning (`run_experiment_comparison.py`)

To run experiments simply run the corresponding scipt.
The results will accumulate in a text file (`results_*.txt`).

