# MERGO

This repository contains script which were used to build and train the MERGO model together with the scripts for evaluating the model's performance.

# Get Started

## Dependency

Please, install the following packages:

- numpy
- torch-1.8.0
- torchversion-0.9.0
- transformers-4.5.1
- tqdm

Secondly, install [diamond](https://github.com/bbuchfink/diamond) program on your system (diamond command should be available)

## Content

- ./data: the dataset with label and sequence, including CAFA3 and Homo datasets.
- ./feature_extract: extraction of protein sequences with deep semantic view features.

## Usage


### Step 1:

First, download the pretrained models for ESM2 and ProtT5. You can either refer to the original paper's code or use the scripts provided in the `feature_extract` directory of this repository to extract protein sequence features.

### Step 2:

Configure the `config.py` file by setting the paths to the feature files and specifying the necessary model hyperparameters.

### Step 3:

Run the `main.py` script to start training. The model's prediction results will be saved in the `prediction.pkl` file.

