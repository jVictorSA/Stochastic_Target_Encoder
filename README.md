# Stochastic_Target_Encoder
Repository for the Paper: **Stochastic Target Encoder - A new categorical feature encoding applied to urban data regression problems**

## Abstract
**Regression problems are often Machine Learning (ML) tasks found in real world, with most of its data in the tabular form, and many attributes being categorical. Most ML algorithms works only with numerical data, so encoding categorical attributes tends to be necessary, but most encoding methods don't use data properties, which can lead to poor model performance on high cardinality data. Target Encoding methods address this, but encode each attribute into a discrete set of numerical values of equal cardinality to the categorical attribute. We propose a Target Encoder that addresses both issues and achieves results comparable with the existing Target Encoders. We test our method against existing Encoders, showing the robust performance of our method.**

## About the code
This repository contains all the code used in the paper, with the definition of our Stochstic Target Encoder in the `common_scripts/STE.py` file.

## Prepare the enviroment
Use `requirements.txt` to install the necessary packages

## Downloading datasets
To download the datasets use the links in `datasets/raw/datasets.txt`

### Converting Bike Sharing dataset
Run `datasets/raw/convert_bike_sharing.py` to convert Bike Sharing dataset to .csv format

## Preprocess the datasets
Run `preprocess_datasets.py` to performing the preprocessing steps on the datasets

## Run the experiments
Run `run_experiments.sh` to initiate the experiments

## Plot the results
Run `plot_results.py` to generate the results images, which will be on the `plots` directory
