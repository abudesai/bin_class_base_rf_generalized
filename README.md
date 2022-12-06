# RANDOM FOREST BINARY CLASSIFIER

This is a repo for a random forest binary classifier model that uses a data schema as per Ready Tensor specifications.

## Description

The implementation has been built with following capabilities

1. Read training data using data schema
2. Preprocess training data for binary classification problem
3. Train random forest model on the dataset and save the model
4. Evaluate the saved model's performance on test data

The implementation uses Ready Tensor specifications for data schema for the binary classification problem.

## Usage

1. Setup virtual env and use requirements.txt to install dependencies
2. Pick a dataset from one of these three: titanic, mushroom, spam as an example
3. Find the specific files for the dataset in the /app/examples folder. For the specific dataset:
   1. Move its schema file (the json file) into /app/data/data_config. Dont forget to delete any previous file from this folder. There can only be one schema file.
   2. Move the training file (\*\_train.csv) into /app/data/processed_data/training folder. Remove any other file from this folder.
   3. Move the testing file (\*\_test.csv) into /app/data/processed_data/testing folder. Remove any other file from this folder.
4. Run the python script "train.py" in folder /app/src. This will save model artifacts in /app/atifacts.
5. To test the model' performance, run the script "evaluate.py" in folder /app/src. This will save the test predictions as well as scores in /app/results.
6. You may go back to step 2 and try steps 3-5 with the other datasets.
