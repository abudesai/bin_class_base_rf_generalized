import os, warnings, sys
import json
import numpy as np, pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


sys.path.insert(0, "./../../")


import app.src.preprocessing.pipeline as pipeline
import app.src.model.classifier as classifier
import app.src.utils as utils


test_data_path = "./../data/processed_data/testing/"
schema_path = "./../data/data_config/"
artifacts_path = "./../artifacts/"
results_path = "./../results/"

results_fname = "results.json"


def run_evaluation():

    # read train_data
    test_data = utils.get_data(test_data_path)
    # print(test_data.shape)

    # get data schema
    data_schema = utils.get_data_schema(schema_path)

    # get predictions
    predictions = get_predictions(test_data)

    # get scroes
    scores = get_scores(test_data, predictions, data_schema)
    print(scores)
    with open(os.path.join(results_path, results_fname), "w") as outfile:
        json.dump(scores, outfile, indent=2)

    # save predictions
    test_data["prediction"] = predictions
    test_data.to_csv(f"{results_path}predictions.csv", index=False)


def get_predictions(test_data):
    # load preprocessors
    inputs_pipeline = pipeline.load_preprocessor(artifacts_path)

    # preprocess test inputs
    processed_test_inputs = inputs_pipeline.transform(test_data)

    # load model
    model = classifier.load_model(artifacts_path)

    # make predictions
    predictions = model.predict(processed_test_inputs)

    return predictions


def get_scores(test_data, predictions, data_schema):
    # actuals
    Y = test_data[
        data_schema["inputDatasets"]["binaryClassificationBaseMainInput"]["targetField"]
    ]

    # predictions
    Y_hat = np.squeeze(predictions)

    accu = accuracy_score(Y, Y_hat)
    f1 = f1_score(Y, Y_hat)
    precision = precision_score(Y, Y_hat)
    recall = recall_score(Y, Y_hat)
    scores = {
        "accuracy": np.round(accu, 4),
        "f1_score": np.round(f1, 4),
        "precision": np.round(precision, 4),
        "recall": np.round(recall, 4),
        "perc_pred_missing": np.round(
            100 * (1 - predictions.shape[0] / test_data.shape[0]), 2
        ),
    }
    return scores


if __name__ == "__main__":

    run_evaluation()
