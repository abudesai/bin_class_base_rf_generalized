import numpy as np, pandas as pd
import joblib
import sys
import os, warnings

warnings.filterwarnings("ignore")


from sklearn.ensemble import RandomForestClassifier

model_fname = "model.save"
MODEL_NAME = "bin_class_base_random_forest_sklearn"


class Classifier:
    def __init__(
        self, n_estimators=100, min_samples_split=2, min_samples_leaf=1, **kwargs
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.model = self.build_model()

    def build_model(self):
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        )
        return model

    def fit(self, train_X, train_y):
        self.model.fit(train_X, train_y)

    def predict(self, X, verbose=False):
        preds = self.model.predict(X)
        return preds

    def predict_proba(self, X, verbose=False):
        preds = self.model.predict_proba(X)
        return preds

    def summary(self):
        self.model.get_params()

    @property
    def classes_(self):
        return self.model.classes_

    def evaluate(self, x_test, y_test):
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)

    def save(self, model_path):
        joblib.dump(self, os.path.join(model_path, model_fname))

    @classmethod
    def load(cls, model_path):
        classifier = joblib.load(os.path.join(model_path, model_fname))
        # print("where the load function is getting the model from: "+ os.path.join(model_path, model_fname))
        return classifier


def save_model(model, model_path):
    # print(os.path.join(model_path, model_fname))
    joblib.dump(model, os.path.join(model_path, model_fname))  # this one works
    # print("where the save_model function is saving the model to: " + os.path.join(model_path, model_fname))


def load_model(model_path):
    try:
        model = joblib.load(os.path.join(model_path, model_fname))
    except:
        raise Exception(
            f"""Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?"""
        )
    return model


def get_data_based_model_params(data):
    """
    Set any model parameters that are data dependent.
    For example, number of layers or neurons in a neural network as a function of data shape.
    """
    return {"max_features": max(1, 0.5 * data.shape[1])}
