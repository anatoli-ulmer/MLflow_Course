# 03_autolog.py

import pandas as pd
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
import mlflow

# Define tracking_uri
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Define experiment name, run name and artifact_path name
apple_experiment = mlflow.set_experiment("Iris_Models") 

# enable auto tracking
mlflow.autolog()

# Import Database and train model
iris = datasets.load_iris()
parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(iris.data, iris.target)
