
import mlflow  
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from scipy.stats import randint
import tempfile
import os
import json

def load_and_prep_data(data_path: str):
    """Load and prepare data for training."""
    data = pd.read_csv(data_path)
    X = data.drop(columns=["date", "demand"])
    X = X.astype('float')
    y = data["demand"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def main():
    # Basic setup
    EXPERIMENT_NAME = "Autolog_RandomizedSearchCV_on_RandomForestRegressor" 
    TRACKING_URI = "http://127.0.0.1:8080"
    DATA_PATH = "data/fake_data.csv"
    N_TRIALS = 30

    # Set up MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        experiment = client.set_experiment(EXPERIMENT_NAME)

    print(experiment)
    
    # Enable autologging
    mlflow.sklearn.autolog(
        log_models=True
    )

    # Load data
    X_train, X_test, y_train, y_test = load_and_prep_data(DATA_PATH)

    # Define hyperparameter search space
    param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 20),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Create and run RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=N_TRIALS,             # number of random combinations to try
        cv=5,                  # 5-fold cross-validation
        scoring='r2',
        random_state=42,
        n_jobs=-1              # use all cores
    )
    
    search.fit(X_train, y_train)

    # Retrieve information on the best model
    best_model = search.best_estimator_

    # Create a summary of results
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
    print(metrics)

    # Save the summary as an artifact
    summary = {
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "metrics": metrics
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = os.path.join(tmpdir, "model_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path)

if __name__ == "__main__":
    main()