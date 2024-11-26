#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn

def load_data(filepath):
    """Load the test data from a CSV file."""
    data = pd.read_csv(filepath)
    X = data.drop("avg_salary", axis=1)
    y = data["avg_salary"]
    return X, y

def load_model(filepath):
    """Load the trained model from a pickle file."""
    with open(filepath, "rb") as file:
        model = pickle.load(file)
    return model

def evaluate_model(model, X_test, y_test):
    """Make predictions and evaluate the model using R², MAE, and MSE."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return r2, mae, mse

def main():
    # Set up MLflow experiment for evaluation
    mlflow.set_experiment("Model_Evaluation_Experiment")

    # Load test data
    X_test, y_test = load_data('test_data.csv')

    # Load the best model
    model = load_model("best_model.pkl")
    print("Best model loaded from best_model.pkl")

    # Start MLflow run for model evaluation
    with mlflow.start_run(run_name="Model_Evaluation"):
        # Evaluate the model
        r2, mae, mse = evaluate_model(model, X_test, y_test)

        # Log evaluation metrics to MLflow
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)

        # Log the model again (optional)
        mlflow.sklearn.log_model(model, artifact_path="models")

        # Print evaluation results
        print(f"R² Score: {r2}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")

if __name__ == "__main__":
    main()
