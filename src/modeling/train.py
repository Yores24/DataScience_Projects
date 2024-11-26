#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn


def load_data(filepath):
    """Load the training data from a CSV file."""
    data = pd.read_csv(filepath)
    X = data.drop("avg_salary", axis=1)
    y = data["avg_salary"]
    return X, y


def train_model(model, X_train, y_train, model_name):
    """Train a model, evaluate it, and log it to MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Train model
        model.fit(X_train, y_train)
        
        # Cross-validate model
        score = cross_val_score(model, X_train, y_train, scoring="r2", cv=3).mean()
        
        # Log model and metrics to MLflow
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("r2_score", score)
        mlflow.sklearn.log_model(model, artifact_path="models")
        
        print(f"{model_name} logged to MLflow with R² Score: {score}")
    return score


def save_best_model(model, filepath):
    """Save the best model to a pickle file."""
    # Check if the directory exists, if not create it
    model_dir = os.path.dirname(filepath)
    if not os.path.exists(model_dir):
        print(f"Directory '{model_dir}' does not exist. Creating it.")
        os.makedirs(model_dir)
    
    with open(filepath, "wb") as file:
        pickle.dump(model, file)
    print(f"Best model saved to {filepath}")


def main():
    # Set up MLflow experiment
    mlflow.set_experiment("Model_Training_Experiment")

    # Check and create necessary directories for saving logs and models
    if not os.path.exists("models"):
        print("Directory 'models' does not exist. Creating it.")
        os.makedirs("models")
    if not os.path.exists("data/processed"):
        print("Directory 'data/processed' does not exist. Creating it.")
        os.makedirs("data/processed")

    # Load training data
    X_train, y_train = load_data("data/processed/train_data.csv")

    # Define models to evaluate
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(),
        "Random Forest Regressor": RandomForestRegressor(),
    }

    # Train and evaluate each model
    model_scores = {}
    for name, model in models.items():
        r2_score = train_model(model, X_train, y_train, name)
        model_scores[name] = r2_score

    # Select the best model
    best_model_name = max(model_scores, key=model_scores.get)
    best_model = models[best_model_name]
    print(f"\nBest Model: {best_model_name} with R² Score: {model_scores[best_model_name]}")

    # Save the best model
    save_best_model(best_model, "models/best_model.pkl")

    # Log best model information to MLflow
    with mlflow.start_run(run_name="Best_Model"):
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_model_r2_score", model_scores[best_model_name])
        mlflow.sklearn.log_model(best_model, artifact_path="best_model")
        print(f"Best model logged to MLflow as 'best_model'")


if __name__ == "__main__":
    main()
