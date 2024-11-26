#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from logger import CustomLogger  # Assuming your logger file is named `logger.py`

# Initialize the custom logger
custom_logger = CustomLogger(logger_name="DataPreparationLogger").logger

# Ensure directory exists
def ensure_directory(directory_path):
    """
    Check if a directory exists, and create it if not.
    """
    if not os.path.exists(directory_path):
        custom_logger.info(f"Directory {directory_path} does not exist. Creating it.")
        os.makedirs(directory_path, exist_ok=True)
        custom_logger.info(f"Directory {directory_path} created successfully.")
    else:
        custom_logger.info(f"Directory {directory_path} already exists.")

# Load and split dataset
def load_and_split_data(filepath, test_size=0.2, random_state=42):
    """
    Load the dataset, split it into training and testing sets,
    and save them as separate files.
    """
    custom_logger.info(f"Loading dataset from {filepath}")
    try:
        df = pd.read_csv(filepath)
        custom_logger.info(f"Dataset loaded successfully with shape {df.shape}")
    except FileNotFoundError as e:
        custom_logger.error(f"File not found: {filepath}")
        raise e

    custom_logger.info(f"Splitting dataset with test_size={test_size} and random_state={random_state}")
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
    custom_logger.info(f"Data split completed: Train shape {train_data.shape}, Test shape {test_data.shape}")
    return train_data, test_data

# Save data to CSV
def save_data(train_data, test_data, train_filepath, test_filepath):
    """
    Save the training and testing sets to CSV files.
    """
    # Ensure the directories for the files exist
    ensure_directory(os.path.dirname(train_filepath))
    ensure_directory(os.path.dirname(test_filepath))

    custom_logger.info(f"Saving training data to {train_filepath}")
    train_data.to_csv(train_filepath, index=False)
    custom_logger.info(f"Training data saved as {train_filepath}")

    custom_logger.info(f"Saving testing data to {test_filepath}")
    test_data.to_csv(test_filepath, index=False)
    custom_logger.info(f"Testing data saved as {test_filepath}")

# Main function to run data preparation
def main():
    input_filepath = 'data/processed/eda_data.csv'
    train_filepath = 'data/processed/train_data.csv'
    test_filepath = 'data/processed/test_data.csv'

    custom_logger.info("Starting the data preparation process")
    try:
        train_data, test_data = load_and_split_data(input_filepath)
        save_data(train_data, test_data, train_filepath, test_filepath)
        custom_logger.info("Data preparation process completed successfully")
    except Exception as e:
        custom_logger.error(f"Data preparation process failed: {e}")

if __name__ == "__main__":
    main()
