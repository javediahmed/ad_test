"""
Anomaly Detection Test Script
"""

import numpy as np
import pandas as pd
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.abod import ABOD


NUM_COLUMNS_SIMULATED = 8


def get_user_input(prompt):
    """
    Prompt the user for input.

    Args:
        prompt (str): The prompt message.

    Returns:
        str: User input.
    """
    user_input = input(prompt)
    return user_input.strip()


def load_data(filename):
    """
    Load data from a file.

    Args:
        filename (str): The name of the data file.

    Returns:
        pandas.DataFrame: The loaded data.
    """
    data = pd.read_csv(filename)
    return data


def generate_random_data(num_columns):
    """
    Generate random data.

    Args:
        num_columns (int): The number of columns (variables) to generate.

    Returns:
        pandas.DataFrame: The generated data.
    """
    data = pd.DataFrame(
        np.random.randn(1, num_columns),
        columns=[f"Variable_{i}" for i in range(1, num_columns + 1)]
    )
    return data


def process_user_input():
    """
    Process the user input to select the test data source.

    Returns:
        pandas.DataFrame: The selected test data.
    """
    print("Welcome to the Anomaly Detection Test Script.")
    print("Please select the test data source:")
    print("1. Load data from a file")
    print("2. Generate random data")
    data_source = get_user_input("Enter your choice (1 or 2): ")

    if data_source == "1":
        filename = get_user_input("Enter the name of the data file: ")
        data = load_data(filename)
    elif data_source == "2":
        num_columns = get_user_input(
            f"Enter the number of columns (variables) for simulated data (default: {NUM_COLUMNS_SIMULATED}): "
        )
        num_columns = int(num_columns) if num_columns else NUM_COLUMNS_SIMULATED
        data = generate_random_data(num_columns)
    else:
        print("Invalid choice. Exiting the script.")
        return None

    # Print basic summary statistics of the test data
    print("\nSummary statistics of the test data:")
    print(data.describe())

    return data


def main(models):
    """
    Main function to run the anomaly detection test script.

    Args:
        models (dict): Dictionary of models for anomaly detection.
    """
    # Process the user input to select the test data source
    data = process_user_input()
    if data is None:
        return

    while True:
        num_columns = len(data.columns)

        # Prompt the user for the next day's observations
        print("\nPlease enter the next day's observations:")
        next_observations = {}
        for i, column_name in enumerate(data.columns):
            observation = float(get_user_input(f"Observation for {column_name}: "))
            next_observations[column_name] = observation

        # Create a new DataFrame for the next day's observations
        new_observation = pd.DataFrame(next_observations, index=[0])

        # Calculate the mean and standard deviation of the new observation
        observation_mean = new_observation.mean()
        observation_std = new_observation.std()

        # Print the mean and standard deviation of the new observation
        print(f"\nMean of the next observation: {observation_mean}")
        print(f"Standard Deviation of the next observation: {observation_std}")

        # Concatenate the new observation with the existing data
        data = pd.concat([data, new_observation], ignore_index=True)

        # Detect anomalies for each model
        print("\nAnomaly probabilities for the next observation:")
        for model_name, model in models.items():
            anomaly_prob = model.predict_proba(data)[0, 1]
            print(f"Anomaly probability for {model_name}: {anomaly_prob}")


if __name__ == "__main__":
    # Define the models to use for anomaly detection
    models = {
        'HBOS': HBOS(n_bins=10),
        'KNN': KNN(n_neighbors=5),
        'OCSVM': OCSVM(),
        'ABOD': ABOD()
    }

    main(models)
