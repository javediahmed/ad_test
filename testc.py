"""
Anomaly Detection Test Script

This script allows users to perform anomaly detection on a given dataset. It provides options to input training data
from a file or generate random data. The script uses a set of anomaly detection models, including HBOS, KNN, OCSVM,
and ABOD, to train on the provided data and allows users to test the models on new data.

Created on: 2023-05-17
"""

import numpy as np
import pandas as pd
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.abod import ABOD

class AnomalyDetection:
    """
    Anomaly Detection class for training and testing anomaly detection models.

    The AnomalyDetection class provides methods to train the models on a given dataset and test them on new data.
    It allows users to input training data from a file or generate random data. The class uses a set of anomaly
    detection models, including HBOS, KNN, OCSVM, and ABOD.

    Attributes:
        training_data (pandas.DataFrame): The training data.
        models (list): List of anomaly detection models.

    Methods:
        get_user_input: Prompt the user for input.
        load_data: Load data from a file.
        generate_random_data: Generate random data.
        process_training_data_input: Process the user input to select the training data source.
        process_test_data_input: Process the user input for the test data.
        train: Train the anomaly detection models on the training data.
        test: Test the anomaly detection models on new data.
    """

    def __init__(self):
        self.training_data = None
        self.models = [HBOS(), KNN(), OCSVM(), ABOD()]

    def get_user_input(self, prompt):
        """
        Prompt the user for input.

        Args:
            prompt (str): The prompt message.

        Returns:
            str: User input.
        """
        return input(prompt).strip()

    def load_data(self, filename):
        """
        Load data from a file.

        Args:
            filename (str): The name of the data file.

        Returns:
            pandas.DataFrame: The loaded data.
        """
        return pd.read_csv(filename)

    def generate_random_data(self, num_columns):
        """
        Generate random data.

        Args:
            num_columns (int): The number of columns (variables) to generate.

        Returns:
            pandas.DataFrame: The generated data.
        """
        start_date = pd.Timestamp("2020-01-01")
        end_date = pd.Timestamp.now().floor("D")
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        data = pd.DataFrame(
            np.random.randn(len(dates), num_columns),
            columns=[f"Variable_{i}" for i in range(1, num_columns + 1)],
            index=dates,
        )
        return data

    def process_training_data_input(self):
        """
        Process the user input to select the training data source.

        Returns:
            pandas.DataFrame: The selected training data.
        """
        print("Welcome to the Anomaly Detection Test Script.")
        print("Please select the training data source:")
        print("1. Load data from a file")
        print("2. Generate random data")
        data_source = self.get_user_input("Enter your choice (1 or 2): ")

        if data_source == "1":
            filename = self.get_user_input("Enter the file path: ")
            training_data = self.load_data(filename)
        elif data_source == "2":
            num_columns = int(self.get_user_input("Enter the number of columns (variables): "))
            training_data = self.generate_random_data(num_columns)
        else:
            print("Invalid choice. Exiting the script.")
            training_data = None

        return training_data

    def process_test_data_input(self):
        """
        Process the user input for the test data.

        Returns:
            pandas.DataFrame: The processed test data.
        """
        next_date = self.training_data.index.max() + pd.DateOffset(days=1)
        print(f"\nPlease select the test data source for {next_date.date()}:")
        test_data = pd.DataFrame(columns=self.training_data.columns, index=[next_date])

        user_choice = self.get_user_input("1. File\n2. Simulate\nEnter your choice: ")

        if user_choice == "1":  # File
            file_path = self.get_user_input("Enter the file path: ")
            column_data = self.load_data(file_path)
            test_data.loc[next_date] = column_data.iloc[0].values

        elif user_choice == "2":  # Simulate
            simulated_data = pd.DataFrame(
                np.random.normal(self.training_data.mean(), self.training_data.std(), size=(1, len(self.training_data.columns))),
                columns=self.training_data.columns,
                index=[next_date]
            )
            test_data = test_data.combine_first(simulated_data)

        else:
            print("Invalid choice. Exiting the script.")

        return test_data

    def train(self):
        """
        Train the anomaly detection models on the training data.
        """
        print("\nTraining the anomaly detection models...")
        for model in self.models:
            model.fit(self.training_data)

    def test(self):
        """
        Test the anomaly detection models on new data.
        """
        anomaly_probabilities = pd.DataFrame(columns=[model.__class__.__name__ for model in self.models])

        while True:
            test_data = self.process_test_data_input()

            if test_data is None:
                break

            print("\nRunning anomaly detection on the test data...")
            probabilities = pd.DataFrame()
            for model in self.models:
                scores = model.decision_function(test_data)
                probabilities[model.__class__.__name__] = scores

            anomaly_probabilities.loc[test_data.index[0]] = probabilities.iloc[0]

        print("\nAnomaly Detection Results:")
        print(anomaly_probabilities)


if __name__ == "__main__":
    anomaly_detection = AnomalyDetection()
    anomaly_detection.training_data = anomaly_detection.process_training_data_input()

    if anomaly_detection.training_data is not None:
        anomaly_detection.train()
        anomaly_detection.test()
