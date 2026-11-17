import pandas as pd
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris
from evidently import Report
from evidently.presets import DataDriftPreset


def check_data_drift():

    print("Checking data drift...")

    # Load Iris dataset
    iris = load_iris(as_frame=True)
    data = iris.data

    # Reference and current datasets
    reference_data = data.sample(n=100, random_state=42).copy()
    current_data = data.sample(n=100, random_state=99).copy()

    # Simulate data drift
    current_data["sepal length (cm)"] *= 1.5

    # Generate Evidently report
    try:
        report = Report(metrics=[DataDriftPreset()])

        result = report.run(
            reference_data=reference_data,
            current_data=current_data
        )

        # Save HTML report
        result.save_html("drift_report.html")

        print("Evidently report saved: drift_report.html")

    except Exception as error:
        print("Evidently report generation issue:", error)

    # Statistical drift detection using KS test
    drifted_columns = 0

    for column in reference_data.columns:

        statistic, p_value = ks_2samp(
            reference_data[column],
            current_data[column]
        )

        print(f"{column}: p-value = {p_value:.4f}")

        if p_value < 0.05:
            drifted_columns += 1

    drift_detected = drifted_columns >= 1

    print("Drifted columns:", drifted_columns)
    print("Dataset Drift Detected:", drift_detected)

    return drift_detected


if __name__ == "__main__":
    check_data_drift()