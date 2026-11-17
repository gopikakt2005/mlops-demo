from unittest.mock import patch


def test_retraining_pipeline_runs():
    from src import retrain_pipeline

    with patch.object(
        retrain_pipeline,
        "check_data_drift",
        return_value=False
    ):
        retrain_pipeline.automated_continuous_training()
