import pandas as pd

# Check if dataset exists and is readable
def test_dataset_exists():
    df = pd.read_csv("data/heart.csv")
    assert df.shape[0] > 0

# Ensure target column contains only binary values
def test_target_binary():
    df = pd.read_csv("data/heart.csv")
    assert set(df["target"].unique()).issubset({0, 1})

# Ensure expected number of features are present
def test_feature_count():
    df = pd.read_csv("data/heart.csv")
    X = df.drop("target", axis=1)
    assert X.shape[1] >= 10
