from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from train import load_data, build_preprocessor
import pandas as pd

def test_dataset_target_and_rows():
    df, X, y = load_data(ROOT/"data"/"tourism.csv")
    assert len(df) > 1000
    assert set(y.unique()) == {0,1}
    assert "CustomerID" not in X.columns
    assert "ProdTaken" not in X.columns

def test_preprocessor_transforms_data():
    df, X, y = load_data(ROOT/"data"/"tourism.csv")
    prep=build_preprocessor(X)
    transformed=prep.fit_transform(X.iloc[:50],y.iloc[:50])
    assert transformed.shape[0] == 50
    assert transformed.shape[1] >= X.shape[1]
