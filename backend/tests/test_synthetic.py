import os
import numpy as np
from interpy_bg.synthetic import synthetic_5d, synthetic_5d_pickle


def test_synthetic_shapes_and_types(tmp_path):
    X, y = synthetic_5d(50, seed=42)
    assert X.shape == (50, 5)
    assert y.shape == (50, 1)
    assert X.dtype == np.float32
    assert y.dtype == np.float32

    out_path = tmp_path / "synthetic.pkl"
    written = synthetic_5d_pickle(str(out_path), 50, seed=42)
    assert os.path.exists(written)

    import pickle
    with open(written, "rb") as f:
        data = pickle.load(f)

    assert "X" in data and "y" in data and "metadata" in data
    assert data["metadata"]["n_samples"] == 50
    assert data["metadata"]["n_features"] == 5
    assert data["metadata"]["targetname"] == "y"
    assert isinstance(data["metadata"]["feature_names"], list)