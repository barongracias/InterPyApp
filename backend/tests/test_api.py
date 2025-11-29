import os
import sys
import pickle
import numpy as np
from fastapi.testclient import TestClient

# Ensure backend modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app, OUTPUT_DIR


def _make_dataset_file(tmp_path):
    data = {
        "X": np.random.rand(20, 5),
        "y": np.random.rand(20, 1),
    }
    path = tmp_path / "dataset.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


def test_health():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "ok"


def test_upload_and_stats(tmp_path):
    client = TestClient(app)
    dataset_path = _make_dataset_file(tmp_path)

    with open(dataset_path, "rb") as f:
        files = {"file": ("dataset.pkl", f, "application/octet-stream")}
        resp = client.post("/upload", files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert data["message"].startswith("File uploaded")
    stats = data["stats"]
    assert stats["rows"] == 20
    assert stats["features"] == 5
    assert len(stats["x_min"]) == 5
    assert len(stats["x_max"]) == 5


def test_train_and_predict(tmp_path):
    client = TestClient(app)
    dataset_path = _make_dataset_file(tmp_path)

    # Upload
    with open(dataset_path, "rb") as f:
        resp = client.post("/upload", files={"file": ("dataset.pkl", f, "application/octet-stream")})
    assert resp.status_code == 200
    filename = os.path.basename(dataset_path)

    # Train with small epochs for speed
    form = {
        "pkl_filename": filename,
        "hidden_sizes": "4,2",
        "Lambda": "0.01",
        "epochs": "5",
        "learning_rate": "0.01",
        "train_val_split": "0.8",
        "beta1": "0.9",
        "beta2": "0.999",
        "epsilon": "1e-8",
    }
    resp = client.post("/train", data=form)
    assert resp.status_code == 200
    body = resp.json()
    assert "train_loss_end" in body
    assert "val_loss_end" in body

    # Artifacts exist
    for fname in ["model_weights.npz", "normalisation_values.npz", "model_metadata.json"]:
        assert os.path.exists(os.path.join(OUTPUT_DIR, fname))

    # Predict with manual values
    predict_form = {
        "hidden_sizes": "4,2",  # ignored but required by schema
        "Lambda": "0.01",
        "input_values": "0.1,0.2,0.3,0.4,0.5",
    }
    resp = client.post("/predict", data=predict_form)
    assert resp.status_code == 200
    preds = resp.json().get("y_pred")
    assert isinstance(preds, list)
    assert len(preds) == 1