import os
import sys
import pickle
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app, OUTPUT_TF_DIR  # noqa: E402


def _make_dataset_file(tmp_path):
    data = {
        "X": np.random.rand(30, 5),
        "y": np.random.rand(30, 1),
    }
    path = tmp_path / "dataset.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


def test_train_predict_evaluate_tf(tmp_path):
    pytest.importorskip("tensorflow")
    client = TestClient(app)
    dataset_path = _make_dataset_file(tmp_path)

    # Upload
    with open(dataset_path, "rb") as f:
        resp = client.post("/upload", files={"file": ("dataset.pkl", f, "application/octet-stream")})
    assert resp.status_code == 200
    stored_filename = resp.json()["stored_filename"]

    # Train TensorFlow backend with minimal epochs for speed
    form = {
        "pkl_filename": stored_filename,
        "hidden_sizes": "4,2",
        "Lambda": "0.01",
        "epochs": "2",
        "learning_rate": "0.01",
        "train_val_split": "0.8",
        "beta1": "0.9",
        "beta2": "0.999",
        "epsilon": "1e-8",
        "model_type": "tf",
        "batch_size": "8",
    }
    resp = client.post("/train", data=form)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_type"] == "tf"
    assert "train_loss_end" in body
    assert "val_loss_end" in body

    # Artifacts exist and are served
    for fname in ["model_tf.keras", "normalisation_values_tf.npz", "tf_model_metadata.json"]:
        assert os.path.exists(os.path.join(OUTPUT_TF_DIR, fname))
        art_resp = client.get(f"/artifacts/{fname}")
        assert art_resp.status_code == 200

    # Predict
    predict_form = {
        "model_type": "tf",
        "input_values": "0.1,0.2,0.3,0.4,0.5",
    }
    resp = client.post("/predict", data=predict_form)
    assert resp.status_code == 200
    preds = resp.json().get("y_pred")
    assert isinstance(preds, list)
    assert len(preds) == 1

    # Evaluate
    with open(dataset_path, "rb") as f:
        eval_resp = client.post("/evaluate", files={"file": ("dataset.pkl", f, "application/octet-stream")})
    assert eval_resp.status_code == 200
    eval_body = eval_resp.json()
    assert "rmse" in eval_body
    assert eval_body["model_type"] in {"tf", "numpy"}  # may prefer numpy if trained separately
