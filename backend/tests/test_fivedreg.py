import os
import pickle
import sys
import pytest
import numpy as np

# make fivedreg importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

tf = pytest.importorskip("tensorflow", reason="TensorFlow is required for fivedreg tests")

from fivedreg.trainer_tf import TrainerTF  # noqa: E402
from fivedreg.tester_tf import TesterTF  # noqa: E402
from interpy_synth import synthetic_5d  # noqa: E402


def _make_pkl(tmpdir: str, fname: str = "train.pkl", n: int = 20, seed: int = 7) -> str:
    X, y = synthetic_5d(n, seed=seed)
    path = os.path.join(tmpdir, fname)
    with open(path, "wb") as f:
        pickle.dump({"X": X, "y": y}, f)
    return path


def test_fivedreg_smoke(tmp_path):
    out_dir = tmp_path / "tf_smoke"
    os.makedirs(out_dir, exist_ok=True)
    train_pkl = _make_pkl(str(tmp_path), n=20, seed=321)

    trainer = TrainerTF(
        directory=str(out_dir),
        hidden_sizes=[8, 4],
        epochs=5,
        learning_rate=0.01,
        train_val_split=0.8,
    )
    train_loss, val_loss = trainer.train(train_pkl)

    assert len(train_loss) == len(val_loss) > 0
    assert all(np.isfinite(train_loss))
    assert all(np.isfinite(val_loss))
    assert os.path.exists(out_dir / "model_tf.keras")
    assert os.path.exists(out_dir / "normalisation_values_tf.npz")
    assert os.path.exists(out_dir / "rmse_vs_epochs.png")
    assert os.path.exists(out_dir / "ytrue_vs_ypred.png")
    meta_path = out_dir / "tf_model_metadata.json"
    assert os.path.exists(meta_path)
    # metadata sanity: best_epoch and final metrics present
    import json
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta.get("best_epoch") is None or meta["best_epoch"] >= 1
    assert meta.get("best_val_rmse") is None or meta["best_val_rmse"] >= 0
    assert meta.get("model_type") == "tf"

    tester = TesterTF(directory=str(out_dir))
    X_test, _ = synthetic_5d(3, seed=9999)
    preds = tester.predict(X_test)
    assert preds.shape == (3, 1)
    assert np.all(np.isfinite(preds))
