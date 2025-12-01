import os
import pickle
import sys
import pytest
import numpy as np

# add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

tf = pytest.importorskip("tensorflow", reason="TensorFlow is required for fivedreg tests")

from fivedreg.trainer_tf import TrainerTF  # noqa: E402
from fivedreg.tester_tf import TesterTF  # noqa: E402
from interpy_synth import synthetic_5d  # noqa: E402


def _make_train_pickle(tmpdir: str, n: int = 40, seed: int = 42) -> str:
    X, y = synthetic_5d(n, seed=seed)
    path = os.path.join(tmpdir, "train.pkl")
    with open(path, "wb") as f:
        pickle.dump({"X": X, "y": y}, f)
    return path


def test_fivedreg_end_to_end(tmp_path):
    out_dir = tmp_path / "tf_out"
    os.makedirs(out_dir, exist_ok=True)
    train_pkl = _make_train_pickle(str(tmp_path), n=40, seed=123)

    trainer = TrainerTF(
        directory=str(out_dir),
        hidden_sizes=[16, 8],
        epochs=15,
        learning_rate=0.01,
        train_val_split=0.8,
        early_stop_patience=5,
    )
    train_loss, val_loss = trainer.train(train_pkl)

    assert len(train_loss) == len(val_loss) > 0
    assert all(np.isfinite(train_loss))
    assert all(np.isfinite(val_loss))

    # artifacts exist
    assert os.path.exists(out_dir / "model_tf.keras")
    assert os.path.exists(out_dir / "normalisation_values_tf.npz")
    assert os.path.exists(out_dir / "rmse_vs_epochs.png")
    assert os.path.exists(out_dir / "ytrue_vs_ypred.png")
    assert os.path.exists(out_dir / "tf_model_metadata.json")

    tester = TesterTF(directory=str(out_dir))

    X_test, _ = synthetic_5d(5, seed=999)
    preds = tester.predict(X_test)
    assert preds.shape == (5, 1)
    assert np.all(np.isfinite(preds))

    # test pkl input
    pkl_path = out_dir / "X_test.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(X_test, f)
    preds_pkl = tester.predict(str(pkl_path))
    assert preds_pkl.shape == (5, 1)
    assert np.all(np.isfinite(preds_pkl))
