import os
import pickle
import sys
import json
import pytest
import numpy as np

# make fivedreg_tf importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

tf = pytest.importorskip("tensorflow", reason="TensorFlow is required for fivedreg_tf tests")

from fivedreg_tf.trainer_tf import TrainerTF  # noqa: E402
from fivedreg_tf.tester_tf import TesterTF  # noqa: E402
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


def test_fivedreg_small_batch(tmp_path):
    out_dir = tmp_path / "tf_small_batch"
    os.makedirs(out_dir, exist_ok=True)
    train_pkl = _make_pkl(str(tmp_path), n=12, seed=11)

    trainer = TrainerTF(
        directory=str(out_dir),
        hidden_sizes=[4],
        epochs=3,
        learning_rate=0.01,
        train_val_split=0.75,
        batch_size=4,
        grad_clip=1.0,
    )
    train_loss, val_loss = trainer.train(train_pkl)
    assert len(train_loss) == len(val_loss) > 0
    assert os.path.exists(out_dir / "model_tf.keras")

    tester = TesterTF(directory=str(out_dir))
    X_test, _ = synthetic_5d(2, seed=123)
    preds = tester.predict(X_test)
    assert preds.shape == (2, 1)


def test_tf_hyperparams_wiring(tmp_path):
    out_dir = tmp_path / "tf_hparams"
    os.makedirs(out_dir, exist_ok=True)
    train_pkl = _make_pkl(str(tmp_path), n=16, seed=7)

    activation = "tanh"
    weight_init = "xavier"
    beta1 = 0.8
    beta2 = 0.9
    epsilon = 1e-7
    batch_size = 8
    grad_clip = 2.5

    trainer = TrainerTF(
        directory=str(out_dir),
        hidden_sizes=[6],
        epochs=2,
        learning_rate=0.005,
        train_val_split=0.75,
        activation=activation,
        weight_init=weight_init,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        batch_size=batch_size,
        grad_clip=grad_clip,
    )
    train_loss, val_loss = trainer.train(train_pkl)
    assert train_loss and val_loss

    dense_layer = trainer.model.layers[1]
    cfg = dense_layer.get_config()
    assert cfg["activation"] == activation
    assert cfg["kernel_initializer"]["class_name"].lower().startswith("glorot")

    opt = trainer.model.optimizer
    assert abs(float(opt.beta_1.numpy()) - beta1) < 1e-6
    assert abs(float(opt.beta_2.numpy()) - beta2) < 1e-6
    assert abs(float(opt.epsilon) - epsilon) < 1e-9
    assert opt.clipnorm == grad_clip

    meta_path = out_dir / "tf_model_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta.get("activation") == activation
    assert meta.get("weight_init") == weight_init
    assert meta.get("beta1") is not None and abs(meta["beta1"] - beta1) < 1e-6
    assert meta.get("beta2") is not None and abs(meta["beta2"] - beta2) < 1e-6
    assert meta.get("epsilon") is not None and abs(meta["epsilon"] - epsilon) < 1e-9
    assert meta.get("batch_size") == batch_size
    assert meta.get("grad_clip") == grad_clip
