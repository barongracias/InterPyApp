import os
import sys
import time
import tracemalloc
import shutil
from pathlib import Path
import numpy as np
import pytest

# make backend modules importable when running from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

tf = pytest.importorskip("tensorflow", reason="TensorFlow is required for fivedreg_tf performance tests")

from interpy_synth import synthetic_5d, synthetic_5d_pickle  # noqa: E402
from fivedreg_tf.trainer_tf import TrainerTF  # noqa: E402
from fivedreg_tf.tester_tf import TesterTF  # noqa: E402


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true_mean) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")


def _run_fivedreg_tf_performance(
    base_dir: Path,
    sizes=(1000, 5000, 10000),
    test_samples: int = 500,
    seed: int = 42,
    epochs: int = 200,
) -> list[dict]:
    data_dir = base_dir / "data"
    out_root = base_dir / "outputs_tf"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for n in sizes:
        data_path = data_dir / f"synthetic_{n}.pkl"
        synthetic_5d_pickle(str(data_path), n=n, seed=seed)

        out_dir = out_root / f"size_{n}"
        out_dir.mkdir(parents=True, exist_ok=True)

        trainer = TrainerTF(
            directory=str(out_dir),
            hidden_sizes=[64, 32, 16],
            Lambda=0.01,
            epochs=epochs,
            learning_rate=0.01,
            train_val_split=0.8,
            batch_size=None,
            grad_clip=5.0,
            early_stop_patience=None,
            lr_decay=None,
            seed=seed,
        )

        tracemalloc.start()
        t0 = time.perf_counter()
        train_loss, val_loss = trainer.train(str(data_path))
        train_time = time.perf_counter() - t0
        _, peak_train = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tester = TesterTF(directory=str(out_dir))
        X_test, y_test = synthetic_5d(test_samples, seed=seed + 123)

        tracemalloc.start()
        t0 = time.perf_counter()
        y_pred = tester.predict(X_test)
        pred_time = time.perf_counter() - t0
        _, peak_pred = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append(
            {
                "n": n,
                "train_time_s": train_time,
                "pred_time_s": pred_time,
                "train_peak_mb": peak_train / (1024 * 1024),
                "pred_peak_mb": peak_pred / (1024 * 1024),
                "train_rmse_end": float(train_loss[-1]),
                "val_rmse_end": float(val_loss[-1]),
                "mse": mse(y_test, y_pred),
                "r2": r2(y_test, y_pred),
            }
        )

        # ensure artifacts saved for this size
        for fname in [
            "model_tf.keras",
            "normalisation_values_tf.npz",
            "rmse_vs_epochs.png",
            "ytrue_vs_ypred.png",
            "tf_model_metadata.json",
        ]:
            assert (out_dir / fname).exists(), f"Missing artifact {fname} for n={n}"

    return results


def test_fivedreg_performance(tmp_path):
    results = _run_fivedreg_tf_performance(Path(tmp_path))
    for r in results:
        assert r["train_time_s"] > 0
        assert r["pred_time_s"] > 0
        assert np.isfinite(r["mse"])
        assert np.isfinite(r["r2"])
        assert r["r2"] > -1  # sanity bound


if __name__ == "__main__":
    if tf is None:  # pragma: no cover - handled by pytest.importorskip
        raise SystemExit("TensorFlow is required to run this script.")

    base = Path(__file__).parent
    # clean previous run outputs for clarity
    for sub in ["data", "outputs_tf"]:
        shutil.rmtree(base / sub, ignore_errors=True)

        results = _run_fivedreg_tf_performance(base)

    print("\n=== Benchmarking fivedreg_tf (TensorFlow) ===")
    header = (
        f"{'n':>6} | {'train_s':>8} | {'pred_s':>7} | {'train_mb':>8} | {'pred_mb':>7} | "
        f"{'train_rmse':>10} | {'val_rmse':>8} | {'mse':>8} | {'r2':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['n']:6d} | "
            f"{r['train_time_s']:8.3f} | "
            f"{r['pred_time_s']:7.4f} | "
            f"{r['train_peak_mb']:8.2f} | "
            f"{r['pred_peak_mb']:7.2f} | "
            f"{r['train_rmse_end']:10.4f} | "
            f"{r['val_rmse_end']:8.4f} | "
            f"{r['mse']:8.6f} | "
            f"{r['r2']:6.4f}"
        )

    print(f"\nArtifacts and plots saved under: {base / 'outputs_tf'}")
