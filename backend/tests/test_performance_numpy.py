import os
import sys
import time
import tracemalloc
import numpy as np

# Ensure backend modules are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from interpy_synth import synthetic_5d, synthetic_5d_pickle
from interpy_bg.trainer import Trainer
from interpy_bg.tester import Tester


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true_mean) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")


def benchmark_sizes(sizes=(1000, 5000, 10000), test_samples=500, seed=42, epochs=200):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    out_root = os.path.join(base_dir, "outputs_numpy")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_root, exist_ok=True)

    results = []

    for n in sizes:
        print(f"\n=== Benchmarking n={n} samples ===")
        data_path = os.path.join(data_dir, f"synthetic_{n}.pkl")
        synthetic_5d_pickle(data_path, n=n, seed=seed)

        out_dir = os.path.join(out_root, f"size_{n}")
        os.makedirs(out_dir, exist_ok=True)

        trainer = Trainer(
            directory=out_dir,
            hidden_sizes=[64, 32, 16],
            Lambda=0.01,
            epochs=epochs,
            learning_rate=0.01,
            train_val_split=0.8,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            activation="relu",
        )

        tracemalloc.start()
        t0 = time.perf_counter()
        train_loss, val_loss = trainer.train(data_path)
        train_time = time.perf_counter() - t0
        _, peak_train = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        metadata = Trainer.dataset_stats(data_path)

        # Prepare tester from saved metadata
        tester_metadata = Tester.load_metadata(directory=out_dir)
        tester = Tester(
            hidden_sizes=[int(x) for x in tester_metadata["hidden_sizes"]],
            Lambda=float(tester_metadata["Lambda"]),
            directory=out_dir,
            activation="relu",
        )

        # ensure artifacts saved for this size
        for fname in [
            "model_weights.npz",
            "normalisation_values.npz",
            "rmse_vs_epochs.png",
            "ytrue_vs_ypred.png",
            "model_metadata.json",
        ]:
            assert os.path.exists(os.path.join(out_dir, fname)), f"Missing artifact {fname} for n={n}"

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
                "rows": metadata["rows"],
                "features": metadata["features"],
            }
        )

    print("\n=== Summary ===")
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

if __name__ == "__main__":
    benchmark_sizes()
