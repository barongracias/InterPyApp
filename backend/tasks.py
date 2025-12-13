import json
import logging
import os
import time
from typing import Optional, Sequence

import numpy as np
from rq import get_current_job

from interpy_bg.trainer import Trainer
from interpy_bg.tester import Tester


def get_logger():
    logger = logging.getLogger("interpy_app.tasks")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - (%(name)s) - [%(levelname)s]: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = get_logger()


def run_training_job(
    *,
    pkl_path: str,
    hidden_sizes: Sequence[int],
    Lambda: float,
    epochs: int,
    learning_rate: float,
    train_val_split: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    early_stop_patience: Optional[int],
    lr_decay: Optional[float],
    activation: str,
    weight_init: str,
    batch_size: Optional[int],
    grad_clip: Optional[float],
    seed: Optional[int],
    model_type: str,
    output_numpy_dir: str,
    output_tf_dir: str,
) -> dict:
    """
    Execute a training job (NumPy or TensorFlow) synchronously.

    Returns a result dict with metrics/artifact names; intended to be run inside an RQ worker.
    """
    start = time.monotonic()
    job = get_current_job()
    model_type = model_type.lower()
    if model_type not in {"numpy", "tf"}:
        raise ValueError("model_type must be 'numpy' or 'tf'")

    if model_type == "numpy":
        trainer = Trainer(
            directory=output_numpy_dir,
            hidden_sizes=list(hidden_sizes),
            Lambda=Lambda,
            epochs=epochs,
            learning_rate=learning_rate,
            train_val_split=train_val_split,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            early_stop_patience=early_stop_patience,
            lr_decay=lr_decay,
            activation=activation,
            weight_init=weight_init,
            batch_size=batch_size,
            grad_clip=grad_clip,
            seed=seed,
        )
        train_loss, val_loss = trainer.train(pkl_path)
        result = {
            "message": "Training completed successfully.",
            "model_type": "numpy",
            "train_loss_start": train_loss[0],
            "train_loss_end": train_loss[-1],
            "val_loss_start": val_loss[0],
            "val_loss_end": val_loss[-1],
            "best_val_rmse": trainer.best_val_rmse,
            "best_train_rmse": trainer.best_train_rmse,
            "best_epoch": trainer.best_epoch,
            "epochs_run": len(train_loss),
            "baseline_rmse": trainer.baseline_rmse,
            "final_train_r2": trainer.final_train_r2,
            "final_val_r2": trainer.final_val_r2,
            "plots": ["rmse_vs_epochs.png", "ytrue_vs_ypred.png"],
            "artifacts": ["model_weights.npz", "normalisation_values.npz", "model_metadata.json"],
        }
    else:
        from fivedreg_tf.trainer_tf import TrainerTF  # lazy import to keep TF optional at import time

        trainer_tf = TrainerTF(
            directory=output_tf_dir,
            hidden_sizes=list(hidden_sizes),
            Lambda=Lambda,
            epochs=epochs,
            learning_rate=learning_rate,
            train_val_split=train_val_split,
            seed=seed,
            activation=activation,
            weight_init=weight_init,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            early_stop_patience=early_stop_patience,
            lr_decay=lr_decay,
            batch_size=batch_size,
            grad_clip=grad_clip,
        )

        train_loss, val_loss = trainer_tf.train(pkl_path)
        result = {
            "message": "Training completed successfully.",
            "model_type": "tf",
            "train_loss_start": train_loss[0] if train_loss else None,
            "train_loss_end": train_loss[-1] if train_loss else None,
            "val_loss_start": val_loss[0] if val_loss else None,
            "val_loss_end": val_loss[-1] if val_loss else None,
            "best_val_rmse": trainer_tf.best_val_rmse,
            "best_train_rmse": trainer_tf.best_train_rmse,
            "best_epoch": trainer_tf.best_epoch,
            "epochs_run": len(train_loss),
            "baseline_rmse": trainer_tf.baseline_rmse,
            "final_train_r2": trainer_tf.final_train_r2,
            "final_val_r2": trainer_tf.final_val_r2,
            "plots": ["rmse_vs_epochs.png", "ytrue_vs_ypred.png"],
            "artifacts": ["model_tf.keras", "normalisation_values_tf.npz", "tf_model_metadata.json"],
        }

    duration_ms = (time.monotonic() - start) * 1000
    payload = {"duration_ms": round(duration_ms, 2), **result}
    try:
        if job:
            job.meta["result"] = payload
            job.meta["status"] = "finished"
            job.save_meta()
    except Exception:
        pass
    logger.info(f"event=train.completed {json.dumps({'backend': model_type, 'duration_ms': round(duration_ms, 2)})}")
    return payload
