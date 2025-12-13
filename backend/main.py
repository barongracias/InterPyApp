import os
import shutil
import pickle
import numpy as np
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import redis
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from rq import Queue
from starlette.concurrency import run_in_threadpool

from interpy_bg.trainer import Trainer
from interpy_bg.tester import Tester
from tasks import run_training_job

# ----------------------
# DIRECTORIES
# ----------------------
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_NUMPY_DIR = os.path.join(BASE_DIR, "outputs_numpy")
OUTPUT_TF_DIR = os.path.join(BASE_DIR, "outputs_tf")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_ARTIFACTS_NUMPY = {
    "model_weights.npz",
    "normalisation_values.npz",
    "model_metadata.json",
}
ALLOWED_ARTIFACTS_TF = {
    "model_tf.keras",
    "normalisation_values_tf.npz",
    "tf_model_metadata.json",
}
ALLOWED_ARTIFACTS = ALLOWED_ARTIFACTS_NUMPY | ALLOWED_ARTIFACTS_TF
ALLOWED_CONTENT_TYPES = {"application/octet-stream", "application/x-pickle", "application/python-pickle"}
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
QUEUE_NAME = os.getenv("QUEUE_NAME", "train")

def get_app_logger():
    """Return a module-level logger for main app utilities."""
    import logging
    logger = logging.getLogger("interpy_app")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - (%(name)s) - [%(levelname)s]: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

app_logger = get_app_logger()

def log_event(action: str, **fields):
    """Emit a structured log line with a consistent prefix."""
    try:
        payload = json.dumps(fields, default=str, sort_keys=True)
    except Exception as exc:
        payload = json.dumps({"log_error": str(exc)})
    app_logger.info(f"event={action} {payload}")

def error_response(status_code: int, detail: str):
    """Consistent error responses."""
    return HTTPException(status_code=status_code, detail=detail)

def clear_directories():
    """Helper to clean uploads and both NumPy/TF outputs folders without removing mount points."""
    def _clear_dir(path: str) -> None:
        try:
            os.makedirs(path, exist_ok=True)
            for entry in os.scandir(path):
                try:
                    if entry.is_dir(follow_symlinks=False):
                        shutil.rmtree(entry.path)
                    else:
                        os.unlink(entry.path)
                except OSError as exc:
                    app_logger.warning(f"Could not remove {entry.path}: {exc}")
        except OSError as exc:
            app_logger.warning(f"Could not scan {path}: {exc}")

    for d in [UPLOAD_DIR, OUTPUT_NUMPY_DIR, OUTPUT_TF_DIR]:
        _clear_dir(d)
    app_logger.info("Cleared uploads and outputs directories.")

# ----------------------
# LIFESPAN HANDLER (replaces deprecated on_event)
# ----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_dirs()
    # Clean folders on startup
    clear_directories()
    yield
    # Optionally clean again on shutdown
    # clear_directories()

# ----------------------
# APP SETUP
# ----------------------
app = FastAPI(
    title="Interpolator App API",
    description="Endpoints for training and testing the 5D Interpolator neural network.",
    version="0.2.0",
    contact={"name": "Baron Gracias"},
    lifespan=lifespan,
)

# ----------------------
# MIDDLEWARE
# ----------------------
def _allowed_origins():
    raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    return [x.strip() for x in raw.split(",") if x.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# ROUTES
# ----------------------
def _validate_hyperparams(
    hidden_sizes: List[int],
    Lambda: float,
    epochs: int,
    learning_rate: float,
    train_val_split: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> str | None:
    if not hidden_sizes:
        return "Hidden sizes must contain at least one positive integer."
    if any(h <= 0 for h in hidden_sizes):
        return "Hidden sizes must be positive integers."
    if Lambda <= 0:
        return "Lambda must be positive."
    if epochs <= 0:
        return "Epochs must be positive."
    if learning_rate <= 0:
        return "Learning rate must be positive."
    if not 0 < train_val_split < 1:
        return "Train/val split must be between 0 and 1."
    if not 0 < beta1 < 1:
        return "Beta1 must be between 0 and 1."
    if not 0 < beta2 < 1:
        return "Beta2 must be between 0 and 1."
    if epsilon <= 0:
        return "Epsilon must be positive."
    return None
@app.get("/")
async def root():
    return {"message": "Welcome to the 5D Interpolator App API"}

@app.get("/health")
async def health():
    """
    Simple health check endpoint.
    """
    return {"status": "ok", "version": app.version}

def _ensure_dirs():
    """Ensure upload/output dirs exist."""
    for d in (UPLOAD_DIR, OUTPUT_NUMPY_DIR, OUTPUT_TF_DIR):
        os.makedirs(d, exist_ok=True)

def _get_queue() -> Optional[Queue]:
    if not REDIS_URL:
        return None
    try:
        conn = redis.from_url(REDIS_URL)
        # verify connectivity; if Redis is down fall back to sync
        conn.ping()
        return Queue(name=QUEUE_NAME, connection=conn)
    except Exception as exc:
        app_logger.warning(f"Redis queue unavailable: {exc}")
        return None

@app.post("/upload")
async def upload_pickle(file: UploadFile = File(...)):
    """
    Upload a .pkl training or testing file. Temporarily stored in uploads.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise error_response(400, f"Unsupported content type: {file.content_type}")
    safe_name = os.path.basename(file.filename or "")
    if not safe_name.endswith(".pkl"):
        raise error_response(400, "Only .pkl files are accepted.")

    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    # size check
    if os.path.getsize(file_path) > MAX_UPLOAD_BYTES:
        os.remove(file_path)
        raise error_response(400, "File too large. Limit is 10 MB.")

    try:
        stats = Trainer.dataset_stats(file_path)
    except Exception as e:
        raise error_response(400, f"File uploaded but could not parse dataset: {e}")

    log_event("upload.success", filename=safe_name, stored_name=unique_name, bytes=os.path.getsize(file_path))
    return {"message": "File uploaded successfully", "stored_filename": unique_name, "original_filename": safe_name, "stats": stats}

@app.post("/train")
async def train_model(
    pkl_filename: str = Form(...),
    hidden_sizes: str = Form(...),
    Lambda: float = Form(...),
    epochs: int = Form(...),
    learning_rate: float = Form(...),
    train_val_split: float = Form(...),
    beta1: float = Form(0.9),
    beta2: float = Form(0.999),
    epsilon: float = Form(1e-8),
    early_stop_patience: Optional[int] = Form(None),
    lr_decay: Optional[float] = Form(None),
    activation: str = Form("relu"),
    weight_init: str = Form("auto"),
    batch_size: Optional[int] = Form(64),
    grad_clip: Optional[float] = Form(5.0),
    seed: Optional[int] = Form(None),
    model_type: str = Form("numpy"),
):
    """
    Train the neural network model using provided hyperparameters and uploaded .pkl file.
    Enqueues to Redis/RQ when available; falls back to synchronous execution otherwise.
    """
    try:
        safe_pkl = os.path.basename(pkl_filename)
        pkl_path = os.path.join(UPLOAD_DIR, safe_pkl)
        if not os.path.exists(pkl_path):
            raise error_response(400, f"File not found: {safe_pkl}")

        hidden_sizes_list = [int(x.strip()) for x in hidden_sizes.split(",") if x.strip()]
        model_type = model_type.lower()
        if model_type not in {"numpy", "tf"}:
            raise error_response(400, "model_type must be 'numpy' or 'tf'.")
        validation_error = _validate_hyperparams(
            hidden_sizes_list, Lambda, epochs, learning_rate, train_val_split, beta1, beta2, epsilon
        )
        if validation_error:
            raise error_response(400, validation_error)
        if early_stop_patience is not None and early_stop_patience <= 0:
            raise error_response(400, "early_stop_patience must be positive if provided.")
        if lr_decay is not None and not (0 < lr_decay < 1):
            raise error_response(400, "lr_decay must be between 0 and 1 if provided.")
        if activation.lower() not in {"sigmoid", "tanh", "relu", "leakyrelu"}:
            raise error_response(400, "activation must be sigmoid, tanh, relu, or leakyrelu.")
        if weight_init.lower() not in {"auto", "he", "xavier"}:
            raise error_response(400, "weight_init must be auto, he, or xavier.")
        if batch_size is not None and batch_size <= 0:
            raise error_response(400, "batch_size must be positive if provided.")
        if grad_clip is not None and grad_clip <= 0:
            raise error_response(400, "grad_clip must be positive if provided.")
        if seed is not None and seed < 0:
            raise error_response(400, "seed must be non-negative if provided.")

        job_kwargs = {
            "pkl_path": pkl_path,
            "hidden_sizes": hidden_sizes_list,
            "Lambda": Lambda,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "train_val_split": train_val_split,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "early_stop_patience": early_stop_patience,
            "lr_decay": lr_decay,
            "activation": activation,
            "weight_init": weight_init,
            "batch_size": batch_size,
            "grad_clip": grad_clip,
            "seed": seed,
            "model_type": model_type,
            "output_numpy_dir": OUTPUT_NUMPY_DIR,
            "output_tf_dir": OUTPUT_TF_DIR,
        }

        queue = _get_queue()
        if queue:
            try:
                job = queue.enqueue(run_training_job, kwargs=job_kwargs)
                job.meta["params"] = {k: v for k, v in job_kwargs.items() if k != "pkl_path"}
                job.save_meta()
                log_event("train.enqueued", backend=model_type, job_id=job.id)
                return {"job_id": job.id, "status": "queued", "backend": model_type}
            except Exception as exc:
                app_logger.warning(f"Falling back to synchronous training, enqueue failed: {exc}")

        result = await run_in_threadpool(run_training_job, **job_kwargs)
        log_event("train.completed", backend=model_type, duration_ms=result.get("duration_ms"))
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise error_response(500, str(e))

@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    """
    Return status/result for a queued training job.
    """
    queue = _get_queue()
    if not queue:
        raise error_response(503, "Job queue unavailable (set REDIS_URL to enable).")
    job = queue.fetch_job(job_id)
    if not job:
        raise error_response(404, f"Job not found: {job_id}")
    status = job.get_status()
    response = {
        "job_id": job.id,
        "status": status,
        "backend": job.meta.get("params", {}).get("model_type"),
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
    }
    if status == "finished":
        response["result"] = job.meta.get("result")
    if status == "failed":
        response["error"] = str(job.exc_info) if job.exc_info else "Job failed"
    return response

@app.post("/predict")
async def predict(
    hidden_sizes: Optional[str] = Form(None),  # optional; validated if provided
    Lambda: Optional[float] = Form(None),      # optional; validated if provided
    input_file: Optional[UploadFile] = File(None),
    input_values: Optional[str] = Form(None),
    model_type: str = Form("numpy"),
):
    """
    Run predictions using trained model.
    Uses the trained model architecture and weights saved during /train.
    Accepts either:
      - Uploaded .pkl file
      - Comma-separated input values (5 floats)
    """
    try:
        model_type = model_type.lower()
        if model_type not in {"numpy", "tf"}:
            raise error_response(400, "model_type must be 'numpy' or 'tf'.")

        async def _load_input_from_upload(upload: UploadFile) -> np.ndarray:
            if not upload.filename.endswith(".pkl"):
                raise ValueError("Only .pkl files are accepted.")
            data = await upload.read()
            payload = pickle.loads(data)
            if isinstance(payload, dict) and "X" in payload:
                payload = payload["X"]
            X_arr = np.asarray(payload, dtype=float)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)
            if X_arr.shape[1] != 5:
                raise ValueError(f"Input must have exactly 5 features; got shape {X_arr.shape}")
            return X_arr

        def _load_input_from_values(values_str: str) -> np.ndarray:
            vals = np.array([float(x.strip()) for x in values_str.split(",")], dtype=float)
            if vals.size != 5:
                raise ValueError("Input must contain exactly 5 comma-separated values.")
            return vals.reshape(1, -1)

        if model_type == "tf":
            try:
                from fivedreg_tf.tester_tf import TesterTF
            except Exception as e:
                raise error_response(500, f"TensorFlow backend unavailable: {e}")
            model_path = os.path.join(OUTPUT_TF_DIR, "model_tf.keras")
            norm_path = os.path.join(OUTPUT_TF_DIR, "normalisation_values_tf.npz")
            if not (os.path.exists(model_path) and os.path.exists(norm_path)):
                raise error_response(400, "TensorFlow model not trained yet. Train a TF model first.")
            tester_tf = TesterTF(directory=OUTPUT_TF_DIR)
            try:
                if input_file:
                    X_arr = await _load_input_from_upload(input_file)
                elif input_values:
                    X_arr = _load_input_from_values(input_values)
                else:
                    raise error_response(400, "Provide either a .pkl file or input values.")
            except ValueError as e:
                raise error_response(400, str(e))
            started = time.monotonic()
            y_pred = await run_in_threadpool(tester_tf.predict, X_arr)
            duration_ms = (time.monotonic() - started) * 1000
            log_event("predict.completed", backend="tf", samples=len(X_arr), duration_ms=round(duration_ms, 2))
            return {"y_pred": y_pred.tolist(), "model_type": "tf"}

        # NumPy model path
        try:
            metadata = Tester.load_metadata(directory=OUTPUT_NUMPY_DIR)
        except FileNotFoundError:
            raise error_response(400, "Model not trained yet. Train a NumPy model first or set model_type=tf.")

        trained_hidden_sizes = metadata.get("hidden_sizes")
        trained_lambda = metadata.get("Lambda")

        if not trained_hidden_sizes or trained_lambda is None:
            return JSONResponse(status_code=500, content={"error": "Model metadata incomplete. Train the model again."})

        # Optional validation if client sends params
        if hidden_sizes:
            client_hidden = [int(x.strip()) for x in hidden_sizes.split(",") if x.strip()]
            if client_hidden != [int(x) for x in trained_hidden_sizes]:
                raise error_response(400, "Hidden sizes mismatch. Predictions use the trained model architecture. Reset/retrain to change it.")
        if Lambda is not None and float(Lambda) != float(trained_lambda):
            raise error_response(400, "Lambda mismatch. Predictions use the trained model configuration. Reset/retrain to change it.")

        tester = Tester(
            hidden_sizes=[int(x) for x in trained_hidden_sizes],
            Lambda=float(trained_lambda),
            directory=OUTPUT_NUMPY_DIR,
            activation=metadata.get("activation", "sigmoid"),
            weight_init=metadata.get("weight_init", "auto"),
        )

        try:
            if input_file:
                X_arr = await _load_input_from_upload(input_file)
            elif input_values:
                X_arr = _load_input_from_values(input_values)
            else:
                raise error_response(400, "Provide either a .pkl file or input values.")
        except ValueError as e:
            raise error_response(400, str(e))

        started = time.monotonic()
        y_pred = await run_in_threadpool(tester.predict, X_arr)
        duration_ms = (time.monotonic() - started) * 1000
        log_event("predict.completed", backend="numpy", samples=len(X_arr), duration_ms=round(duration_ms, 2))
        return {"y_pred": y_pred.tolist()}

    except HTTPException:
        raise
    except Exception as e:
        raise error_response(500, str(e))

@app.get("/plots/{filename}")
async def get_plot(filename: str, model_type: Optional[str] = None):
    """
    Serve saved plots from outputs_numpy (NumPy) or outputs_tf (TF).
    """
    search_dirs = []
    if model_type:
        mt = model_type.lower()
        if mt == "numpy":
            search_dirs = [OUTPUT_NUMPY_DIR]
        elif mt == "tf":
            search_dirs = [OUTPUT_TF_DIR]
    if not search_dirs:
        search_dirs = [OUTPUT_NUMPY_DIR, OUTPUT_TF_DIR]

    for directory in search_dirs:
        plot_path = os.path.join(directory, filename)
        if os.path.exists(plot_path):
            return FileResponse(plot_path)
    raise error_response(404, f"Plot not found: {filename}")

@app.get("/artifacts/{filename}")
async def get_artifact(filename: str):
    """
    Serve saved artifacts (weights, norm values, metadata).
    """
    if filename not in ALLOWED_ARTIFACTS:
        raise error_response(404, f"Artifact not allowed: {filename}")
    for directory in (OUTPUT_NUMPY_DIR, OUTPUT_TF_DIR):
        artifact_path = os.path.join(directory, filename)
        if os.path.exists(artifact_path):
            return FileResponse(artifact_path)
    raise error_response(404, f"Artifact not found: {filename}")


@app.post("/evaluate")
async def evaluate_model(file: UploadFile = File(...)):
    """
    Evaluate the trained model on a provided .pkl with X and y.
    Returns RMSE on the provided dataset.
    """
    try:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise error_response(400, f"Unsupported content type: {file.content_type}")
        safe_name = os.path.basename(file.filename or "")
        if not safe_name.endswith(".pkl"):
            raise error_response(400, "Only .pkl files are accepted.")
        temp_path = os.path.join(UPLOAD_DIR, f"eval_{uuid.uuid4().hex}_{safe_name}")
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        if os.path.getsize(temp_path) > MAX_UPLOAD_BYTES:
            os.remove(temp_path)
            raise error_response(400, "File too large. Limit is 10 MB.")

        # ensure trained model exists (prefer numpy if available)
        model_type = "numpy"
        tester = None
        if os.path.exists(os.path.join(OUTPUT_NUMPY_DIR, "model_metadata.json")):
            metadata = Tester.load_metadata(directory=OUTPUT_NUMPY_DIR)
            tester = Tester(
                hidden_sizes=[int(x) for x in metadata.get("hidden_sizes", [])],
                Lambda=float(metadata.get("Lambda", 0.01)),
                directory=OUTPUT_NUMPY_DIR,
                activation=metadata.get("activation", "sigmoid"),
                weight_init=metadata.get("weight_init", "auto"),
            )
        elif os.path.exists(os.path.join(OUTPUT_TF_DIR, "model_tf.keras")):
            try:
                from fivedreg_tf.tester_tf import TesterTF
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": f"TensorFlow backend unavailable: {e}"})
            tester = TesterTF(directory=OUTPUT_TF_DIR)
            model_type = "tf"
        else:
            raise error_response(400, "Model not trained yet. Train before evaluating.")

        # load data to get y
        X, y = Trainer.load_raw_data(temp_path)
        y = y.reshape(-1, 1)
        started = time.monotonic()
        y_pred = await run_in_threadpool(tester.predict, X)
        duration_ms = (time.monotonic() - started) * 1000
        rmse = float(Trainer.calc_rmse(y, y_pred))
        log_event("evaluate.completed", backend=model_type, samples=int(len(y)), rmse=rmse, duration_ms=round(duration_ms, 2))

        os.remove(temp_path)
        return {"rmse": rmse, "samples": int(len(y)), "model_type": model_type}
    except Exception as e:
        raise error_response(500, str(e))

@app.post("/reset")
async def reset_directories():
    """
    Clears all uploaded files and generated outputs (plots, models, etc.) for both NumPy and TF backends.
    """
    try:
        # Avoid removing mount points (Docker bind volumes) by clearing contents only
        clear_directories()
        _ensure_dirs()

        return {"message": "All uploads and outputs cleared successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
