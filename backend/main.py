import os
import shutil
import pickle
import numpy as np
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from interpy_bg.trainer import Trainer
from interpy_bg.tester import Tester
from typing import List

# ----------------------
# DIRECTORIES
# ----------------------
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_ARTIFACTS = {
    "model_weights.npz",
    "normalisation_values.npz",
    "model_metadata.json",
}

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

def clear_directories():
    """Helper to clean uploads and outputs folders."""
    for d in [UPLOAD_DIR, OUTPUT_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    app_logger.info("Cleared uploads and outputs directories.")

# ----------------------
# LIFESPAN HANDLER (replaces deprecated on_event)
# ----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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

@app.post("/upload")
async def upload_pickle(file: UploadFile = File(...)):
    """
    Upload a .pkl training or testing file. Temporarily stored in uploads.
    """
    if not file.filename.endswith(".pkl"):
        return JSONResponse(status_code=400, content={"error": "Only .pkl files are accepted."})

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    # size check
    if os.path.getsize(file_path) > MAX_UPLOAD_BYTES:
        os.remove(file_path)
        return JSONResponse(status_code=400, content={"error": "File too large. Limit is 10 MB."})

    try:
        stats = Trainer.dataset_stats(file_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"File uploaded but could not parse dataset: {e}"})

    return {"message": "File uploaded successfully", "path": file_path, "stats": stats}

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
    early_stop_patience: int | None = Form(None),
    lr_decay: float | None = Form(None),
    activation: str = Form("relu"),
    weight_init: str = Form("auto"),
    batch_size: int | None = Form(64),
    grad_clip: float | None = Form(5.0),
    seed: int | None = Form(None),
    model_type: str = Form("numpy"),
):
    """
    Train the neural network model using provided hyperparameters and uploaded .pkl file.
    """
    try:
        pkl_path = os.path.join(UPLOAD_DIR, pkl_filename)
        if not os.path.exists(pkl_path):
            return JSONResponse(status_code=400, content={"error": f"File not found: {pkl_filename}"})

        hidden_sizes_list = [int(x.strip()) for x in hidden_sizes.split(",") if x.strip()]
        model_type = model_type.lower()
        if model_type not in {"numpy", "tf"}:
            return JSONResponse(status_code=400, content={"error": "model_type must be 'numpy' or 'tf'."})
        validation_error = _validate_hyperparams(
            hidden_sizes_list, Lambda, epochs, learning_rate, train_val_split, beta1, beta2, epsilon
        )
        if validation_error:
            return JSONResponse(status_code=400, content={"error": validation_error})
        if early_stop_patience is not None and early_stop_patience <= 0:
            return JSONResponse(status_code=400, content={"error": "early_stop_patience must be positive if provided."})
        if lr_decay is not None and not (0 < lr_decay < 1):
            return JSONResponse(status_code=400, content={"error": "lr_decay must be between 0 and 1 if provided."})
        if activation.lower() not in {"sigmoid", "tanh", "relu", "leakyrelu"}:
            return JSONResponse(status_code=400, content={"error": "activation must be sigmoid, tanh, relu, or leakyrelu."})
        if weight_init.lower() not in {"auto", "he", "xavier"}:
            return JSONResponse(status_code=400, content={"error": "weight_init must be auto, he, or xavier."})
        if batch_size is not None and batch_size <= 0:
            return JSONResponse(status_code=400, content={"error": "batch_size must be positive if provided."})
        if grad_clip is not None and grad_clip <= 0:
            return JSONResponse(status_code=400, content={"error": "grad_clip must be positive if provided."})
        if seed is not None and seed < 0:
            return JSONResponse(status_code=400, content={"error": "seed must be non-negative if provided."})

        if model_type == "numpy":
            trainer = Trainer(
                directory=OUTPUT_DIR,
                hidden_sizes=hidden_sizes_list,
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

            return {
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
                "artifacts": list(ALLOWED_ARTIFACTS),
            }
        else:
            try:
                from fivedreg.trainer_tf import TrainerTF
                from fivedreg.tester_tf import TesterTF
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": f"TensorFlow backend unavailable: {e}"})

            trainer_tf = TrainerTF(
                directory=OUTPUT_DIR,
                hidden_sizes=hidden_sizes_list,
                Lambda=Lambda,
                epochs=epochs,
                learning_rate=learning_rate,
                train_val_split=train_val_split,
                seed=seed,
                early_stop_patience=early_stop_patience,
                lr_decay=lr_decay,
                batch_size=batch_size,
                grad_clip=grad_clip,
            )

            train_loss, val_loss = trainer_tf.train(pkl_path)

            # recompute splits deterministically to score metrics
            splits = Trainer.load_dataset(
                pkl_path,
                train_frac=train_val_split,
                val_frac=1 - train_val_split,
                test_frac=0.0,
                random_state=seed,
            )
            X_train = splits["X_train"]
            y_train = splits["y_train"]
            X_val = splits["X_val"]
            y_val = splits["y_val"]

            tester_tf = TesterTF(directory=OUTPUT_DIR)
            y_pred_train = tester_tf.predict(X_train)
            y_pred_val = tester_tf.predict(X_val)

            train_rmse_end = float(Trainer.calc_rmse(y_train, y_pred_train))
            val_rmse_end = float(Trainer.calc_rmse(y_val, y_pred_val))
            train_r2 = Trainer.calc_r2(y_train, y_pred_train)
            val_r2 = Trainer.calc_r2(y_val, y_pred_val)
            baseline_rmse = float(Trainer.calc_rmse(y_train, np.full_like(y_train, y_train.mean())))
            best_epoch = int(np.argmin(val_loss) + 1) if val_loss else None

            # save TF metadata
            tf_metadata = {
                "hidden_sizes": hidden_sizes_list,
                "Lambda": Lambda,
                "epochs_configured": epochs,
                "epochs_run": len(train_loss),
                "best_epoch": best_epoch,
                "best_val_rmse": float(min(val_loss)) if val_loss else None,
                "best_train_rmse": float(min(train_loss)) if train_loss else None,
                "baseline_rmse": baseline_rmse,
                "final_train_r2": train_r2,
                "final_val_r2": val_r2,
                "early_stop_patience": early_stop_patience,
                "lr_decay": lr_decay,
                "seed": seed,
                "model_type": "tf",
            }
            meta_path = os.path.join(OUTPUT_DIR, "tf_model_metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(tf_metadata, f)

            return {
                "message": "Training completed successfully.",
                "model_type": "tf",
                "train_loss_start": train_loss[0] if train_loss else None,
                "train_loss_end": train_loss[-1] if train_loss else None,
                "val_loss_start": val_loss[0] if val_loss else None,
                "val_loss_end": val_loss[-1] if val_loss else None,
                "best_val_rmse": float(min(val_loss)) if val_loss else None,
                "best_train_rmse": float(min(train_loss)) if train_loss else None,
                "best_epoch": best_epoch,
                "epochs_run": len(train_loss),
                "baseline_rmse": baseline_rmse,
                "final_train_r2": train_r2,
                "final_val_r2": val_r2,
                "plots": ["rmse_vs_epochs.png", "ytrue_vs_ypred.png"],
                "artifacts": ["model_tf.keras", "normalisation_values_tf.npz", "tf_model_metadata.json"],
            }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict")
async def predict(
    hidden_sizes: str = Form(None),  # optional; validated if provided
    Lambda: float = Form(None),      # optional; validated if provided
    input_file: UploadFile = File(None),
    input_values: str = Form(None),
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
        # load trained metadata to ensure architecture matches saved weights
        metadata = Tester.load_metadata(directory=OUTPUT_DIR)
        model_type = model_type.lower()
        if model_type not in {"numpy", "tf"}:
            return JSONResponse(status_code=400, content={"error": "model_type must be 'numpy' or 'tf'."})

        if model_type == "tf":
            try:
                from fivedreg.tester_tf import TesterTF
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": f"TensorFlow backend unavailable: {e}"})
            tester_tf = TesterTF(directory=OUTPUT_DIR)

            if input_file:
                if not input_file.filename.endswith(".pkl"):
                    return JSONResponse(status_code=400, content={"error": "Only .pkl files are accepted."})
                file_path = os.path.join(UPLOAD_DIR, input_file.filename)
                with open(file_path, "wb") as f:
                    f.write(await input_file.read())
                y_pred = tester_tf.predict(file_path)
                os.remove(file_path)
            elif input_values:
                values = np.array([[float(x.strip()) for x in input_values.split(",")]], dtype=float)
                if values.shape != (1, 5):
                    return JSONResponse(status_code=400, content={"error": "Input must contain exactly 5 values."})
                y_pred = tester_tf.predict(values)
            else:
                return JSONResponse(status_code=400, content={"error": "Provide either a .pkl file or input values."})

            return {"y_pred": y_pred.tolist(), "model_type": "tf"}

        trained_hidden_sizes = metadata.get("hidden_sizes")
        trained_lambda = metadata.get("Lambda")

        if not trained_hidden_sizes or trained_lambda is None:
            return JSONResponse(status_code=500, content={"error": "Model metadata incomplete. Train the model again."})

        # Optional validation if client sends params
        if hidden_sizes:
            client_hidden = [int(x.strip()) for x in hidden_sizes.split(",") if x.strip()]
            if client_hidden != [int(x) for x in trained_hidden_sizes]:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Hidden sizes mismatch. Predictions use the trained model architecture. Reset/retrain to change it."},
                )
        if Lambda is not None and float(Lambda) != float(trained_lambda):
            return JSONResponse(
                status_code=400,
                content={"error": "Lambda mismatch. Predictions use the trained model configuration. Reset/retrain to change it."},
            )

        tester = Tester(
            hidden_sizes=[int(x) for x in trained_hidden_sizes],
            Lambda=float(trained_lambda),
            directory=OUTPUT_DIR,
            activation=metadata.get("activation", "sigmoid"),
            weight_init=metadata.get("weight_init", "auto"),
        )

        if input_file:
            if not input_file.filename.endswith(".pkl"):
                return JSONResponse(status_code=400, content={"error": "Only .pkl files are accepted."})

            file_path = os.path.join(UPLOAD_DIR, input_file.filename)
            with open(file_path, "wb") as f:
                f.write(await input_file.read())

            y_pred = tester.predict(file_path)
            os.remove(file_path)

        elif input_values:
            values = np.array([[float(x.strip()) for x in input_values.split(",")]], dtype=float)
            if values.shape != (1, 5):
                return JSONResponse(status_code=400, content={"error": "Input must contain exactly 5 values."})
            y_pred = tester.predict(values)

        else:
            return JSONResponse(status_code=400, content={"error": "Provide either a .pkl file or input values."})

        return {"y_pred": y_pred.tolist()}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/plots/{filename}")
async def get_plot(filename: str):
    """
    Serve saved plots from outputs directory.
    """
    plot_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(plot_path):
        return FileResponse(plot_path)
    return JSONResponse(status_code=404, content={"error": f"Plot not found: {filename}"})

@app.get("/artifacts/{filename}")
async def get_artifact(filename: str):
    """
    Serve saved artifacts (weights, norm values, metadata).
    """
    allowed = ALLOWED_ARTIFACTS
    if filename not in allowed:
        return JSONResponse(status_code=404, content={"error": f"Artifact not allowed: {filename}"})
    if filename in ALLOWED_ARTIFACTS and os.path.exists(os.path.join(OUTPUT_DIR, filename)):
        return FileResponse(os.path.join(OUTPUT_DIR, filename))
    return JSONResponse(status_code=404, content={"error": f"Artifact not found: {filename}"})


@app.post("/evaluate")
async def evaluate_model(file: UploadFile = File(...)):
    """
    Evaluate the trained model on a provided .pkl with X and y.
    Returns RMSE on the provided dataset.
    """
    try:
        if not file.filename.endswith(".pkl"):
            return JSONResponse(status_code=400, content={"error": "Only .pkl files are accepted."})
        temp_path = os.path.join(UPLOAD_DIR, f"eval_{file.filename}")
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        if os.path.getsize(temp_path) > MAX_UPLOAD_BYTES:
            os.remove(temp_path)
            return JSONResponse(status_code=400, content={"error": "File too large. Limit is 10 MB."})

        # ensure trained model exists (prefer numpy if available)
        model_type = "numpy"
        tester = None
        if os.path.exists(os.path.join(OUTPUT_DIR, "model_metadata.json")):
            metadata = Tester.load_metadata(directory=OUTPUT_DIR)
            tester = Tester(
                hidden_sizes=[int(x) for x in metadata.get("hidden_sizes", [])],
                Lambda=float(metadata.get("Lambda", 0.01)),
                directory=OUTPUT_DIR,
                activation=metadata.get("activation", "sigmoid"),
                weight_init=metadata.get("weight_init", "auto"),
            )
        elif os.path.exists(os.path.join(OUTPUT_DIR, "model_tf.keras")):
            try:
                from fivedreg.tester_tf import TesterTF
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": f"TensorFlow backend unavailable: {e}"})
            tester = TesterTF(directory=OUTPUT_DIR)
            model_type = "tf"
        else:
            return JSONResponse(status_code=400, content={"error": "Model not trained yet. Train before evaluating."})

        # load data to get y
        X, y = Trainer.load_raw_data(temp_path)
        y = y.reshape(-1, 1)
        y_pred = tester.predict(temp_path)
        rmse = float(Trainer.calc_rmse(y, y_pred))
        os.remove(temp_path)
        return {"rmse": rmse, "samples": int(len(y)), "model_type": model_type}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/reset")
async def reset_directories():
    """
    Clears all uploaded files and generated outputs (plots, models, etc.).
    """
    try:
        # Remove and recreate both directories
        for directory in [UPLOAD_DIR, OUTPUT_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)

        return {"message": "All uploads and outputs cleared successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
