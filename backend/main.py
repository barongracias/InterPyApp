import os
import shutil
import pickle
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from interpy_bg.trainer import Trainer
from interpy_bg.tester import Tester

# ----------------------
# DIRECTORIES
# ----------------------
UPLOAD_DIR = os.path.join("uploads")
OUTPUT_DIR = os.path.join("outputs")

def clear_directories():
    """Helper to clean uploads and outputs folders."""
    for d in [UPLOAD_DIR, OUTPUT_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    print("✅ Cleared uploads and outputs directories.")

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
@app.get("/")
async def root():
    return {"message": "Welcome to the 5D Interpolator App API"}

@app.post("/reset")
async def reset_directories():
    """
    Clear uploads and outputs.
    Triggered when user clicks 'Start Over' or reloads session.
    """
    clear_directories()
    return {"status": "success", "message": "Uploads and outputs cleared."}

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

    return {"message": "File uploaded successfully", "path": file_path}

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
    epsilon: float = Form(1e-8)
):
    """
    Train the neural network model using provided hyperparameters and uploaded .pkl file.
    """
    try:
        pkl_path = os.path.join(UPLOAD_DIR, pkl_filename)
        if not os.path.exists(pkl_path):
            return JSONResponse(status_code=400, content={"error": f"File not found: {pkl_filename}"})

        hidden_sizes_list = [int(x.strip()) for x in hidden_sizes.split(",") if x.strip()]

        trainer = Trainer(
            directory=OUTPUT_DIR,
            hidden_sizes=hidden_sizes_list,
            Lambda=Lambda,
            epochs=epochs,
            learning_rate=learning_rate,
            train_val_split=train_val_split,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon
        )

        train_loss, val_loss = trainer.train(pkl_path)

        return {
            "message": "Training completed successfully.",
            "train_loss_start": train_loss[0],
            "train_loss_end": train_loss[-1],
            "val_loss_start": val_loss[0],
            "val_loss_end": val_loss[-1],
            "plots": ["rmse_vs_epochs.png", "ytrue_vs_ypred.png"]
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict")
async def predict(
    hidden_sizes: str = Form(...),
    Lambda: float = Form(...),
    input_file: UploadFile = File(None),
    input_values: str = Form(None)
):
    """
    Run predictions using trained model.
    Accepts either:
      - Uploaded .pkl file
      - Comma-separated input values (5 floats)
    """
    try:
        tester = Tester(
            hidden_sizes=[int(x.strip()) for x in hidden_sizes.split(",") if x.strip()],
            Lambda=Lambda,
            directory=OUTPUT_DIR
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
    if not os.path.exists(plot_path):
        return JSONResponse(status_code=404, content={"error": f"Plot not found: {filename}"})
    return FileResponse(plot_path)

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