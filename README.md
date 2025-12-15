# InterPyApp

Full-stack project for 5D → 1D interpolation. Includes a FastAPI backend (with NumPy and TensorFlow backends), a shared synthetic data package, and a Next.js frontend.

- [Read the Docs](https://interpyapp.readthedocs.io/en/latest/index.html)

## Repository layout
- `backend/` — FastAPI app (`main.py`), Python packages, and docs
  - `interpy_bg/` — NumPy implementation (`pip install interpy_bg`)
  - `fivedreg_tf/` — TensorFlow implementation (`pip install fivedreg_tf`)
  - `interpy_synth/` — synthetic data utilities (`pip install interpy-synth`)
  - `docs/` — Sphinx docs for the backend packages
  - `tests/` — backend test suites
- `frontend/` — Next.js UI
- `scripts/`, `coursework/`, etc. — project-specific assets

## What the app does (end-to-end)
- Upload a `.pkl` file containing `X` (n,5) and `y` (n,1) or generate synthetic data. Uploads are content-type checked and stored with UUID-prefixed filenames; use the returned `stored_filename` for training/prediction calls.
- Train a 5D→1D regressor using either NumPy (`interpy_bg`) or TensorFlow (`fivedreg_tf`) backends (TensorFlow backend is installed by default). When `REDIS_URL` is set, `/train` pings Redis and enqueues an RQ job (returns `job_id`; check `/jobs/{job_id}` for status/result). If Redis is unavailable or enqueue fails, `/train` logs a warning and runs synchronously.
- Track RMSE and metadata; save artifacts (weights, normalisation stats, metadata JSON) and plots (RMSE vs epochs, predicted vs true). Structured logs emit timings and RMSE summaries tagged by backend.
- Make predictions from arrays or `.pkl` inputs, and download artifacts/plots via the API/Frontend.

## Docker start
- Build images: `./scripts/docker_build.sh` (uses `--platform=linux/amd64` so TensorFlow wheels resolve on Apple Silicon)
- Run stack: `./scripts/docker_up.sh` (backend on :8000, frontend on :3000, redis + worker for job queue)
- Stop stack: `./scripts/docker_down.sh`
- Compose file: `docker-compose.yml`
- Frontend env example: `frontend/.env.example` (API URLs point to `backend` service). GPU is not required; TensorFlow uses the CPU build.

## Local one-shot setup/run (backend + frontend):

```bash
./scripts/run_local.sh   # creates venv, installs backend (NumPy) + frontend deps, runs uvicorn and next dev
# Requires Python 3.11+ (for TensorFlow). On macOS installs tensorflow-macos; on Linux installs tensorflow-cpu. Set PYTHON_BIN=python3.11 if needed.
```

## Backend quick start
Requires Python 3.11+ (TensorFlow backend is required; use PYTHON_BIN=python3.11 if needed).

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock      # installs interpy_bg, interpy_synth, fivedreg_tf (TF required)
uvicorn main:app --reload             # start FastAPI on :8000
```

Key endpoints (see `backend/main.py`):
- `/health`, `/upload`, `/train`, `/jobs/{id}`, `/predict`, `/plots/{file}`, `/artifacts/{file}`, `/reset`
- `/evaluate` accepts a `.pkl` with `X` and `y` and returns RMSE (prefers NumPy artifacts, falls back to TF)
- `/train` and `/predict` accept `model_type` of `numpy` or `tf`

Synthetic data helpers come from `interpy_synth`:

```python
from interpy_synth import synthetic_5d, synthetic_5d_pickle
X, y = synthetic_5d(1000, seed=42)
path = synthetic_5d_pickle("outputs_numpy/train.pkl", n=1000, seed=42)
```

## Frontend quick start
Requires Node.js >= 18.17 (Next.js 14).

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

## Frontend UI (what you can do)
- Upload training `.pkl` (dict with `X`, `y`), view dataset stats.
- Configure training hyperparameters (hidden sizes, Lambda, epochs, activation/init, batch size, grad clip, early stop, lr decay, seed, Adam betas/epsilon) and choose backend (`numpy` default, `tf` optional; TF now honours activation/init and Adam betas/epsilon).
- Monitor training summary, view/download plots and artifacts from the UI.
- Run predictions from comma-separated values or `.pkl` test files; switch between NumPy/TF models.
- Optional toggles: training/network options panels; upload history (last few filenames).

## Hyperparameter guide (UI/API)
- `hidden_sizes`: Layer widths per hidden layer; more/larger layers add capacity, training time, and overfitting risk.
- `Lambda`: L2 regularisation strength; higher shrinks weights harder to curb overfitting but can underfit.
- `activation`: ReLU default; LeakyReLU avoids dead units; tanh/sigmoid bound outputs and may slow training.
- `weight_init`: Auto chooses He for ReLU/LeakyReLU and Xavier for tanh/sigmoid; override to experiment.
- `epochs`: Full passes over the data. More epochs can fit better but take longer and may overfit.
- `learning_rate`: Step size for gradient updates. Higher learns faster but may diverge; lower is steadier.
- `train_val_split`: Fraction for training vs validation/early stopping. Smaller training splits reduce fit quality.
- `batch_size`: Samples per gradient step. Larger batches smooth updates but use more memory; blank/full-batch is allowed.
- `grad_clip`: Upper bound on gradient norm to prevent exploding gradients. Lower clips more aggressively.
- `lr_decay`: Multiplier (<1) applied per epoch to the learning rate. Leave unset to keep LR constant.
- `early_stop_patience`: Stop after this many epochs without validation improvement; lower stops sooner to limit overfitting.
- `beta1` / `beta2`: Adam momentum terms for first/second moments; higher values smooth updates but react slower.
- `epsilon`: Small constant for numerical stability in Adam; keep default unless debugging NaNs.
- `seed`: Set for deterministic initialisation/shuffling; leave unset for nondeterministic runs.

Applicability:
- NumPy backend: all above fields apply.
- TensorFlow backend: respects activation, weight_init, batch_size, grad_clip, lr_decay, early_stop_patience, beta1/beta2/epsilon, learning_rate, hidden_sizes, Lambda, train_val_split, seed.

## Tests / CI
- Backend: `pytest backend/tests`
- Frontend: `npm run lint` and `npm test` (Node test runner; includes a mocked backend integration flow)
- CI: GitHub Actions in `.github/workflows/ci.yml` runs backend tests and frontend lint/build/tests on pushes/PRs.

## Ops & security notes
- Containers run non-root; backend has a healthcheck via compose.
- Prefer `requirements.lock` for reproducible backend installs; `npm ci` for frontend (package-lock present).
- Set production CORS/ingress and TLS at your proxy; tighten allowed origins/headers as needed.
- Logging/metrics: backend emits structured events (upload/train/predict/evaluate) with durations and RMSE; shared decorators add timing/call logs. Frontend uses typed API helpers and an error boundary to surface validation/server errors inline.
- Outputs: backend writes to `backend/outputs_numpy` (NumPy) and `backend/outputs_tf` (TF); uploads stored in `backend/uploads` using UUID-prefixed names. `/reset` (API) or restarting clears these. Compose mounts these directories as volumes.
- Local venv: `scripts/run_local.sh` creates/uses `backend/.venv` (ignored in git); optional TF install requires a supported Python/platform.
- TensorFlow optimiser: prefers `tf.keras.optimizers.legacy.Adam` when available (avoids slower Apple Silicon path) and falls back to `tf.keras.optimizers.Adam`.

## Packaging (PyPI)
The three backend packages have been uploaded to PyPI, you can find the links here:
[interpy_bg](https://pypi.org/project/interpy-bg/)
[interpy_synth](https://pypi.org/project/interpy-synth/)
[fivedreg_tf](https://pypi.org/project/fivedreg-tf/)

To build and upload the packages, run from each package directory (sdist + universal wheel), ideally in a clean env and after updating the version number:

```bash
python -m build --sdist --wheel --no-isolation
twine check dist/*
twine upload dist/*              # when ready to publish
```

Packages: `backend/interpy_synth`, `backend/interpy_bg`, `backend/fivedreg_tf`. CI builds on Linux to avoid platform-tagged wheels.

## Documentation
- Sphinx docs: `cd backend/docs && make html`
- Read the Docs configuration: `.readthedocs.yaml`

## AI/LLM Usage
I have used Codex (ChatGPT) in my coursework for the following reasons:
- Generating documentation (README and RTD), docstrings and general comments around code.
- Verifying integration between frontend, backend and middleware - primarily in the development of `main.py`.
- Generating generic ignore files.
- Generating unit tests and CI tests.
- Developing the frontend `page.tsx`.
- Reviewing overall project structure for completeness, consistency, accuracy and best practises in software engineering.