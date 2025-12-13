==========================
InterPyApp
==========================

InterPyApp is a full-stack project for 5D → 1D interpolation. It lets you upload data, train NumPy or TensorFlow models, monitor metrics/plots, and download artifacts via an API and a guided frontend. It includes:

- A FastAPI backend (`backend/main.py`) that exposes training, prediction, and artifact endpoints.
- Two installable ML backends:
  - `interpy_bg`: NumPy implementation with training/testing/plotting utilities.
  - `fivedreg_tf`: TensorFlow/Keras implementation (CPU-only).
- A shared synthetic data helper package: `interpy_synth`.
- A separate frontend (not documented here) that consumes the API.

End-to-end workflow
-------------------

- Upload a `.pkl` file containing `X` (n,5) and `y` (n,1) or generate synthetic data. Uploads are content-type checked and stored with UUID-prefixed names; use the returned ``stored_filename`` when calling ``/train``.
- Train using `model_type=numpy` or `model_type=tf` with configurable hyperparameters.
- Artifacts: weights (`model_weights.npz` or `model_tf.keras`), normalisation values, metadata JSON, plots (`rmse_vs_epochs.png`, `ytrue_vs_ypred.png`).
- Predict from arrays or `.pkl`, and download plots/artifacts via `/plots` and `/artifacts`.
- Outputs stored in `backend/outputs_numpy` (NumPy) and `backend/outputs_tf` (TF); uploads in `backend/uploads`; `/reset` clears them.

Telemetry and logging
---------------------
- API endpoints emit structured log events (`upload.success`, `train.completed`, `predict.completed`, `evaluate.completed`) with backend tags and duration/RMSE summaries.
- Training/prediction functions are wrapped in shared `timer`/`log_call` decorators for lightweight instrumentation; NumPy logs every epoch and TensorFlow now logs at the same cadence (~20 times per run) for parity.

Frontend (high level)
---------------------
- Upload training data and view dataset stats.
- Configure hyperparameters (hidden sizes, Lambda, epochs, activation/init, batch size, grad clip, early stop, lr decay, seed, Adam betas/epsilon) and choose backend (NumPy or TensorFlow; TF respects activation/init and Adam betas/epsilon).
- Trigger training, view summaries, and download plots/artifacts.
- Run predictions from values or `.pkl` test files; switch between NumPy/TF models.

.. toctree::
   :maxdepth: 1

   architecture

Hyperparameter guide (UI/API)
-----------------------------
- ``hidden_sizes``: Layer widths per hidden layer; more/larger layers add capacity, training time, and overfitting risk.
- ``Lambda``: L2 regularisation strength; higher shrinks weights harder to curb overfitting but can underfit.
- ``activation``: ReLU default; LeakyReLU avoids dead units; tanh/sigmoid bound outputs and may slow training.
- ``weight_init``: Auto chooses He for ReLU/LeakyReLU and Xavier for tanh/sigmoid; override to experiment.
- ``epochs``: Full passes over the data. More epochs can fit better but take longer and may overfit.
- ``learning_rate``: Step size for gradient updates. Higher learns faster but may diverge; lower is steadier.
- ``train_val_split``: Fraction for training vs validation/early stopping. Smaller training splits reduce fit quality.
- ``batch_size``: Samples per gradient step. Larger batches smooth updates but use more memory; blank/full-batch is allowed.
- ``grad_clip``: Upper bound on gradient norm to prevent exploding gradients. Lower clips more aggressively.
- ``lr_decay``: Multiplier (<1) applied per epoch to the learning rate. Leave unset to keep LR constant.
- ``early_stop_patience``: Stop after this many epochs without validation improvement; lower stops sooner to limit overfitting.
- ``beta1`` / ``beta2``: Adam momentum terms for first/second moments; higher values smooth updates but react slower.
- ``epsilon``: Small constant for numerical stability in Adam; keep default unless debugging NaNs.
- ``seed``: Set for deterministic initialisation/shuffling; leave unset for nondeterministic runs.
- Applicability: NumPy backend uses all fields; TensorFlow backend honours activation, weight_init, batch_size, grad_clip, lr_decay, early_stop_patience, beta1/beta2/epsilon, learning_rate, hidden_sizes, Lambda, train_val_split, seed. TensorFlow (CPU) is installed by default via the pinned requirements.

Getting Started
===============

Installation (from repo)
------------------------

.. code-block:: bash

    git clone https://github.com/barongracias/InterPyApp.git
    cd InterPyApp/backend
    pip install -r requirements.lock   # installs pinned interpy_bg, interpy_synth, and fivedreg_tf (Python 3.11+ required; TensorFlow CPU included)

PyPI installs
-------------

.. code-block:: bash

    pip install interpy_bg     # NumPy backend (pulls interpy-synth)
    pip install fivedreg_tf    # TensorFlow backend (pulls interpy-synth + tensorflow)

Backend API quick start
-----------------------

Run the FastAPI server (from `backend/`):

.. code-block:: bash

    uvicorn main:app --reload

Key endpoints:
- `/train` and `/predict` support `model_type` of `numpy` or `tf`.
- `/upload`, `/evaluate` (RMSE on supplied X/y pickle; prefers NumPy artifacts, falls back to TF), `/plots/{filename}`, `/artifacts/{filename}`, `/reset`.

Docker quick start
------------------

.. code-block:: bash

    ./scripts/docker_build.sh
    ./scripts/docker_up.sh   # backend on :8000, frontend on :3000

Notes: TensorFlow uses the CPU build; GPU is not required. Frontend API URLs point to the `backend` service in Docker.

Usage at a glance
-----------------
- Install backend deps: ``pip install -r requirements.lock`` (includes TensorFlow CPU build; Python 3.11+ required); then install packages editable if developing.
- Run API: ``uvicorn main:app --reload`` from ``backend/``.
- Train: POST to ``/upload`` then ``/train`` (choose ``model_type=numpy``/``tf``); predict via ``/predict``. When ``REDIS_URL`` is set, `/train` pings Redis and enqueues a job (status via `/jobs/{id}`); if Redis is unavailable or enqueue fails, it logs a warning and runs synchronously.
- Frontend: ``npm install && npm run dev`` in ``frontend/`` (or ``./scripts/run_local.sh`` to run both).

Deployment (quick)
------------------
- Build: ``./scripts/docker_build.sh`` (uses amd64 for TF wheel compatibility)
- Run: ``./scripts/docker_up.sh`` (stop with ``./scripts/docker_down.sh``)
- Env: see ``frontend/.env.example`` and ``backend/.env.example`` for API URLs/CORS.
- Local dev: ``scripts/run_local.sh`` creates/uses ``backend/.venv`` (git-ignored) and installs both backends with Python 3.11+ (uses ``tensorflow-macos`` on macOS, ``tensorflow-cpu`` on Linux).

CI
--

- GitHub Actions workflow `.github/workflows/ci.yml` runs backend pytest and frontend lint/build/tests on pushes/PRs.
- Use `requirements.lock` (backend) and `npm ci` (frontend) for reproducible installs. Frontend tests include a mocked-backend integration flow via Node's test runner.

Logging & outputs
-----------------
- Backend uses simple console logging (see `interpy_bg.logger` and `fivedreg_tf.logger`); decorators (`timer`, `log_call`) add timing/call logs.
- Outputs: `backend/outputs_numpy` (NumPy) and `backend/outputs_tf` (TF), uploads in `backend/uploads`. These are mounted as volumes in Docker; `/reset` clears them.

Performance/profiling
---------------------
- CPU-only: TensorFlow uses `tensorflow-cpu`; choose batch sizes/hidden layers appropriate for your CPU.
- Optimiser: TensorFlow backend prefers `tf.keras.optimizers.legacy.Adam` when available (avoids the slower Apple Silicon path) and falls back to `tf.keras.optimizers.Adam`.
- Loss curves and RMSE metadata are stored in artifacts for inspection; small synthetic datasets can produce noisy validation curves—tests allow some tolerance.

Engineering choices & best practices
------------------------------------
- Reproducibility: pinned deps and lockfiles (`backend/requirements.lock`, `backend/fivedreg_tf/requirements.lock`, `npm ci` for frontend). Docker builds run as non-root and include a healthcheck. CI uses the lockfile to avoid drift.
- Modularity: three decoupled packages (`interpy_bg`, `fivedreg_tf`, `interpy_synth`) and a FastAPI layer; synthetic data pulled into its own package so both backends stay independent.
- Logging/decorators: shared `log_call`/`timer` decorators instrument calls and timing without cluttering business logic; both backends expose lightweight console loggers.
- Vectorisation and validation: data loaders reshape/validate inputs, enforce 5 columns, impute NaNs, and use NumPy vectorised ops for efficiency and safety.
- Class design/API symmetry: `Tester` inherits from `NeuralNetwork` to reuse normalisation/forward-pass logic; TF classes mirror the NumPy API for drop-in parity (`model_type` switch).
- Tests and CI: backend unit/integration tests (API, trainers/testers, synthetic data, performance, TF small-batch); frontend tests for env docs and opt-in API check. GitHub Actions runs lint/build/tests on pushes/PRs.
- Outputs and lifecycle: artifacts and uploads isolated (`backend/outputs_numpy`, `backend/outputs_tf`, `backend/uploads`); `/reset` clears them; Docker mounts them as volumes for persistence and cleanup is scripted.
- Configurability: hyperparameters exposed via API/UI; CORS via `ALLOWED_ORIGINS`; backend selectable (`model_type`) at train/predict; frontend/env examples provided.
- Security/ops: non-root containers, CORS controls, compose healthcheck, CPU-only TF for predictable deploys, explicit ports/env wiring, and scripted local runner (`scripts/run_local.sh`).
- Docs and source: RTD hosts API/usage docs, GitHub for source; READMEs include links and package-specific notes.

Package quick start
-------------------

NumPy backend (`interpy_bg`):

.. code-block:: python

    from interpy_bg.trainer import Trainer
    from interpy_bg.tester import Tester
    from interpy_synth import synthetic_5d_pickle

    train_pkl = synthetic_5d_pickle("outputs_numpy/train.pkl", n=1000, seed=42)
    trainer = Trainer(directory="outputs_numpy", hidden_sizes=[16, 8], epochs=200)
    train_loss, val_loss = trainer.train(train_pkl)

    tester = Tester(directory="outputs_numpy", hidden_sizes=[16, 8])
    y_pred = tester.predict([[0.1, 0.2, 0.3, 0.4, 0.5]])

TensorFlow backend (`fivedreg_tf`):

.. code-block:: python

    from fivedreg_tf.trainer_tf import TrainerTF
    from fivedreg_tf.tester_tf import TesterTF
    from interpy_synth import synthetic_5d_pickle

    data_path = synthetic_5d_pickle("outputs_tf/train.pkl", n=1000, seed=42)
    trainer = TrainerTF(directory="outputs_tf", hidden_sizes=[64, 32, 16], epochs=50)
    trainer.train(data_path)

    tester = TesterTF(directory="outputs_tf")
    preds = tester.predict([[0.1, 0.2, 0.3, 0.4, 0.5]])

Performances and profiling
--------------------------

Use the bundled benchmarks to measure throughput, memory, and accuracy:

- NumPy: ``python backend/tests/test_performance_numpy.py`` (writes to ``outputs_numpy/size_<n>/``)
- TensorFlow: ``python backend/tests/test_performance_tensorflow.py`` (writes to ``outputs_tf/size_<n>/``)

Latest run (CPU, hidden sizes [64, 32, 16], 200 epochs, batch_size=None, constant LR, no early stop/decay):

.. list-table:: NumPy (interpy_bg)
   :header-rows: 1

   * - n
     - train_s
     - pred_s
     - train_mb
     - pred_mb
     - train_rmse
     - val_rmse
     - mse
     - r2
   * - 1000
     - 5.723
     - 0.0592
     - 3.04
     - 1.03
     - 0.0988
     - 0.1924
     - 0.028908
     - 0.8975
   * - 5000
     - 9.773
     - 0.0150
     - 12.42
     - 0.99
     - 0.1102
     - 0.1310
     - 0.014873
     - 0.9473
   * - 10000
     - 15.433
     - 0.0148
     - 24.70
     - 0.99
     - 0.1121
     - 0.1166
     - 0.013702
     - 0.9514

.. list-table:: TensorFlow (fivedreg_tf)
   :header-rows: 1

   * - n
     - train_s
     - pred_s
     - train_mb
     - pred_mb
     - train_rmse
     - val_rmse
     - mse
     - r2
   * - 1000
     - 9.530
     - 0.1040
     - 4.13
     - 0.24
     - 0.1163
     - 0.1262
     - 0.013592
     - 0.9518
   * - 5000
     - 31.711
     - 0.0926
     - 8.81
     - 0.24
     - 0.1388
     - 0.1365
     - 0.020859
     - 0.9261
   * - 10000
     - 59.856
     - 0.0926
     - 15.66
     - 0.23
     - 0.1343
     - 0.1127
     - 0.012607
     - 0.9553

Complexity notes (from these runs)
- Training time scales near-linearly (NumPy: ~6→15 s from 1k→10k; TF: ~10→60 s) consistent with O(n · epochs · layer_cost), though TF climbs faster on CPU.
- Prediction remains effectively O(n · hidden_sizes) with sub-0.12 s for 10k samples on both backends.
- Memory grows roughly linearly with n for NumPy (3.0→24.7 MB) and modestly for TF (4.1→15.7 MB); deeper/wider nets add parameter overhead.
- Small-n synthetic runs are a bit noisier (NumPy @1k R²≈0.9) but improve quickly with scale. Use these as baselines for [64,32,16] at 200 epochs; scaling depth/width/epochs increases the training term proportionally.

Usage at a glance
-----------------
- Install backend deps: ``pip install -r requirements.lock`` (plus ``pip install -r backend/fivedreg_tf/requirements.lock`` for TF).
- Run API: ``uvicorn main:app --reload`` from ``backend/``.
- Train: POST to ``/upload`` then ``/train`` (choose ``model_type=numpy``/``tf``); predict via ``/predict``.
- Frontend: ``npm install && npm run dev`` in ``frontend/`` (or ``./scripts/run_local.sh`` to run both).

Artifacts & outputs
-------------------
- NumPy artifacts: ``backend/outputs_numpy`` (weights, normalisation, metadata, plots).
- TF artifacts: ``backend/outputs_tf`` (``model_tf.keras``, normalisation, metadata, plots).
- Uploads: ``backend/uploads``. ``/reset`` clears uploads and both output folders; Docker mounts these as volumes.

Contents
========

.. toctree::
   :maxdepth: 1
   :caption: Package Contents

   interpy_bg
   fivedreg_tf
   interpy_synth
   modules

Resources
---------
- Documentation: https://interpyapp.readthedocs.io/en/latest/index.html# (full API/usage)
- Source: https://github.com/barongracias/InterPyApp (code, issues)

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
