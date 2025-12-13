Configuration & Operations
==========================

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

Telemetry and logging
---------------------
- API endpoints emit structured log events (``upload.success``, ``train.completed``, ``predict.completed``, ``evaluate.completed``) with backend tags and duration/RMSE summaries.
- Training/prediction functions are wrapped in shared ``timer``/``log_call`` decorators for lightweight instrumentation; NumPy logs every epoch and TensorFlow logs at the same cadence (~20 times per run) for parity.

Artifacts & outputs
-------------------
- NumPy artifacts: ``backend/outputs_numpy`` (weights, normalisation, metadata, plots).
- TF artifacts: ``backend/outputs_tf`` (``model_tf.keras``, normalisation, metadata, plots).
- Uploads: ``backend/uploads``. ``/reset`` clears uploads and both output folders; Docker mounts these as volumes.

Engineering choices & best practices
------------------------------------
- Reproducibility: pinned deps and lockfiles (``backend/requirements.lock``, ``backend/fivedreg_tf/requirements.lock``, ``npm ci`` for frontend). Docker builds run as non-root and include a healthcheck. CI uses the lockfile to avoid drift.
- Modularity: three decoupled packages (``interpy_bg``, ``fivedreg_tf``, ``interpy_synth``) and a FastAPI layer; synthetic data pulled into its own package so both backends stay independent.
- Logging/decorators: shared ``log_call``/``timer`` decorators instrument calls and timing without cluttering business logic; both backends expose lightweight console loggers.
- Vectorisation and validation: data loaders reshape/validate inputs, enforce 5 columns, impute NaNs, and use NumPy vectorised ops for efficiency and safety.
- Class design/API symmetry: TF classes mirror the NumPy API for drop-in parity (``model_type`` switch) and testers reuse saved metadata.
- Tests and CI: backend unit/integration tests (API, trainers/testers, synthetic data, performance, TF small-batch); frontend tests for env docs and opt-in API check. GitHub Actions runs lint/build/tests on pushes/PRs.
- Outputs and lifecycle: artifacts and uploads isolated (``backend/outputs_numpy``, ``backend/outputs_tf``, ``backend/uploads``); ``/reset`` clears them; Docker mounts them as volumes for persistence and cleanup is scripted.
- Configurability: hyperparameters exposed via API/UI; CORS via ``ALLOWED_ORIGINS``; backend selectable (``model_type``) at train/predict; frontend/env examples provided.
- Security/ops: non-root containers, CORS controls, compose healthcheck, CPU-only TF for predictable deploys, explicit ports/env wiring, and scripted local runner (``scripts/run_local.sh``).
- Docs and source: RTD hosts API/usage docs, GitHub for source; READMEs include links and package-specific notes.
