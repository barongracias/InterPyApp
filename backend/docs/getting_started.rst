Getting Started
===============

Sources
-------

GitHub: https://github.com/barongracias/InterPyApp.git

The three backend packages have been uploaded to PyPI, you can find the links here:

- interpy_bg: https://pypi.org/project/interpy-bg/
- interpy_synth: https://pypi.org/project/interpy-synth/
- fivedreg_tf: https://pypi.org/project/fivedreg-tf/


Quick start (at a glance)
-------------------------
- Backend + frontend locally: ``./scripts/run_local.sh`` (creates venv, installs backend/frontend deps, runs uvicorn + Next.js)
- Backend only: ``cd backend && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.lock && uvicorn main:app --reload``
- Docker: ``./scripts/docker_build.sh && ./scripts/docker_up.sh`` (backend on :8000, frontend on :3000)

Installation (from GitHub repo)
-------------------------------

.. code-block:: bash

    git clone https://github.com/barongracias/InterPyApp.git
    cd InterPyApp/backend
    pip install -r requirements.lock   # installs pinned interpy_bg, interpy_synth, and fivedreg_tf (Python 3.11+ required; TensorFlow CPU included)

Local one-shot setup/run (backend + frontend)
---------------------------------------------

.. code-block:: bash

   ./scripts/run_local.sh
   # creates venv, installs backend (NumPy) + frontend deps, runs uvicorn and next dev
   # Requires Python 3.11+ (for TensorFlow). On macOS installs tensorflow-macos; on Linux installs tensorflow-cpu.
   # Set PYTHON_BIN=python3.11 if needed.

Docker quick start
------------------

.. code-block:: bash

    ./scripts/docker_build.sh
    ./scripts/docker_up.sh   # backend on :8000, frontend on :3000

Notes: TensorFlow uses the CPU build; GPU is not required. Frontend API URLs point to the ``backend`` service in Docker.

Deployment (quick)
------------------
- Build: ``./scripts/docker_build.sh`` (uses amd64 for TF wheel compatibility)
- Run: ``./scripts/docker_up.sh`` (stop with ``./scripts/docker_down.sh``)
- Env: see ``frontend/.env.example`` and ``backend/.env.example`` for API URLs/CORS.
- Local dev: ``scripts/run_local.sh`` creates/uses ``backend/.venv`` (git-ignored) and installs both backends with Python 3.11+ (uses ``tensorflow-macos`` on macOS, ``tensorflow-cpu`` on Linux).

Usage at a glance
-----------------
- Install backend deps: ``pip install -r requirements.lock`` (includes TensorFlow CPU build; Python 3.11+ required); then install packages editable if developing.
- Run API: ``uvicorn main:app --reload`` from ``backend/``.
- Train: POST to ``/upload`` then ``/train`` (choose ``model_type=numpy``/``tf``); predict via ``/predict``. Training returns metrics plus artifact names when complete.
- Frontend: ``npm install && npm run dev`` in ``frontend/`` (or ``./scripts/run_local.sh`` to run both).

API endpoints
-------------
- ``/health`` – service heartbeat
- ``/upload`` – accept .pkl (with ``X``/``y``) and return stored filename plus dataset stats
- ``/train`` – run training (``model_type=numpy``/``tf``) with hyperparameters in form fields; returns metrics and artifact names
- ``/predict`` – predict from a .pkl file or comma-separated values (select backend via ``model_type``)
- ``/plots/{filename}`` and ``/artifacts/{filename}`` – serve saved plots/artifacts from outputs folders
- ``/evaluate`` – compute RMSE on supplied X/y pickle (prefers NumPy artifacts, falls back to TF)
- ``/reset`` – clear uploads and outputs folders

CI
--

- GitHub Actions workflow ``.github/workflows/ci.yml`` runs backend pytest and frontend lint/build/tests on pushes/PRs.
- Use ``requirements.lock`` (backend) and ``npm ci`` (frontend) for reproducible installs. Frontend tests include a mocked-backend integration flow via Node's test runner.

Package quick start
-------------------

NumPy backend (``interpy_bg``):

.. code-block:: python

    from interpy_bg.trainer import Trainer
    from interpy_bg.tester import Tester
    from interpy_synth import synthetic_5d_pickle

    train_pkl = synthetic_5d_pickle("outputs_numpy/train.pkl", n=1000, seed=42)
    trainer = Trainer(directory="outputs_numpy", hidden_sizes=[16, 8], epochs=200)
    train_loss, val_loss = trainer.train(train_pkl)

    tester = Tester(directory="outputs_numpy", hidden_sizes=[16, 8])
    y_pred = tester.predict([[0.1, 0.2, 0.3, 0.4, 0.5]])

TensorFlow backend (``fivedreg_tf``):

.. code-block:: python

    from fivedreg_tf.trainer_tf import TrainerTF
    from fivedreg_tf.tester_tf import TesterTF
    from interpy_synth import synthetic_5d_pickle

    data_path = synthetic_5d_pickle("outputs_tf/train.pkl", n=1000, seed=42)
    trainer = TrainerTF(directory="outputs_tf", hidden_sizes=[64, 32, 16], epochs=50)
    trainer.train(data_path)

    tester = TesterTF(directory="outputs_tf")
    preds = tester.predict([[0.1, 0.2, 0.3, 0.4, 0.5]])
