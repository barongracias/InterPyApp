==========================
InterPyApp
==========================

InterPyApp is a full stack project for 5D → 1D interpolation. It includes:

- A FastAPI backend (`backend/main.py`) that exposes training, prediction, and artifact endpoints.
- Two installable ML backends:
  - `interpy_bg`: NumPy implementation with training/testing/plotting utilities.
  - `fivedreg_tf`: TensorFlow/Keras implementation mirroring the NumPy API.
- A shared synthetic data helper package: `interpy_synth`.
- A separate frontend (not documented here) that consumes the API.

Getting Started
===============

Installation (from repo)
------------------------

.. code-block:: bash

    git clone https://github.com/barongracias/InterPyApp.git
    cd InterPyApp/backend
    pip install -r requirements.lock   # installs pinned interpy_bg, interpy_synth, and fivedreg_tf (TF optional)

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
- `/upload`, `/plots/{filename}`, `/artifacts/{filename}`, `/reset`.

Docker quick start
------------------

.. code-block:: bash

    ./scripts/docker_build.sh
    ./scripts/docker_up.sh   # backend on :8000, frontend on :3000

Notes: TensorFlow uses the CPU build; GPU is not required. Frontend API URLs point to the `backend` service in Docker.

CI
--

- GitHub Actions workflow `.github/workflows/ci.yml` runs backend pytest and frontend lint/build/tests on pushes/PRs.
- Use `requirements.lock` (backend) and `npm ci` (frontend) for reproducible installs.

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

Latest run (CPU, hidden sizes [64, 32, 16], 200 epochs):

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
     - 2.847
     - 0.0089
     - 3.60
     - 1.38
     - 0.1624
     - 0.1835
     - 0.021614
     - 0.9234
   * - 5000
     - 6.511
     - 0.0070
     - 16.33
     - 1.33
     - 0.1254
     - 0.1322
     - 0.015862
     - 0.9438
   * - 10000
     - 11.261
     - 0.0073
     - 32.52
     - 1.33
     - 0.1291
     - 0.1287
     - 0.016535
     - 0.9414

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
     - 4.454
     - 0.0904
     - 5.32
     - 0.31
     - 0.1602
     - 0.1685
     - 0.024084
     - 0.9078
   * - 5000
     - 6.699
     - 0.0895
     - 5.27
     - 0.31
     - 0.1236
     - 0.1308
     - 0.013365
     - 0.9488
   * - 10000
     - 9.177
     - 0.0939
     - 5.80
     - 0.31
     - 0.1106
     - 0.1195
     - 0.009920
     - 0.9620

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Package Contents:

   interpy_bg
   fivedreg_tf
   interpy_synth
   modules

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
