fivedreg Package
================

`fivedreg` is the TensorFlow/Keras backend for 5D → 1D regression. It provides
training/testing helpers plus plotting utilities. CPU-only (`tensorflow-cpu`) is used.

Overview
--------

fivedreg provides:

- TensorFlow model builder (`tf_model.py`)
- Training utilities (`trainer_tf.py`)
- Testing utilities (`tester_tf.py`)
- Plotting and logging helpers (`plotter.py`, `logger.py`, `utils.py`)
- Synthetic data examples rely on the shared `interpy_synth` package.

Submodules
----------

.. autosummary::
   :toctree: _autosummary
   :caption: Core Modules
   :recursive:

   fivedreg.tf_model
   fivedreg.trainer_tf
   fivedreg.tester_tf
   fivedreg.plotter
   fivedreg.logger
   fivedreg.utils

Ops notes
---------
- CPU-only: depends on `tensorflow-cpu`.
- Use `requirements.lock` in `backend/fivedreg/` for reproducible installs.
- Batch size and grad clipping can help stabilise training on small datasets.
