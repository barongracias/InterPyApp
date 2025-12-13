fivedreg_tf Package
====================

`fivedreg_tf` is the TensorFlow/Keras backend for 5D → 1D regression. It provides
training/testing helpers plus plotting utilities. CPU-only (`tensorflow-cpu`) is used.

Overview
--------

fivedreg_tf provides:

- TensorFlow model builder (`tf_model.py`)
- Training utilities (`trainer_tf.py`)
- Testing utilities (`tester_tf.py`)
- Plotting and logging helpers (`plotter.py`, `logger.py`, `utils.py`)
- Synthetic data examples rely on the shared `interpy_synth` package.

API
---

.. automodule:: fivedreg_tf.tf_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fivedreg_tf.trainer_tf
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fivedreg_tf.tester_tf
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fivedreg_tf.plotter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fivedreg_tf.logger
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: fivedreg_tf.utils
   :members:
   :undoc-members:
   :show-inheritance:

Ops notes
---------
- CPU-only: depends on `tensorflow-cpu`.
- Use `requirements.lock` in `backend/fivedreg_tf/` for reproducible installs.
- Batch size and grad clipping can help stabilise training on small datasets.
