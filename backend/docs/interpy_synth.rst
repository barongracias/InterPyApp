interpy_synth Package
=====================

`interpy_synth` is a lightweight companion package that generates the synthetic
5D → 1D data used throughout this project. Both `interpy_bg` (NumPy) and
`fivedreg_tf` (TensorFlow) depend on it for examples and tests.

Installation
------------

.. code-block:: bash

   pip install interpy-synth
   # or from this repo:
   # pip install git+https://github.com/barongracias/InterPyApp.git#egg=interpy-synth&subdirectory=backend/interpy_synth

Usage
-----

.. code-block:: python

   from interpy_synth import synthetic_5d, synthetic_5d_pickle

   # arrays
   X, y = synthetic_5d(1000, seed=42)

   # pickle with metadata
   path = synthetic_5d_pickle("outputs_numpy/synth.pkl", n=1000, seed=42)

API
---

.. automodule:: interpy_synth.synthetic
   :members:
   :undoc-members:
   :show-inheritance:
