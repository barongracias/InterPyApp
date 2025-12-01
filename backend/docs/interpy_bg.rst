interpy_bg Package
==================

The `interpy_bg` package provides a feedforward neural network library for
5D → 1D interpolation. It includes functionality for training, testing,
plotting, and logging.

Overview
--------

interpy_bg provides:

- Neural network core (`neural_network.py`)
- Training utilities (`trainer.py`)
- Testing utilities (`tester.py`)
- Plotting functions (`plotter.py`)
- Logging configuration (`logger.py`)
- Synthetic data generation via the companion `interpy_synth` package (installed as a dependency)

Submodules
----------

.. autosummary::
   :toctree: _autosummary
   :caption: Core Modules
   :recursive:

   interpy_bg.neural_network
   interpy_bg.trainer
   interpy_bg.tester
   interpy_bg.plotter
   interpy_bg.logger
