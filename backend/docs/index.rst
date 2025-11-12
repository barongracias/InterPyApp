==========================
interpy_bg
==========================

Feedforward neural network library for 5D → 1D interpolation.
This package includes modules for training, testing, plotting, and logging.

Getting Started
===============

Installation
------------

You can install the package via PyPI (once published):

.. code-block:: bash

    pip install interpy-bg

Or install directly from the repository:

.. code-block:: bash

    git clone https://github.com/barongracias/InterPyApp.git
    cd InterPyApp/backend
    pip install .

Usage
-----

.. code-block:: python
    
    from interpy_bg.trainer import Trainer
    from interpy_bg.tester import Tester
    import numpy as np

    # Example: training
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100, 1)

    trainer = Trainer(hidden_sizes=[10, 5], Lambda=0.01, epochs=500, learning_rate=0.01, train_val_split=0.8)
    train_loss, val_loss = trainer.train(X_train, y_train)

    # Example: testing
    tester = Tester(hidden_sizes=[10, 5], Lambda=0.01)
    y_pred = tester.predict(X_train)

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Package Contents:

   interpy_bg
   modules

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`