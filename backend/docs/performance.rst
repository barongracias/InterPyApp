Performance & Profiling
=======================

Benchmarks exercise both backends over multiple dataset sizes, logging wall time, memory, and accuracy. They use synthetic data, full-batch training, constant learning rate, and 200 epochs with hidden sizes ``[64, 32, 16]``.

- NumPy: ``python backend/tests/test_performance_numpy.py`` (writes to ``outputs_numpy/size_<n>/``)
- TensorFlow: ``python backend/tests/test_performance_tensorflow.py`` (writes to ``outputs_tf/size_<n>/``)

Latest run (CPU, batch_size=None, constant LR, no early stop/decay)
-------------------------------------------------------------------

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

Complexity notes
----------------

- Training time scales near-linearly with dataset size (NumPy: ~6→15 s from 1k→10k; TF: ~10→60 s) consistent with ``O(n · epochs · layer_cost)``, though TF climbs faster on CPU.
- Prediction stays effectively ``O(n · hidden_sizes)`` with sub-0.12 s for 10k samples on both backends.
- Memory grows roughly linearly with ``n`` for NumPy (3.0→24.7 MB) and modestly for TF (4.1→15.7 MB); deeper/wider nets add parameter overhead.
- Small-n synthetic runs are noisier (NumPy @1k R²≈0.9) but improve with scale. Use these as baselines for ``[64, 32, 16]`` at 200 epochs; scaling depth/width/epochs increases the training term proportionally.
