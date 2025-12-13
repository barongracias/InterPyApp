# interpy_synth

Lightweight synthetic data generator for 5D → 1D regression workloads.
Provides helper functions to create NumPy arrays or persist pickles with metadata.
Used by both the NumPy (`interpy_bg`) and TensorFlow (`fivedreg_tf`) packages, but can be installed standalone.

- Docs: https://interpyapp.readthedocs.io/en/latest/index.html#
- Source: https://github.com/barongracias/InterPyApp
- Version: 0.1.3 (synthetic helpers shared by both backends)

## Installation

```bash
pip install interpy-synth
# or from this repo:
# pip install git+https://github.com/barongracias/InterPyApp.git#egg=interpy-synth&subdirectory=backend/interpy_synth
```

## Usage

```python
from interpy_synth import synthetic_5d, synthetic_5d_pickle

X, y = synthetic_5d(1000, seed=42)
path = synthetic_5d_pickle("outputs_numpy/synth.pkl", n=1000, seed=42)
```

`synthetic_5d` returns float32 arrays shaped `(n, 5)` and `(n, 1)`.  
`synthetic_5d_pickle` writes a pickle with `X`, `y`, and metadata (feature names, seed, timestamp).
