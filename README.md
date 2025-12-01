# InterPyApp

Full-stack project for 5D → 1D interpolation. Includes a FastAPI backend (with NumPy and TensorFlow backends), a shared synthetic data package, and a Next.js frontend.

## Repository layout
- `backend/` — FastAPI app (`main.py`), Python packages, and docs
  - `interpy_bg/` — NumPy implementation (`pip install interpy_bg`)
  - `fivedreg/` — TensorFlow implementation (`pip install fivedreg`)
  - `interpy_synth/` — synthetic data utilities (`pip install interpy-synth`)
  - `docs/` — Sphinx docs for the backend packages
  - `tests/` — backend test suites
- `frontend/` — Next.js UI
- `scripts/`, `coursework/`, etc. — project-specific assets

## Backend quick start
Requires Python 3.10+.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt       # installs interpy_bg, interpy_synth, fivedreg (TF optional)
uvicorn main:app --reload             # start FastAPI on :8000
```

Key endpoints (see `backend/main.py`):
- `/health`, `/upload`, `/train`, `/predict`, `/plots/{file}`, `/artifacts/{file}`, `/reset`
- `/train` and `/predict` accept `model_type` of `numpy` or `tf`

Synthetic data helpers come from `interpy_synth`:

```python
from interpy_synth import synthetic_5d, synthetic_5d_pickle
X, y = synthetic_5d(1000, seed=42)
path = synthetic_5d_pickle("outputs/train.pkl", n=1000, seed=42)
```

## Frontend quick start
Requires Node.js (Next.js 16).

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

## Tests
- Backend: `pytest backend/tests tests`
- Frontend: `npm run lint` (add your test script if present)

## Packaging (PyPI)
Run from each package directory:

```bash
python -m build --no-isolation   # creates dist/ wheel + sdist
twine upload dist/*              # when ready to publish
```

Packages: `backend/interpy_synth`, `backend` (interpy_bg), `backend/fivedreg`.

## Documentation
- Sphinx docs: `cd backend/docs && make html`
- Read the Docs configuration: `.readthedocs.yaml`