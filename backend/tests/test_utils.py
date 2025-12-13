import os
import pickle
import types
import numpy as np
import pytest

from interpy_bg.trainer import Trainer
from interpy_bg.utils import timer, log_call


def test_timer_and_log_call_decorators_capture_with_logger():
    logs = []

    class Dummy:
        def __init__(self):
            self.logger = types.SimpleNamespace(debug=lambda msg: logs.append(msg))

        @log_call
        @timer
        def do_work(self, x):
            return x * 2

    d = Dummy()
    result = d.do_work(3)
    assert result == 6
    # Both decorators should emit debug logs
    assert any("[CALL]" in msg for msg in logs)
    assert any("[TIMER]" in msg for msg in logs)


def test_trainer_load_raw_data_validates_shape(tmp_path):
    bad_path = tmp_path / "bad.pkl"
    with open(bad_path, "wb") as f:
        pickle.dump({"X": np.random.rand(10, 4), "y": np.random.rand(10, 1)}, f)
    with pytest.raises(ValueError):
        Trainer.load_raw_data(str(bad_path))


def test_trainer_dataset_stats(tmp_path):
    data = {"X": np.random.rand(5, 5), "y": np.random.rand(5, 1)}
    pkl_path = tmp_path / "data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    stats = Trainer.dataset_stats(str(pkl_path))
    assert stats["rows"] == 5
    assert stats["features"] == 5
    assert len(stats["x_min"]) == 5
    assert len(stats["x_max"]) == 5
    assert isinstance(stats["y_min"], float)
    assert isinstance(stats["y_max"], float)
