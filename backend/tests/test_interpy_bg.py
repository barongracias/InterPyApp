import os
import pickle
import sys
import numpy as np

# make backend modules importable when running from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from interpy_bg.neural_network import NeuralNetwork
from interpy_bg.trainer import Trainer
from interpy_bg.tester import Tester
from interpy_synth import synthetic_5d


def test_neural_network_forward_and_io(tmp_path):
    X = np.random.rand(12, 5)
    y = np.random.rand(12, 1)

    nn = NeuralNetwork([8, 4], 0.01, directory=str(tmp_path))
    output = nn.forward(X)
    assert output.shape == (12, 1)

    cost = nn.cost_function(X, y)
    assert np.isfinite(cost)

    dW, db = nn.backprop(X, y, output)
    for i, (dw, dbias) in enumerate(zip(dW, db)):
        assert dw.shape == nn.weights[i].shape
        assert dbias.shape == nn.biases[i].shape

    weights_file = tmp_path / "weights.npz"
    nn.save_weights(weights_file.name, directory=str(tmp_path))
    assert weights_file.exists()

    original_weights = [w.copy() for w in nn.weights]
    nn.weights = [np.random.randn(*w.shape) for w in nn.weights]
    nn.load_weights(weights_file.name, directory=str(tmp_path))
    for loaded, original in zip(nn.weights, original_weights):
        assert np.allclose(loaded, original)


def test_trainer_tester_end_to_end(tmp_path):
    X_train, y_train = synthetic_5d(80, seed=123)
    train_pkl = tmp_path / "train.pkl"
    with open(train_pkl, "wb") as f:
        pickle.dump({"X": X_train, "y": y_train}, f)

    out_dir = tmp_path / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        directory=str(out_dir),
        hidden_sizes=[16, 8],
        Lambda=0.01,
        epochs=40,
        learning_rate=0.01,
        train_val_split=0.8,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        activation="relu",
        batch_size=16,
        seed=321,
    )
    train_loss, val_loss = trainer.train(str(train_pkl))
    assert len(train_loss) == len(val_loss) > 0
    assert np.isfinite(train_loss[-1]) and np.isfinite(val_loss[-1])

    for fname in ["model_weights.npz", "normalisation_values.npz", "model_metadata.json"]:
        assert (out_dir / fname).exists()

    tester = Tester(hidden_sizes=[16, 8], Lambda=0.01, directory=str(out_dir))

    X_test, _ = synthetic_5d(6, seed=456)
    preds = tester.predict(X_test)
    assert preds.shape == (6, 1)
    assert np.all(np.isfinite(preds))

    pkl_path = out_dir / "X_test.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(X_test, f)
    preds_pkl = tester.predict(str(pkl_path))
    assert preds_pkl.shape == (6, 1)
    assert np.all(np.isfinite(preds_pkl))
