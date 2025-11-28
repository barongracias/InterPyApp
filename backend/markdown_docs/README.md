<a id="main"></a>

# main

<a id="interpy_bg"></a>

# interpy\_bg

<a id="interpy_bg.tester"></a>

# interpy\_bg.tester

<a id="interpy_bg.tester.Tester"></a>

## Tester Objects

```python
class Tester(NeuralNetwork)
```

Tester class for trained feedforward neural network. Loads model weights and normalisation values, applies normalisation,
and calculates predicted outputs for given test inputs.

Inherits from NeuralNetwork.

**Attributes**:

- `directory` _str_ - Directory path to save output files.
- `mean` _np.ndarray | None_ - Mean of the input training data
- `std` _np.ndarray | None_ - Standard deviation of the input training data

<a id="interpy_bg.tester.Tester.__init__"></a>

#### \_\_init\_\_

```python
def __init__(hidden_sizes: list[int], Lambda: float, directory: str)
```

Initialise Trainer with hyperparameters and call NeuralNetwork constructor.

**Arguments**:

- `hidden_sizes` _list[int]_ - Number of neurons in each hidden layer.
- `Lambda` _float_ - L2 regularization parameter.
- `directory` _str_ - Directory path to save output files.

<a id="interpy_bg.tester.Tester.load_norm_vals"></a>

#### load\_norm\_vals

```python
def load_norm_vals(filename: str = "normalisation_values.npz",
                   directory: str = None) -> None
```

Load stored normalisation values (mean and standard deviation)

**Arguments**:

- `filename` _str_ - Name of the file.
- `directory` _str_ - Directory path to save file.

<a id="interpy_bg.tester.Tester.normalise"></a>

#### normalise

```python
def normalise(X: np.ndarray) -> np.ndarray
```

Normalise input using mean and standard deviation.

**Arguments**:

- `X` _np.ndarray_ - Input data, shape (N, 5).
  

**Returns**:

- `np.ndarray` - Normalised input data, shape (N, 5).

<a id="interpy_bg.tester.Tester.load_test_data"></a>

#### load\_test\_data

```python
@staticmethod
def load_test_data(X_data: np.ndarray | str) -> np.ndarray
```

Load test data from numpy array or pickle (.pkl) file.

**Arguments**:

- `X_data` _np.ndarray | str_ - Input data (N, 5) or path to .pkl file.
  

**Returns**:

- `np.ndarray` - Input data of shape (N, 5).

<a id="interpy_bg.tester.Tester.predict"></a>

#### predict

```python
def predict(X_data: np.ndarray | str) -> np.ndarray
```

Perform interpolation using the trained model.

**Arguments**:

- `X_data` _np.ndarray | str_ - Input data (N, 5) or path to .pkl file.
  

**Returns**:

- `np.ndarray` - Predicted outputs of shape (N, 1).

<a id="interpy_bg.logger"></a>

# interpy\_bg.logger

<a id="interpy_bg.logger.get_console_logger"></a>

#### get\_console\_logger

```python
def get_console_logger(name: str, directory: str) -> logging.Logger
```

Creates a console logger with the given name and log level.

**Arguments**:

- `name` _str_ - Name of the logger (typically __name__ of the module).
- `directory` _str_ - Directory to store log files. Must exist or be created.

**Returns**:

- `logging.Logger` - Conffigured logger instance.

<a id="interpy_bg.neural_network"></a>

# interpy\_bg.neural\_network

<a id="interpy_bg.neural_network.NeuralNetwork"></a>

## NeuralNetwork Objects

```python
class NeuralNetwork()
```

Feedforward neural network for 5D to 1 interpolation.

**Attributes**:

- `input_size` _int_ - Number of input features (fixed at 5).
- `hidden_sizes` _list[int]_ - Number of neurons in each hidden layer.
- `output_size` _int_ - Number of output neurons (fixed at 1).
- `Lambda` _float_ - L2 regularization parameter.
- `directory` _str_ - Directory path to save output files.
- `logger` _logging.Logger_ - Logger instance for class.
- `layer_sizes` _list[int]_ - Complete list of layer sizes including input, hidden, output.
- `weights` _list[np.ndarray]_ - Weight matrices for each layer connection.
- `biases` _list[np.ndarray]_ - Bias vectors for each layer (excluding input layer).

<a id="interpy_bg.neural_network.NeuralNetwork.__init__"></a>

#### \_\_init\_\_

```python
def __init__(hidden_sizes: list[int], Lambda: float, directory: str)
```

Initialize the neural network with random weights and zero biases.

**Arguments**:

- `hidden_sizes` _list[int]_ - List specifying number of neurons in each hidden layer.
- `Lambda` _float_ - L2 regularisation parameter.
- `directory` _str_ - Directory path to save output files.

<a id="interpy_bg.neural_network.NeuralNetwork.activation"></a>

#### activation

```python
def activation(z: np.ndarray) -> np.ndarray
```

Apply the sigmoid activation function element-wise to the input.
Numerically stable version to avoid overflow for large negative values.

**Arguments**:

- `z` _np.ndarray_ - Input array of any shape.
  

**Returns**:

- `np.ndarray` - Sigmoid activation applied element-wise to keep the same shape as z.

<a id="interpy_bg.neural_network.NeuralNetwork.forward"></a>

#### forward

```python
def forward(X: np.ndarray) -> np.ndarray
```

Perform a forward pass through the network.

**Arguments**:

- `X` _np.ndarray_ - Input data of shape (N, 5) where N is the number of data points.
  

**Returns**:

- `np.ndarray` - Network output of shape (N, 1).

<a id="interpy_bg.neural_network.NeuralNetwork.activation_deriv"></a>

#### activation\_deriv

```python
def activation_deriv(z: np.ndarray) -> np.ndarray
```

Derivative of the sigmoid activation function applied element-wise.
The derivative is set as sigmoid(z) * (1 - sigmoid(z)).

**Arguments**:

- `z` _np.ndarray_ - Input array of any shape (pre-activation values).
  

**Returns**:

- `np.ndarray` - Element-wise derivative of the sigmoid, same shape as z.

<a id="interpy_bg.neural_network.NeuralNetwork.cost_function"></a>

#### cost\_function

```python
def cost_function(X: np.ndarray, y: np.ndarray) -> float
```

Compute the cost (loss) of the neural network for given inputs and targets.

The cost equals mean squared error (MSE) plus L2 regularisation on the weights.

**Arguments**:

- `X` _np.ndarray_ - Input data of shape (N, 5), where N is the number of data points.
- `y` _np.ndarray_ - True target values of shape (N, 1).
  

**Returns**:

- `float` - Scalar cost value.

<a id="interpy_bg.neural_network.NeuralNetwork.backprop"></a>

#### backprop

```python
def backprop(X: np.ndarray, y: np.ndarray,
             y_hat: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]
```

Calculate the gradients of the cost function w.r.t. to weights and biases using backpropagation.

**Arguments**:

- `X` _np.ndarray_ - Input data of shape (N, 5), where N is the number of data points.
- `y` _np.ndarray_ - True target values of shape (N, 1).
- `y_hat` _np.ndarray_ - Predicted target values of shape (N, 1).
  

**Returns**:

- `tuple` - Two lists:
  - dW: list of np.ndarray, gradients of weights for each layer
  - db: list of np.ndarray, gradients of biases for each layer

<a id="interpy_bg.neural_network.NeuralNetwork.save_weights"></a>

#### save\_weights

```python
def save_weights(filename: str = "model_weights.npz",
                 directory: str = None) -> None
```

Save the weights and biases to backend/outputs.

**Arguments**:

- `filename` _str_ - Name of the file.
- `directory` _str_ - Directory path to save file.

<a id="interpy_bg.neural_network.NeuralNetwork.load_weights"></a>

#### load\_weights

```python
def load_weights(filename: str = "model_weights.npz",
                 directory: str = None) -> None
```

Load the weights and biases from backend/outputs.

**Arguments**:

- `filename` _str_ - Name of the file to load.
- `directory` _str_ - Directory path to save file.

<a id="interpy_bg.plotter"></a>

# interpy\_bg.plotter

<a id="interpy_bg.plotter.plot_loss"></a>

#### plot\_loss

```python
def plot_loss(train_loss: list[float],
              val_loss: list[float],
              filename: str = "rmse_vs_epochs.png",
              directory: str = None) -> None
```

Plot training and validation RMSE vs epochs and save the figure in high-quality format.

**Arguments**:

- `train_loss` _list[float]_ - Training RMSE per epoch.
- `val_loss` _list[float]_ - Validation RMSE per epoch.
- `filename` _str_ - Name of the file.
- `directory` _str_ - Directory path to save file.

<a id="interpy_bg.plotter.plot_predictions"></a>

#### plot\_predictions

```python
def plot_predictions(y_true: list[float],
                     y_pred: list[float],
                     filename: str = "ytrue_vs_ypred.png",
                     directory: str = None) -> None
```

Plot predicted vs true values for the model and save as a figure.

**Arguments**:

- `y_true` _list[float]_ - True target values.
- `y_pred` _list[float]_ - Predicted values from the neural network.
- `filename` _str_ - Name of the file.
- `directory` _str_ - Directory path to save file.

<a id="interpy_bg.trainer"></a>

# interpy\_bg.trainer

<a id="interpy_bg.trainer.Trainer"></a>

## Trainer Objects

```python
class Trainer(NeuralNetwork)
```

Trainer class for feedforward neural network.

Inherits from NeuralNetwork and adds functionality for training, calculating RMSE and saving trained models.

**Attributes**:

- `epochs` _int_ - Number of training iterations.
- `learning_rate` _float_ - Step size for gradient descent updates.
- `train_val_split` _float_ - Fraction of data used for training.
- `train_loss_history` _list[float]_ - RMSE per epoch for training set.
- `val_loss_history` _list[float]_ - RMSE per epoch for validation set.
- `directory` _str_ - Directory path to save output files.
- `mean` _np.ndarray | None_ - Mean of the input training data
- `std` _np.ndarray | None_ - Standard deviation of the input training data

<a id="interpy_bg.trainer.Trainer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(hidden_sizes: list[int], Lambda: float, epochs: int,
             learning_rate: float, train_val_split: float, directory: str)
```

Initialise Trainer with hyperparameters and call NeuralNetwork constructor.

**Arguments**:

- `hidden_sizes` _list[int]_ - Number of neurons in each hidden layer.
- `Lambda` _float_ - L2 regularization parameter.
- `epochs` _int_ - Number of training iterations.
- `learning_rate` _float_ - Learning rate for gradient descent.
- `train_val_split` _float_ - Fraction of dataset used for training.
- `directory` _str_ - Directory path to save output files.

<a id="interpy_bg.trainer.Trainer.load_dataset"></a>

#### load\_dataset

```python
@staticmethod
def load_dataset(pkl_path: str, train_frac: float = 0.7,
                 val_frac: float = 0.15, test_frac: float = 0.15,
                 random_state: int | None = None) -> dict
```

Load and prepare dataset from a pickle file (dict with keys "X" and "y" or legacy tuple/list). Validates shapes, imputes NaNs with column means, shuffles, splits into train/val/test, and standardises features using the train split.

**Arguments**:

- `pkl_path` _str_ - Path to the pickle file containing training data.
- `train_frac` _float_ - Fraction of data for training.
- `val_frac` _float_ - Fraction of data for validation.
- `test_frac` _float_ - Fraction of data for testing.
- `random_state` _int | None_ - Optional seed for reproducible shuffling.
  

**Returns**:

- `dict` - Normalised splits and mean/std (`X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`, `mean`, `std`)

<a id="interpy_bg.trainer.Trainer.norm_vals"></a>

#### norm\_vals

```python
def norm_vals(X_train: np.ndarray) -> None
```

Calculate and store mean and standard deviation into instance for normalisation.

**Arguments**:

- `X_train` _np.ndarray_ - Input training data.

<a id="interpy_bg.trainer.Trainer.normalise"></a>

#### normalise

```python
def normalise(X: np.ndarray) -> np.ndarray
```

Normalise input using mean and standard deviation.

**Arguments**:

- `X` _np.ndarray_ - Input data, shape (N, 5).
  

**Returns**:

- `np.ndarray` - Normalised input data, shape (N, 5).

<a id="interpy_bg.trainer.Trainer.save_norm_vals"></a>

#### save\_norm\_vals

```python
def save_norm_vals(filename: str = "normalisation_values.npz",
                   directory: str = None) -> None
```

Save normalisation values to outputs.

**Arguments**:

- `filename` _str_ - Name of the file.
- `directory` _str_ - Directory path to save file.

<a id="interpy_bg.trainer.Trainer.calc_rmse"></a>

#### calc\_rmse

```python
@staticmethod
def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float
```

Calculate root mean squared error between predictions and true values.

**Arguments**:

- `y_true` _np.ndarray_ - True target values, shape (N, 1).
- `y_pred` _np.ndarray_ - Predicted values, shape (N, 1).
  

**Returns**:

- `float` - RMSE value.

<a id="interpy_bg.trainer.Trainer.train"></a>

#### train

```python
def train(pkl_path: str) -> tuple[list[float], list[float]]
```

Train the neural network using gradient descent and track RMSE. Saves RMSE vs epochs plot.

**Arguments**:

- `pkl_path` _str_ - Path to .pkl file containing training data.
  

**Returns**:

- `tuple` - Two lists of floats:
  - train_loss_history: RMSE for training set per epoch
  - val_loss_history: RMSE for validation set per epoch
