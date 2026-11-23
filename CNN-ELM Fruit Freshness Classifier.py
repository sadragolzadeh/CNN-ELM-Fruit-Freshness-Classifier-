# ======================================================================
# PHASE 1
# ======================================================================


import numpy as np


# ---------------------------------------------------------------------
# 1) Activation function for the hidden layer
# ---------------------------------------------------------------------
def activation_fn(Z: np.ndarray, kind: str = "sigmoid") -> np.ndarray:
    """
    Compute element-wise activation over Z.

    Args:
      Z   : np.ndarray of shape [N, L] (pre-activation hidden responses)
      kind: str, one of {'sigmoid', 'tanh', 'relu', 'linear'}

    Returns:
      H   : np.ndarray of shape [N, L] (post-activation hidden responses)
    """
    if kind == "sigmoid":
        return 1 / (1 + np.exp(-Z))
    elif kind == "tanh":
        return np.tanh(Z)
    elif kind == "relu":
        return np.maximum(0, Z)
    elif kind == "linear":
        return Z
    else:
        raise ValueError(f"Unknown activation: {kind}. Choose from 'sigmoid', 'tanh', 'relu', 'linear'")


# ---------------------------------------------------------------------
# 2) Random initialization of input weights and biases
# ---------------------------------------------------------------------
def init_params(d: int, L: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize ELM input weights W and biases b with a normal distribution.

    Args:
      d   : int, input dimension
      L   : int, number of hidden units
      seed: optional int, RNG seed

    Returns:
      W   : np.ndarray of shape [d, L]
      b   : np.ndarray of shape [L]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize weights with small random values from normal distribution
    W = np.random.randn(d, L) * 0.1
    # Initialize biases with small random values
    b = np.random.randn(L) * 0.1
    
    return W, b


# ---------------------------------------------------------------------
# 3) Hidden response H = g(XW + b)
# ---------------------------------------------------------------------
def hidden_response(X: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str = "sigmoid") -> np.ndarray:
    """
    Compute hidden-layer response matrix H.

    Args:
      X         : np.ndarray of shape [N, d]
      W         : np.ndarray of shape [d, L]
      b         : np.ndarray of shape [L]
      activation: str, passed to activation_fn

    Returns:
      H         : np.ndarray of shape [N, L]
    """
    # Compute linear transformation: Z = X @ W + b
    # Broadcasting automatically adds b to each row
    Z = X @ W + b
    
    # Apply activation function
    H = activation_fn(Z, activation)
    
    return H


# ---------------------------------------------------------------------
# 4) Simple one-hot encoder (for classification outputs)
# ---------------------------------------------------------------------
def to_one_hot(y: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """
    Convert integer labels to a one-hot encoded matrix Y.

    Args:
      y : np.ndarray of shape [N,]
          Integer labels (0, 1, ..., m-1)
      num_classes : int, optional
          Total number of classes m

    Returns:
      Y : np.ndarray of shape [N, m]
          One-hot encoded matrix
    """
    # Ensure y is 1D array of integers
    y = np.array(y, dtype=int).ravel()
    
    # Infer number of classes if not provided
    if num_classes is None:
        num_classes = int(np.max(y)) + 1
    
    # Use np.eye to create one-hot encoding
    Y = np.eye(num_classes)[y]
    
    return Y


# ---------------------------------------------------------------------
# 5) Moore–Penrose pseudo-inverse solver (least-squares)
# ---------------------------------------------------------------------
def pinv_solver(H: np.ndarray) -> np.ndarray:
    """
    Compute the Moore–Penrose pseudo-inverse of H.

    Args:
      H: np.ndarray of shape [N, L]

    Returns:
      H_pinv: np.ndarray of shape [L, N] (the pseudo-inverse of H)
    """
    return np.linalg.pinv(H)


# ---------------------------------------------------------------------
# 5) Ridge solver
#    Returns the matrix A such that Beta = A @ Y
# ---------------------------------------------------------------------
def ridge_solver(H: np.ndarray, lam: float = 1e-4) -> np.ndarray:
    """
    Build the ridge-regression 'solver' matrix A = (H^T H + lam I)^{-1} H^T.

    Args:
      H  : np.ndarray of shape [N, L]
      lam: float, ridge regularization coefficient

    Returns:
      A  : np.ndarray of shape [L, N]
    """
    L = H.shape[1]
    
    # Compute G = H^T @ H
    G = H.T @ H
    
    # Add regularization term: G + lambda * I
    G_reg = G + lam * np.eye(L)
    
    # Compute the inverse and multiply by H^T
    A = np.linalg.inv(G_reg) @ H.T
    
    return A


# ---------------------------------------------------------------------
# 6) Fit ELM (train output weights Beta analytically)
# ---------------------------------------------------------------------
def fit_elm(
    X: np.ndarray,
    Y: np.ndarray,
    L: int = 20,
    activation: str = "sigmoid",
    use_ridge: bool = False,
    lam: float = 1e-4,
    seed: int | None = 42,
) -> dict:
    """
    Train a minimal ELM:
      1) Initialize W, b
      2) Compute H = g(XW + b)
      3) Solve for Beta:
         - if use_ridge: Beta = (H^T H + lam I)^{-1} H^T Y
         - else        : Beta = pinv(H) @ Y

    Args:
      X         : np.ndarray of shape [N, d]
      Y         : np.ndarray of shape [N, m]
      L         : int, number of hidden units
      activation: str, activation name for hidden layer
      use_ridge : bool, if True use ridge closed-form; otherwise use pseudo-inverse
      lam       : float, ridge coefficient (used only if use_ridge=True)
      seed      : optional int, RNG seed

    Returns:
      model     : dict with keys {'W', 'b', 'beta', 'activation'}
    """
    # Get input dimension
    d = X.shape[1]
    
    # 1) Initialize parameters
    W, b = init_params(d, L, seed)
    
    # 2) Compute hidden layer response
    H = hidden_response(X, W, b, activation)
    
    # 3) Solve for output weights beta
    if use_ridge:
        # Ridge solution
        A = ridge_solver(H, lam)
        beta = A @ Y
    else:
        # Pseudo-inverse solution
        H_pinv = pinv_solver(H)
        beta = H_pinv @ Y
    
    # 4) Return model dictionary
    model = {
        'W': W,
        'b': b,
        'beta': beta,
        'activation': activation
    }
    
    return model


# ---------------------------------------------------------------------
# 7) Predict with a trained ELM model
# ---------------------------------------------------------------------
def predict_elm(X: np.ndarray, model: dict) -> np.ndarray:
    """
    Predict using a trained ELM model.

    Args:
      X    : np.ndarray of shape [N, d]
      model: dict with keys 'W', 'b', 'beta', 'activation'

    Returns:
      Y_hat: np.ndarray of shape [N, m]
    """
    # Extract parameters from model
    W = model['W']
    b = model['b']
    beta = model['beta']
    activation = model['activation']
    
    # Compute hidden layer response
    H = hidden_response(X, W, b, activation)
    
    # Compute output
    Y_hat = H @ beta
    
    return Y_hat


# =========================
# Test
# =========================
if __name__ == "__main__":
    # NOTE: uncomment these lines AFTER you implement the TODOs.

    # Tiny classification-like example (N=4, d=3, m=2)
    X = np.array([
        [0.0, 0.2, 0.1],
        [0.9, 0.7, 0.3],
        [0.1, 0.8, 0.2],
        [0.8, 0.1, 0.9],
    ])
    y = np.array([0, 1, 1, 0], dtype=int)
    Y = np.eye(2)[y]  # simplest one-hot without helper

    model = fit_elm(X, Y, L=8, activation="tanh", use_ridge=False, seed=0)
    Y_hat = predict_elm(X, model)
    y_pred = np.argmax(Y_hat, axis=1)
    print("Y_hat:\n", Y_hat)
    print("y_pred:", y_pred)

    # Tiny regression-like example (N=4, d=1, m=1)
    Xr = np.array([[0.0], [0.5], [1.0], [1.5]])
    Yr = np.array([[0.0], [0.7], [1.0], [1.3]])
    model_r = fit_elm(Xr, Yr, L=10, activation="relu", use_ridge=True, lam=1e-3, seed=1)
    Yhat_r = predict_elm(Xr, model_r)
    print("Yhat_r:", Yhat_r.ravel())
    pass


import numpy as np


# ---------------------------------------------------------------------
# 1) Activation function and derivative
# ---------------------------------------------------------------------
def activation_fn(Z: np.ndarray, kind: str = "relu") -> np.ndarray:
    """
    Compute the element-wise activation.

    Args:
      Z   : np.ndarray [N, units]
      kind: activation type

    Returns:
      A   : np.ndarray [N, units]
    """
    if kind == "relu":
        return np.maximum(0, Z)
    elif kind == "tanh":
        return np.tanh(Z)
    elif kind == "sigmoid":
        return 1 / (1 + np.exp(-Z))
    else:
        raise ValueError(f"Unknown activation: {kind}")


def activation_derivative(A: np.ndarray, kind: str = "relu") -> np.ndarray:
    """
    Compute derivative of activation with respect to pre-activation Z.

    Args:
      A   : np.ndarray [N, units]
      kind: activation type

    Returns:
      dA  : np.ndarray [N, units]
    """
    if kind == "relu":
        # Derivative of ReLU: 1 if A>0, else 0
        return (A > 0).astype(float)
    elif kind == "tanh":
        # Derivative of tanh: 1 - tanh^2(z) = 1 - A^2
        return 1 - A**2
    elif kind == "sigmoid":
        # Derivative of sigmoid: sigmoid(z) * (1 - sigmoid(z)) = A * (1 - A)
        return A * (1 - A)
    else:
        raise ValueError(f"Unknown activation: {kind}")


# ---------------------------------------------------------------------
# 2) Parameter initialization
# ---------------------------------------------------------------------
def initialize_parameters(input_dim: int, hidden_dim: int, output_dim: int, seed: int = 0) -> dict:
    """
    Initialize weights and biases for a 1-hidden-layer MLP.

    Returns:
      params: dict containing W1, b1, W2, b2
    """
    np.random.seed(seed)
    
    # Initialize weights with small random values
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros((1, hidden_dim))
    
    W2 = np.random.randn(hidden_dim, output_dim) * 0.01
    b2 = np.zeros((1, output_dim))
    
    params = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    
    return params


# ---------------------------------------------------------------------
# 3) Forward pass
# ---------------------------------------------------------------------
def forward_pass(X: np.ndarray, params: dict, activation: str = "relu") -> dict:
    """
    Compute forward pass through the network:
        Z1 = X @ W1 + b1
        A1 = activation_fn(Z1)
        Z2 = A1 @ W2 + b2
        A2 = softmax or linear (depending on output)

    Returns:
      cache: dict containing Z1, A1, Z2, A2
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    
    # Layer 1: Hidden layer
    Z1 = X @ W1 + b1
    A1 = activation_fn(Z1, activation)
    
    # Layer 2: Output layer
    Z2 = A1 @ W2 + b2
    
    # For classification: apply softmax
    # For regression: keep linear
    # We'll use softmax for multi-class classification
    exp_Z2 = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))  # numerical stability
    A2 = exp_Z2 / np.sum(exp_Z2, axis=1, keepdims=True)
    
    cache = {
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2
    }
    
    return cache


# ---------------------------------------------------------------------
# 4) Loss function
# ---------------------------------------------------------------------
def compute_loss(Y_true: np.ndarray, Y_pred: np.ndarray, task: str = "classification") -> float:
    """
    Compute loss value.

    Args:
      Y_true: [N, m]
      Y_pred: [N, m]
      task: "classification" or "regression"

    Returns:
      loss: float
    """
    N = Y_true.shape[0]
    
    if task == "classification":
        # Cross-entropy loss
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        Y_pred_clipped = np.clip(Y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(Y_true * np.log(Y_pred_clipped), axis=1))
    elif task == "regression":
        # Mean Squared Error
        loss = np.mean((Y_true - Y_pred)**2)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return loss


# ---------------------------------------------------------------------
# 5) Backward pass
# ---------------------------------------------------------------------
def backward_pass(X: np.ndarray, Y: np.ndarray, params: dict, cache: dict, activation: str = "relu", task: str = "classification") -> dict:
    """
    Compute gradients for all parameters using backpropagation.

    Returns:
      grads: dict containing dW1, db1, dW2, db2
    """
    N = X.shape[0]
    
    W1 = params['W1']
    W2 = params['W2']
    
    A1 = cache['A1']
    A2 = cache['A2']
    
    # Gradient of output layer
    if task == "classification":
        # For softmax + cross-entropy, the gradient simplifies to:
        dZ2 = A2 - Y
    else:  # regression
        # For MSE loss
        dZ2 = 2 * (A2 - Y)
    
    # Gradients for W2 and b2
    dW2 = (A1.T @ dZ2) / N
    db2 = np.sum(dZ2, axis=0, keepdims=True) / N
    
    # Backpropagate to hidden layer
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * activation_derivative(A1, activation)
    
    # Gradients for W1 and b1
    dW1 = (X.T @ dZ1) / N
    db1 = np.sum(dZ1, axis=0, keepdims=True) / N
    
    grads = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }
    
    return grads


# ---------------------------------------------------------------------
# 6) Parameter update (Gradient Descent)
# ---------------------------------------------------------------------
def update_parameters(params: dict, grads: dict, lr: float = 0.01) -> dict:
    """
    Update parameters using gradient descent.

    Returns:
      params: updated parameter dict
    """
    params['W1'] = params['W1'] - lr * grads['dW1']
    params['b1'] = params['b1'] - lr * grads['db1']
    params['W2'] = params['W2'] - lr * grads['dW2']
    params['b2'] = params['b2'] - lr * grads['db2']
    
    return params


# ---------------------------------------------------------------------
# 7) Training loop
# ---------------------------------------------------------------------
def train_mlp(
    X: np.ndarray,
    Y: np.ndarray,
    hidden_dim: int,
    activation: str = "relu",
    task: str = "classification",
    epochs: int = 1000,
    lr: float = 0.01,
    seed: int = 0
) -> dict:
    """
    Full training loop for a 1-hidden-layer MLP.

    Returns:
      params: trained model parameters
    """
    # Get dimensions
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    
    # Initialize parameters
    params = initialize_parameters(input_dim, hidden_dim, output_dim, seed)
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        cache = forward_pass(X, params, activation)
        
        # Compute loss
        loss = compute_loss(Y, cache['A2'], task)
        
        # Backward pass
        grads = backward_pass(X, Y, params, cache, activation, task)
        
        # Update parameters
        params = update_parameters(params, grads, lr)
        
        # Print loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    return params


# ---------------------------------------------------------------------
# 8) Prediction
# ---------------------------------------------------------------------
def predict_mlp(X: np.ndarray, params: dict, activation: str = "relu", task: str = "classification") -> np.ndarray:
    """
    Predict outputs from trained MLP.

    Returns:
      Y_pred: predicted labels or values
    """
    # Forward pass
    cache = forward_pass(X, params, activation)
    
    if task == "classification":
        # Return class labels (argmax of probabilities)
        Y_pred = np.argmax(cache['A2'], axis=1)
    else:  # regression
        # Return raw predictions
        Y_pred = cache['A2']
    
    return Y_pred


# =========================
# Test
# =========================
if __name__ == "__main__":
    # Students can uncomment this after implementation to check their MLP
    # Example tiny dataset (binary classification)

    X = np.array([
        [0.0, 0.2],
        [0.9, 0.7],
        [0.1, 0.8],
        [0.8, 0.1],
    ])
    y = np.array([0, 1, 1, 0])
    Y = np.eye(2)[y]  # One-hot labels

    model_params = train_mlp(
        X, Y,
        hidden_dim=6,
        activation="tanh",
        task="classification",
        epochs=1000,
        lr=0.05,
        seed=0
    )

    y_pred = predict_mlp(X, model_params, activation="tanh", task="classification")
    print("True:", y)
    print("Pred:", y_pred)
    pass


import os

# ======================================================================
# PHASE 2
# ======================================================================


os.system('pip install -q tensorflow scikit-learn tqdm matplotlib seaborn hpelm')


os.system('wget -O /content/6ps7gtp2wg-1.zip "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/6ps7gtp2wg-1.zip"')


import zipfile, os

zip_path = "/content/6ps7gtp2wg-1.zip"
data_root = "content/data"
print(os.path.exists(zip_path))
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_root)
    print("Extracted to", data_root)
else:
   print("ZIP file not found. Please upload the file and check the path:", zip_path)


import os, glob
import pandas as pd

def gather_fresh_rotten(data_root):
    paths, labels, fruits = [], [], []

    # Define the search pattern for subdirectories
    search_pattern = os.path.join(data_root, "*")
    folders = glob.glob(search_pattern)

    for folder in folders:
        folder_name = os.path.basename(folder)

        # Determine label: F_ prefix is Fresh, S_ (Stale) prefix is Rotten
        if folder_name.startswith("F_"):
            label = "fresh"
        elif folder_name.startswith("S_"):
            label = "rotten"
        else:
            continue

        # Extract fruit name (e.g., Apple from F_Apple)
        fruit_name = folder_name.split("_")[-1]

        # Gather all image formats
        img_extensions = ["*.jpg", "*.jpeg", "*.png"]
        for ext in img_extensions:
            for img_path in glob.glob(os.path.join(folder, ext)):
                paths.append(img_path)
                labels.append(label)
                fruits.append(fruit_name)

    return pd.DataFrame({'path': paths, 'label': labels, 'fruit': fruits})

# Execute the function
df = gather_fresh_rotten(data_root)
print(df.head(), "\nTotal samples:", len(df))
print(df["label"].value_counts())


import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Split df into train and val with stratified sampling
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)

# Define label mapping
label_map = {"fresh": 0, "rotten": 1}

def load_image(path, label):
    # Read and decode
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize and normalize
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0

    # Map label using the dictionary
    lbl = label_map[label.decode('utf-8')]
    return img, lbl

def load_wrapper(path, label):
    # Wrap Python logic for TensorFlow
    [img, lbl] = tf.py_function(load_image, [path, label], [tf.float32, tf.int32])
    img.set_shape((*IMG_SIZE, 3))
    lbl.set_shape([])
    return img, lbl

def make_ds(df, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values))
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.map(load_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# Build datasets
train_ds = make_ds(train_df, shuffle=True)
val_ds   = make_ds(val_df)


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, Model
from tqdm import tqdm
import numpy as np

# Create MobileNetV2 base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))

# GlobalAveragePooling2D to get feature vector
x = layers.GlobalAveragePooling2D()(base_model.output)
feature_extractor = Model(inputs=base_model.input, outputs=x)

# Freeze layers
feature_extractor.trainable = False
print("Feature dim:", feature_extractor.output_shape[-1])

def extract_features(ds):
    feats, labels = [], []
    for imgs, lbls in tqdm(ds):
        # MobileNetV2 expects input in range [-1, 1], so we multiply back and preprocess
        pre = preprocess_input(imgs * 255.0)
        f = feature_extractor.predict(pre, verbose=0)
        feats.append(f)
        labels.append(lbls.numpy())
    return np.vstack(feats), np.concatenate(labels)

# Run extraction
X_train, y_train = extract_features(train_ds)
X_val, y_val     = extract_features(val_ds)


from hpelm import ELM
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Prepare labels
ohe = OneHotEncoder(sparse_output=False)
y_train_ohe = ohe.fit_transform(y_train.reshape(-1, 1))

# Initialize ELM (Input dim: 1280, Output dim: 2)
elm_model = ELM(X_train.shape[1], y_train_ohe.shape[1])
elm_model.add_neurons(1024, "sigm")
elm_model.train(X_train, y_train_ohe, "c")

# Predictions
y_val_pred_proba = elm_model.predict(X_val)
y_val_pred = np.argmax(y_val_pred_proba, axis=1)

# Evaluate
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred, target_names=label_map.keys()))


# TODO: Experiment with different number of hidden neurons and activation functions in the ELM model

import numpy as np
from hpelm import ELM
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------
# Set hyperparameters
# TODO: Define number of hidden neurons and activation function type
hidden_neurons = 2048  # Try values like 512, 1024, 2048, or 4096
activation = "tanh"    # Options: "sigm", "tanh", "lin", "tribas", "rbf_l1"

# ---------------------------------------------------------------------
# Build ELM model
# TODO: Initialize and add hidden neurons with chosen activation
# Note: X_train.shape[1] is the input dimension (1280),
# y_train_ohe.shape[1] is the output dimension (2)
exp_elm_model = ELM(X_train.shape[1], y_train_ohe.shape[1])
exp_elm_model.add_neurons(hidden_neurons, activation)

# ---------------------------------------------------------------------
# Train ELM
# TODO: Train the model on training data
print(f"Training ELM with {hidden_neurons} neurons and '{activation}' activation...")
exp_elm_model.train(X_train, y_train_ohe, "c") # "c" specifies a classification task

# ---------------------------------------------------------------------
# Evaluate ELM
# TODO: Perform predictions on validation data
y_val_pred_proba = exp_elm_model.predict(X_val)
y_val_pred = np.argmax(y_val_pred_proba, axis=1)

# ---------------------------------------------------------------------
# TODO: Print evaluation metrics
print("-" * 30)
print(f"Results for Neurons: {hidden_neurons} | Activation: {activation}")
print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print("-" * 30)
print(classification_report(y_val, y_val_pred, target_names=["fresh", "rotten"]))


# TODO: Plot confusion matrix for ELM classification results

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# Compute the confusion matrix
# TODO: Use sklearn.metrics.confusion_matrix with true and predicted labels
# We use y_val (true labels) and y_val_pred (predicted labels from the ELM)
cm = confusion_matrix(y_val, y_val_pred)

# ---------------------------------------------------------------------
# Visualize with seaborn heatmap
# TODO: Create a matplotlib figure and plot a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_map.keys(),
            yticklabels=label_map.keys())

# ---------------------------------------------------------------------
# Labeling and title
# TODO: Set axis labels and title, then show the plot
plt.title('Confusion Matrix - ELM Fruit Classification')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


from sklearn.neural_network import MLPClassifier
import time

# MLP Setup
mlp = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=300, random_state=SEED)

# ELM Training and Timing
start_elm = time.time()
elm_model.train(X_train, y_train_ohe, "c")
elm_time = time.time() - start_elm
elm_acc = accuracy_score(y_val, np.argmax(elm_model.predict(X_val), axis=1))

# MLP Training and Timing
start_mlp = time.time()
mlp.fit(X_train, y_train)
mlp_time = time.time() - start_mlp
mlp_acc = mlp.score(X_val, y_val)

# Comparison Summary
comparison_df = pd.DataFrame({
    "Model": ["ELM", "MLP"],
    "Accuracy": [elm_acc, mlp_acc],
    "Training Time (s)": [elm_time, mlp_time]
})

print(comparison_df)
