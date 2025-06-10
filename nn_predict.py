
import numpy as np
import json

# === Activation functions ===
def relu(x):
    return np.maximum(0,x)

def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
    elif x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        raise ValueError("Softmax input must be 1D or 2D")

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# Inference using architecture and weights
def nn_forward_h5(model_arch, weights, data):
    x = data.astype(np.float64)
    for layer in model_arch:
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            activation = cfg.get("activation")
            if activation == "relu":
                x = relu(x)
            elif activation == "softmax":
                x = softmax(x)

    return x

# Entry point for testing
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)
