import numpy as np
import pandas as pd
from tqdm import tqdm

# Data Preparation Function
def data_prepration(name, test_size):
    data_file = pd.read_csv(name)
    data_file = data_file.sample(frac=1).reset_index(drop=True)

    data_file.drop(["artist_name", "track_name", "track_id"], axis=1, inplace=True)

    genre_data = data_file.pop("genre")
    data_file = pd.get_dummies(data_file, columns=["mode", "key", "time_signature"], dtype=int)
    genre_data = pd.get_dummies(genre_data, dtype=int)
    data_file = (data_file - data_file.mean()) / data_file.std()

    index = round((1 - test_size) * len(data_file))
    X_Train = data_file[:index].reset_index(drop=True)
    x_test = data_file[index:].reset_index(drop=True)

    Y_Train = genre_data[:index].reset_index(drop=True)
    y_test = genre_data[index:].reset_index(drop=True)

    return X_Train, x_test, Y_Train, y_test

# Activation Functions
def non_linearity(type, x):
    if type == "sigmoid":
        return 1 / (1 + np.exp(-x))
    elif type == "relu":
        return np.maximum(0, x)
    elif type == "tanh":
        return np.tanh(x)

def deriv_nonlinearity(type, x):
    if type == "sigmoid":
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)
    elif type == "relu":
        return np.where(x > 0, 1.0, 0.0)
    elif type == "tanh":
        return 1 - np.tanh(x) ** 2

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability improvement
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def calc_accuracy(y_pred, y_true):
    y_pred_class = np.argmax(y_pred, axis=1)
    y_true_class = np.argmax(y_true, axis=1)
    return np.sum(y_pred_class == y_true_class) / len(y_true_class)

# Forward Propagation
def forward_propagate(input, type_of_non):
    # First hidden layer
    scores_1 = np.dot(input, weights["W1"]) + biases["B0"]
    hidden_output_1 = non_linearity(type_of_non, scores_1)

    # Second hidden layer
    scores_2 = np.dot(hidden_output_1, weights["W2"]) + biases["B1"]
    hidden_output_2 = non_linearity(type_of_non, scores_2)

    # Third hidden layer
    scores_3 = np.dot(hidden_output_2, weights["W3"]) + biases["B2"]
    hidden_output_3 = non_linearity(type_of_non, scores_3)

    # Output layer
    scores_4 = np.dot(hidden_output_3, weights["W4"]) + biases["B3"]
    y_pred = softmax(scores_4)

    return scores_1, hidden_output_1, scores_2, hidden_output_2, scores_3, hidden_output_3, scores_4, y_pred

# Backpropagation
def back_prop(input, y_real, scores, hidden_outputs, y_pred, type_of_non):
    # Unpack scores and hidden outputs
    scores_1, hidden_output_1, scores_2, hidden_output_2, scores_3, hidden_output_3, scores_4 = scores
    _, _, _, _, _, _, y_pred = hidden_outputs

    # Output layer deltas
    delta_4 = y_pred - y_real

    # Third hidden layer deltas
    delta_3 = np.dot(delta_4, weights["W4"].T) * deriv_nonlinearity(type_of_non, scores_3)

    # Second hidden layer deltas
    delta_2 = np.dot(delta_3, weights["W3"].T) * deriv_nonlinearity(type_of_non, scores_2)

    # First hidden layer deltas
    delta_1 = np.dot(delta_2, weights["W2"].T) * deriv_nonlinearity(type_of_non, scores_1)

    # Gradients for weights
    gradients["Gr4"] = np.dot(hidden_output_3.T, delta_4) / input.shape[0]
    gradients["Gr3"] = np.dot(hidden_output_2.T, delta_3) / input.shape[0]
    gradients["Gr2"] = np.dot(hidden_output_1.T, delta_2) / input.shape[0]
    gradients["Gr1"] = np.dot(input.T, delta_1) / input.shape[0]

    # Gradients for biases
    bias_gradients["BiasGr3"] = np.mean(delta_4, axis=0)
    bias_gradients["BiasGr2"] = np.mean(delta_3, axis=0)
    bias_gradients["BiasGr1"] = np.mean(delta_2, axis=0)
    bias_gradients["BiasGr0"] = np.mean(delta_1, axis=0)

# Training Function
def train(learning_rate, type_of_non, mini_batch_size, epoch, X_Train, Y_Train):
    X_Train = X_Train.values
    Y_Train = Y_Train.values

    t = 0  # Timestep for Adam optimizer
    for i in range(epoch):
        total_batches = int(np.ceil(len(X_Train) / mini_batch_size))
        indices = np.random.permutation(len(X_Train))  # Shuffle data
        X_Train, Y_Train = X_Train[indices], Y_Train[indices]

        for j in tqdm(range(total_batches), desc=f"Epoch {i+1}/{epoch}", unit="batch"):
            t += 1  # Increment timestep

            # Mini-batch extraction
            start_idx = j * mini_batch_size
            end_idx = min(start_idx + mini_batch_size, len(X_Train))
            X_batch = X_Train[start_idx:end_idx]
            Y_batch = Y_Train[start_idx:end_idx]

            # Forward propagation
            scores = forward_propagate(X_batch, type_of_non)

            # Backpropagation
            back_prop(X_batch, Y_batch, scores[:-1], scores[1:], scores[-1], type_of_non)

            # Adam updates for weights and biases
            for key in weights.keys():
                # Adam updates
                m_weights[key] = beta1 * m_weights[key] + (1 - beta1) * gradients["Gr" + key[-1]]
                v_weights[key] = beta2 * v_weights[key] + (1 - beta2) * (gradients["Gr" + key[-1]] ** 2)

                m_hat = m_weights[key] / (1 - beta1 ** t)
                v_hat = v_weights[key] / (1 - beta2 ** t)

                weights[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            for key in biases.keys():
                # Adam updates
                m_biases[key] = beta1 * m_biases[key] + (1 - beta1) * bias_gradients["BiasGr" + key[-1]]
                v_biases[key] = beta2 * v_biases[key] + (1 - beta2) * (bias_gradients["BiasGr" + key[-1]] ** 2)

                m_hat = m_biases[key] / (1 - beta1 ** t)
                v_hat = v_biases[key] / (1 - beta2 ** t)

                biases[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        # Calculate epoch accuracy
        _, _, _, _, _, _, _, y_pred = forward_propagate(X_Train, type_of_non)
        epoch_accuracy = calc_accuracy(y_pred, Y_Train)
        print(f"Epoch {i+1}, Accuracy: {epoch_accuracy:.4f}")

# Test Function
def test(type_of_non, X_test, Y_test):
    X_test = X_test.values
    Y_test = Y_test.values

    _, _, _, _, _, _, _, y_pred = forward_propagate(X_test, type_of_non)
    accuracy = calc_accuracy(y_pred, Y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

# Hyperparameters and Initial Setup
n_input_neurons = 30
n_1st_hidden = 40
n_2nd_hidden = 40
n_3rd_hidden = 40
n_output_neurons = 27

# Xavier Initialization
xavier_threshold = np.sqrt(6 / (n_input_neurons + n_output_neurons))

# Initialize Weights and Biases
weights = {
    "W1": np.random.uniform(-xavier_threshold, xavier_threshold, (n_input_neurons, n_1st_hidden)),
    "W2": np.random.uniform(-xavier_threshold, xavier_threshold, (n_1st_hidden, n_2nd_hidden)),
    "W3": np.random.uniform(-xavier_threshold, xavier_threshold, (n_2nd_hidden, n_3rd_hidden)),
    "W4": np.random.uniform(-xavier_threshold, xavier_threshold, (n_3rd_hidden, n_output_neurons)),
}

biases = {
    "B0": np.zeros(n_1st_hidden),
    "B1": np.zeros(n_2nd_hidden),
    "B2": np.zeros(n_3rd_hidden),
    "B3": np.zeros(n_output_neurons),
}

# Gradients
gradients = {f"Gr{i+1}": np.zeros_like(w) for i, w in enumerate(weights.values())}
bias_gradients = {f"BiasGr{i}": np.zeros_like(b) for i, b in enumerate(biases.values())}

# Adam Parameters
m_weights = {key: np.zeros_like(w) for key, w in weights.items()}
v_weights = {key: np.zeros_like(w) for key, w in weights.items()}
m_biases = {key: np.zeros_like(b) for key, b in biases.items()}
v_biases = {key: np.zeros_like(b) for key, b in biases.items()}

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Load Data and Train the Model
X_Train, X_Test, Y_Train, Y_Test = data_prepration("Spotify_Features.csv", test_size=0.2)

learning_rate = 0.001
type_of_non = "tanh"
mini_batch_size = 128
epochs = 50

train(learning_rate, type_of_non, mini_batch_size, epochs, X_Train, Y_Train)
test(type_of_non, X_Test, Y_Test)
