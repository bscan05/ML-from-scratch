import numpy as np
import pandas as pd
from tqdm import tqdm

def data_prepration(name):
    data_file = pd.read_csv(name)
    data_file = data_file.sample(frac=1)
    data_file.pop("artist_name")
    data_file.pop("track_name")
    data_file.pop("track_id")
    genre_data = data_file.pop("genre")
    data_file = pd.get_dummies(data_file, columns=["mode", "key", "time_signature"], dtype=int)
    genre_data = pd.get_dummies(genre_data, columns=["genre"], dtype=int)
    data_file = (data_file - data_file.mean()) / data_file.std()
    return data_file, genre_data


n_input_neurons = 30
n_1st_hidden = 40
n_2nd_hidden = 40
n_output_neurons = 27

xavier_treshold = np.sqrt(6 / (n_input_neurons + n_output_neurons))

w1 = np.random.uniform(-xavier_treshold, xavier_treshold, (n_input_neurons,n_1st_hidden))
w2 = np.random.uniform(-xavier_treshold, xavier_treshold, (n_1st_hidden,n_2nd_hidden))
w3 = np.random.uniform(-xavier_treshold, xavier_treshold, (n_2nd_hidden,n_output_neurons))

weights = {"W1": w1,"W2": w2,"W3": w3}

input = np.zeros(n_input_neurons)

output = np.array(n_output_neurons)

b0 = np.random.rand(n_1st_hidden)
b1 = np.random.rand(n_2nd_hidden)
b2 = np.random.rand(n_output_neurons)

biases = {"B0" : b0,"B1" : b1,"B2" : b2}


b0_gradient = np.zeros(n_1st_hidden)
b1_gradient = np.zeros(n_2nd_hidden)
b2_gradient = np.zeros(n_output_neurons)

bias_gradients = {"B0" : b0_gradient,"B1" : b1_gradient,"B2" : b2_gradient}

# +1s are for biases
gr1 = np.random.rand(n_input_neurons,n_1st_hidden)
gr2 = np.random.rand(n_1st_hidden,n_2nd_hidden)
gr3 = np.random.rand(n_2nd_hidden,n_output_neurons)

gradients = {"Gr1": gr1,"Gr2": gr2,"Gr3": gr3}

scores = np.array([np.array(n_1st_hidden), np.array(n_2nd_hidden),np.array(n_output_neurons)], dtype=object)
hidden_outputs = np.array([np.array(n_1st_hidden), np.array(n_2nd_hidden)], dtype=object)

deltas = np.array([np.array(n_1st_hidden), np.array(n_2nd_hidden),np.array(n_output_neurons)], dtype=object)




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
        return np.where(x > 0, 1, 0)
    elif type == "tanh":
        return 1 - np.tanh(x)**2

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def forward_propagate(input, type_of_non, scores, hidden_outputs, output):
    scores[0] = np.dot(weights["W1"].T, input) + biases["B0"]
    hidden_outputs[0] = non_linearity(type_of_non, scores[0])
    scores[1] = np.dot(weights["W2"].T, hidden_outputs[0]) + biases["B1"]
    hidden_outputs[1] = non_linearity(type_of_non, scores[1])
    scores[2] = np.dot(weights["W3"].T, hidden_outputs[1]) + biases["B2"]
    output = softmax(scores[2])
    return output

def log_loss(y_pred, y_real):
    loss = -np.sum(y_real * np.log(y_pred))
    return loss

def calc_accuracy(y_pred, y_true):
    y_pred_class = np.argmax(y_pred)
    y_true_class = np.argmax(y_true)
    return y_pred_class == y_true_class

def back_prop(y_real, y_pred, type_of_non):
    deltas[2] = y_pred - y_real
    deltas[1] = np.dot(weights["W3"], deltas[2]) * deriv_nonlinearity(type_of_non, scores[1])
    deltas[0] = np.dot(weights["W2"], deltas[1]) * deriv_nonlinearity(type_of_non, scores[0])

def calc_gradients(input_data):
    pass

# Data Preparation
data_file, genre_data = data_prepration("Spotify_Features.csv")

# Training Loop
epoch = 20
learning_rate = 0.01
data_size = 512
non_linearity_type = "tanh"

for k in range(epoch):
    
    num_of_batch = 400
    accuracy = 0
    for j in tqdm(range(num_of_batch), desc="Batch", unit="iteration"):
        accuracy = 0
        for i in range(data_size):

            i = j*data_size + i

            row = data_file.iloc[i]
            y_true = genre_data.iloc[i]
            result = forward_propagate(row, non_linearity_type, scores, hidden_outputs, output)
            back_prop(y_true, result, non_linearity_type)

            gradients["Gr3"] += np.outer(hidden_outputs[1], deltas[2])
            gradients["Gr2"] += np.outer(hidden_outputs[0], deltas[1])
            gradients["Gr1"] += np.outer(row, deltas[0])

            bias_gradients["B0"] += deltas[0]
            bias_gradients["B1"] += deltas[1]
            bias_gradients["B2"] += deltas[2]

            if calc_accuracy(result, y_true):
                accuracy += 1

        # Update weights
        weights["W3"] -= gradients["Gr3"] * (learning_rate / data_size)
        weights["W2"] -= gradients["Gr2"] * (learning_rate / data_size)
        weights["W1"] -= gradients["Gr1"] * (learning_rate / data_size)

        biases["B0"] -= bias_gradients["B0"] * (learning_rate / data_size)
        biases["B1"] -= bias_gradients["B1"] * (learning_rate / data_size)
        biases["B2"] -= bias_gradients["B2"] * (learning_rate / data_size)
        
        # Clear gradients
        gradients["Gr1"] = np.zeros((n_input_neurons, n_1st_hidden))
        gradients["Gr2"] = np.zeros((n_1st_hidden, n_2nd_hidden))
        gradients["Gr3"] = np.zeros((n_2nd_hidden, n_output_neurons))


        bias_gradients["B0"] = np.zeros(n_1st_hidden)
        bias_gradients["B1"] = np.zeros(n_2nd_hidden)
        bias_gradients["B2"] = np.zeros(n_output_neurons)

    print(f"Epoch: {k + 1}/{epoch}, Accuracy: {(accuracy*100) / (data_size):.4f}%")

# Evaluation on a random subset of data
random_indices = np.random.randint(0, len(data_file), 20000)
accuracy = 0
for i in random_indices:
    row = data_file.iloc[i]
    y_true = genre_data.iloc[i]
    result = forward_propagate(row, non_linearity_type, scores, hidden_outputs, output)
    if calc_accuracy(result, y_true):
        accuracy += 1