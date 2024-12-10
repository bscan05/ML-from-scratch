import numpy as np
import pandas as pd

def data_prepration(name):

    data_file = pd.read_csv(name)

    data_file.pop("artist_name")
    data_file.pop("track_name")
    data_file.pop("track_id")


    genre_data = data_file.pop("genre")


    data_file = pd.get_dummies(data_file,columns=["mode","key","time_signature"],dtype=int)

    genre_data = pd.get_dummies(genre_data,columns=["genre"],dtype=int)

    data_file = (data_file - data_file.mean()) / data_file.std()





    return data_file,genre_data

"""
arrays will be used: 

    inputs[] -> one dimensional array with input neurons
    w1[][] -> weights between input and first hidden layer
    w2[][] -> weights between first and second hidden layer
    w3[][] -> weights between second and output layer

    b0[] -> bias for input layer
    b1[] -> bias for 1st hidden layer
    b2[] -> bias for 2nd hidden layer

    hs[][] -> scores for hidden layers
    hv[][] -> output for hidden layers

    gr1[][] -> gradients for weights
    gr2[][] -> gradients for weights
    gr2[][] -> gradients for weights

    delta[][] -> sigma values for each neuron for needed to backpropagation

    outputs[] -> array for output of output layer

"""

n_input_neurons = 30
n_1st_hidden = 20
n_2nd_hidden = 20
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

# +1s are for biases
gr1 = np.random.rand(n_input_neurons,n_1st_hidden)
gr2 = np.random.rand(n_1st_hidden,n_2nd_hidden)
gr3 = np.random.rand(n_2nd_hidden,n_output_neurons)

gradients = {"Gr1": gr1,"Gr2": gr2,"Gr3": gr3}

scores = np.array([np.array(n_1st_hidden), np.array(n_2nd_hidden),np.array(n_output_neurons)], dtype=object)
hidden_outputs = np.array([np.array(n_1st_hidden), np.array(n_2nd_hidden)], dtype=object)

deltas = np.array([np.array(n_1st_hidden), np.array(n_2nd_hidden),np.array(n_output_neurons)], dtype=object)


def non_linearity(type,x):

    if type == "sigmoid":
        return 1 / (1 + np.exp(-x))
    elif type == "relu":
        return np.maximum(0, x)
    elif type == "tanh":
        return np.tanh(x)
    

def deriv_nonlinearity(type,x):

    if type == "sigmoid":
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)
    elif type == "relu":
        return np.maximum(0, x)
    elif type == "tanh":
        return 1 - np.tanh(x)**2


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
    



def forward_propagate(input,type_of_non,scores,hidden_outputs,output):

    scores[0] = np.dot(weights["W1"].T, input) + biases["B0"]
    hidden_outputs[0] = non_linearity(type_of_non,scores[0])

    scores[1] = np.dot(weights["W2"].T, hidden_outputs[0]) + biases["B1"]
    hidden_outputs[1] = non_linearity(type_of_non,scores[1])

    scores[2] = np.dot(weights["W3"].T, hidden_outputs[1]) + biases["B2"]
    
    #for output layer we need softmax
    output = softmax(scores[2])

    return output


def log_loss(y_pred,y_real):
    loss = -np.sum(y_real * np.log(y_pred))
    return loss



def calc_accuracy(y_pred, y_true):
    
    y_pred_class = np.argmax(y_pred)
    y_true_class = np.argmax(y_true)
    
    
    correct_prediction = (y_pred_class == y_true_class)
    
    return correct_prediction


def back_prop(y_real,y_pred,type_of_non):

    deltas[2] = y_pred - y_real

    deltas[1] = np.dot(weights["W3"], deltas[2]) * deriv_nonlinearity(type_of_non, scores[1])
    deltas[0] = np.dot(weights["W2"], deltas[1]) * deriv_nonlinearity(type_of_non, scores[0])
    

def update_gradients(learning_rate,input_data):

    gradients["Gr3"] = np.outer(hidden_outputs[1],deltas[2])
    gradients["Gr2"] = np.outer(hidden_outputs[0],deltas[1])
    gradients["Gr1"] = np.outer(input_data,deltas[0])
    
    weights["W3"] =- gradients["Gr3"] * learning_rate
    weights["W2"] =- gradients["Gr2"] * learning_rate
    weights["W1"] =- gradients["Gr1"] * learning_rate




data_file , genre_data = data_prepration("Spotify_Features.csv")


row = data_file.iloc[2]
y_true = genre_data.iloc[2]


for i in range(200):

    result = forward_propagate(row,"tanh",scores,hidden_outputs,output)


    back_prop(y_true,result,"tanh")
    update_gradients(0.1,row)


print(deltas)

print(calc_accuracy(result,y_true))