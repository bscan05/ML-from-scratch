import numpy as np
import pandas as pd
from tqdm import tqdm
from data_prep import data_prepration

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

n_input_neurons = 25
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


gr1 = np.zeros((n_input_neurons, n_1st_hidden))
gr2 = np.zeros((n_1st_hidden, n_2nd_hidden))
gr3 = np.zeros((n_2nd_hidden, n_output_neurons))

gradients = {"Gr1": gr1,"Gr2": gr2,"Gr3": gr3}


bias_gr0 = np.zeros(n_1st_hidden)
bias_gr1 = np.zeros(n_2nd_hidden)
bias_gr2 = np.zeros(n_output_neurons)

bias_gradients = {"BiasGr0": bias_gr0,"BiasGr1": bias_gr1,"BiasGr2": bias_gr2}


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
    



def forward_propagate(xbatch,type_of_non,scores,hidden_outputs,output):

    scores[0] = np.dot(xbatch,weights["W1"]) + biases["B0"]
    hidden_outputs[0] = non_linearity(type_of_non,scores[0])

    scores[1] = np.dot(hidden_outputs[0],weights["W2"]) + biases["B1"]
    hidden_outputs[1] = non_linearity(type_of_non,scores[1])

    scores[2] = np.dot(hidden_outputs[1],weights["W3"]) + biases["B2"]
    
    #for output layer we need softmax
    output = softmax(scores[2].T).T

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

    deltas[1] = np.dot(deltas[2],weights["W3"].T) * deriv_nonlinearity(type_of_non, scores[1])
    deltas[0] = np.dot(deltas[1],weights["W2"].T) * deriv_nonlinearity(type_of_non, scores[0])
    




def train(learning_rate,type_of_non,mini_batch_size,epoch,X_Train,Y_Train):

    X_Train = X_Train.values
    Y_Train = Y_Train.values

    for i in range(epoch):

        total_batches = round(len(X_Train) / mini_batch_size)-1

        
        correct = 0
        for j in tqdm(range(total_batches), desc="Batch", unit="iteration"):
            correct = 0

            k = (j * mini_batch_size)

            xbatch = X_Train[k:(k+mini_batch_size)]
            ybatch = Y_Train[k:(k+mini_batch_size)]


            result = forward_propagate(xbatch,type_of_non,scores,hidden_outputs,output)
            back_prop(ybatch,result,type_of_non)

            gradients["Gr3"] += np.dot(hidden_outputs[1].T,deltas[2]) / mini_batch_size
            gradients["Gr2"] += np.dot(hidden_outputs[0].T,deltas[1]) / mini_batch_size
            gradients["Gr1"] += np.dot(xbatch.T,deltas[0]) / mini_batch_size

            bias_gradients["BiasGr2"] = np.mean(deltas[2],axis=0)
            bias_gradients["BiasGr1"] = np.mean(deltas[1],axis=0)
            bias_gradients["BiasGr0"] = np.mean(deltas[0],axis=0)



            weights["W3"] -= gradients["Gr3"] * (learning_rate )
            weights["W2"] -= gradients["Gr2"] * (learning_rate )
            weights["W1"] -= gradients["Gr1"] * (learning_rate )

            biases["B2"] -= bias_gradients["BiasGr2"] * (learning_rate )
            biases["B1"] -= bias_gradients["BiasGr1"] * (learning_rate )
            biases["B0"] -= bias_gradients["BiasGr0"] * (learning_rate )

            gradients["Gr1"] = np.zeros((n_input_neurons,n_1st_hidden))
            gradients["Gr2"] = np.zeros((n_1st_hidden,n_2nd_hidden))
            gradients["Gr3"] = np.zeros((n_2nd_hidden,n_output_neurons))


            bias_gradients["BiasGr0"] = np.zeros(n_1st_hidden)
            bias_gradients["BiasGr1"] = np.zeros(n_2nd_hidden)
            bias_gradients["BiasGr2"] = np.zeros(n_output_neurons)

            y_pred_classes = np.argmax(result, axis=1)
            y_true_classes = np.argmax(ybatch, axis=1)
            correct += np.sum(y_pred_classes == y_true_classes)

        print(f"Epoch: {i + 1}/{epoch}, Accuracy: {(correct*100) / (mini_batch_size):.4f}%")


def test(type_of_non,x_test,y_test):

    accuracy = 0
    x_test = x_test.values
    y_test = y_test.values

    result = forward_propagate(x_test, type_of_non, scores, hidden_outputs, output)
    y_pred_classes = np.argmax(result, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    accuracy += np.sum(y_pred_classes == y_true_classes)
    
    print("Test Accuracy: ",(accuracy*100) / len(x_test),"%")
                

X_Train,x_test,Y_Train,y_test = data_prepration("Spotify_Features.csv",test_size=0.2)



learning_rate = 0.01
type_of_non = "tanh"
mini_batch_size = 1024
epoch = 200

train(learning_rate,type_of_non,mini_batch_size,epoch,X_Train,Y_Train)

test(type_of_non,x_test,y_test)






