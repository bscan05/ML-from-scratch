import numpy as np
import pandas as pd
from tqdm import tqdm

def data_prepration(name,test_size):

    data_file = pd.read_csv(name)
    data_file = data_file.sample(frac = 1)

    data_file.pop("artist_name")
    data_file.pop("track_name")
    data_file.pop("track_id")


    genre_data = data_file.pop("genre")
    data_file = pd.get_dummies(data_file,columns=["mode","key","time_signature"],dtype=int)
    genre_data = pd.get_dummies(genre_data,columns=["genre"],dtype=int)
    data_file = (data_file - data_file.mean()) / data_file.std()

    index = round((1 - test_size) * len(data_file)) 
    X_Train = data_file[:index]
    x_test = data_file[index:]

    Y_Train = genre_data[:index]
    y_test = genre_data[index:]



    return X_Train,x_test,Y_Train,y_test

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
n_1st_hidden = 30
n_2nd_hidden = 30
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
    




def train(learning_rate,type_of_non,mini_batch_size,epoch,X_Train,Y_Train):



    for i in range(epoch):

        total_batches = round(len(X_Train) / mini_batch_size)-1

        
        correct = 0
        for j in tqdm(range(total_batches), desc="Batch", unit="iteration"):
            correct = 0
            for k in range(mini_batch_size):
                k = k + (j * mini_batch_size)
                row = X_Train.iloc[k]
                y_true = Y_Train.iloc[k]
                result = forward_propagate(row,type_of_non,scores,hidden_outputs,output)
                back_prop(y_true,result,type_of_non)

                gradients["Gr3"] += np.outer(hidden_outputs[1],deltas[2])
                gradients["Gr2"] += np.outer(hidden_outputs[0],deltas[1])
                gradients["Gr1"] += np.outer(row,deltas[0])

                bias_gradients["BiasGr2"] += deltas[2]
                bias_gradients["BiasGr1"] += deltas[1]
                bias_gradients["BiasGr0"] += deltas[0]

                if(calc_accuracy(result,y_true)):
                    correct+=1

            weights["W3"] -= gradients["Gr3"] * (learning_rate / mini_batch_size)
            weights["W2"] -= gradients["Gr2"] * (learning_rate / mini_batch_size)
            weights["W1"] -= gradients["Gr1"] * (learning_rate / mini_batch_size)

            biases["B2"] -= bias_gradients["BiasGr2"] * (learning_rate / mini_batch_size)
            biases["B1"] -= bias_gradients["BiasGr1"] * (learning_rate / mini_batch_size)
            biases["B0"] -= bias_gradients["BiasGr0"] * (learning_rate / mini_batch_size)

            gradients["Gr1"] = np.zeros((n_input_neurons,n_1st_hidden))
            gradients["Gr2"] = np.zeros((n_1st_hidden,n_2nd_hidden))
            gradients["Gr3"] = np.zeros((n_2nd_hidden,n_output_neurons))


            bias_gradients["BiasGr0"] = np.zeros(n_1st_hidden)
            bias_gradients["BiasGr1"] = np.zeros(n_2nd_hidden)
            bias_gradients["BiasGr2"] = np.zeros(n_output_neurons)

        print(f"Epoch: {i + 1}/{epoch}, Accuracy: {(correct*100) / (mini_batch_size):.4f}%")


def test(type_of_non,x_test,y_test):

    accuracy = 0

    for i in range(len(x_test)):
        row = x_test.iloc[i]
        y_true = y_test.iloc[i]
        result = forward_propagate(row, type_of_non, scores, hidden_outputs, output)
        if calc_accuracy(result, y_true):
            accuracy += 1
    
    print("Accuracy: ",(accuracy*100) / len(x_test),"%")
                

X_Train,x_test,Y_Train,y_test = data_prepration("Spotify_Features.csv",test_size=0.2)



learning_rate = 0.01
type_of_non = "tanh"
mini_batch_size = 2048
epoch = 20

train(learning_rate,type_of_non,mini_batch_size,epoch,X_Train,Y_Train)

test(type_of_non,x_test,y_test)






