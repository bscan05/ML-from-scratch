import numpy as np
import pandas as pd
from tqdm import tqdm
from data_prep import data_prepration
import timeit
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


class NeuralNetwork():

    def __init__(self,n_input_neurons,n_1st_hidden,n_2nd_hidden,n_output_neurons,nonlinearity):
        self.n_input_neurons = n_input_neurons
        self.n_1st_hidden=n_1st_hidden
        self.n_2nd_hidden = n_2nd_hidden
        self.n_output_neurons = n_output_neurons
        self.nonlinearity = nonlinearity

        xavier_treshold = np.sqrt(6 / (n_input_neurons + n_output_neurons))

        w1 = np.random.uniform(-xavier_treshold, xavier_treshold, (n_input_neurons,n_1st_hidden))
        w2 = np.random.uniform(-xavier_treshold, xavier_treshold, (n_1st_hidden,n_2nd_hidden))
        w3 = np.random.uniform(-xavier_treshold, xavier_treshold, (n_2nd_hidden,n_output_neurons))

        self.weights = {"W1": w1,"W2": w2,"W3": w3}

        self.input = np.zeros(n_input_neurons)

        self.output = np.array(n_output_neurons)

        b0 = np.random.rand(n_1st_hidden)
        b1 = np.random.rand(n_2nd_hidden)
        b2 = np.random.rand(n_output_neurons)

        self.biases = {"B0" : b0,"B1" : b1,"B2" : b2}


        gr1 = np.zeros((n_input_neurons, n_1st_hidden))
        gr2 = np.zeros((n_1st_hidden, n_2nd_hidden))
        gr3 = np.zeros((n_2nd_hidden, n_output_neurons))

        self.gradients = {"Gr1": gr1,"Gr2": gr2,"Gr3": gr3}


        bias_gr0 = np.zeros(n_1st_hidden)
        bias_gr1 = np.zeros(n_2nd_hidden)
        bias_gr2 = np.zeros(n_output_neurons)

        self.bias_gradients = {"BiasGr0": bias_gr0,"BiasGr1": bias_gr1,"BiasGr2": bias_gr2}


        self.scores = np.array([np.array(n_1st_hidden), np.array(n_2nd_hidden),np.array(n_output_neurons)], dtype=object)
        self.hidden_outputs = np.array([np.array(n_1st_hidden), np.array(n_2nd_hidden)], dtype=object)

        self.deltas = np.array([np.array(n_1st_hidden), np.array(n_2nd_hidden),np.array(n_output_neurons)], dtype=object)




    def non_linearity(self,type,x):

        if type == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif type == "relu":
            return np.maximum(0, x)
        elif type == "tanh":
            return np.tanh(x)
    

    def deriv_nonlinearity(self,type,x):

        if type == "sigmoid":
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        elif type == "relu":
            if x>0:
                return 1
            else:
                return 0
        elif type == "tanh":
            return 1 - np.tanh(x)**2


    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
        



    def forward_propagate(self,xbatch,scores,hidden_outputs,output):

        type_of_non = self.nonlinearity

        scores[0] = np.dot(xbatch,self.weights["W1"]) + self.biases["B0"]
        hidden_outputs[0] = self.non_linearity(type_of_non,scores[0])

        scores[1] = np.dot(hidden_outputs[0],self.weights["W2"]) + self.biases["B1"]
        hidden_outputs[1] = self.non_linearity(type_of_non,scores[1])

        scores[2] = np.dot(hidden_outputs[1],self.weights["W3"]) + self.biases["B2"]
        
        #for output layer we need softmax
        output = self.softmax(scores[2].T).T

        return output


    def log_loss(self,y_pred,y_real):
        loss = -np.sum(y_real * np.log(y_pred))
        return loss



    def calc_accuracy(self,y_pred, y_true):
        
        y_pred_class = np.argmax(y_pred)
        y_true_class = np.argmax(y_true)
        
        
        correct_prediction = (y_pred_class == y_true_class)
        
        return correct_prediction


    def back_prop(self,y_real,y_pred):

        type_of_non = self.nonlinearity
        self.deltas[2] = y_pred - y_real

        self.deltas[1] = np.dot(self.deltas[2],self.weights["W3"].T) * self.deriv_nonlinearity(type_of_non, self.scores[1])
        self.deltas[0] = np.dot(self.deltas[1],self.weights["W2"].T) * self.deriv_nonlinearity(type_of_non, self.scores[0])
        




    def train(self,learning_rate,mini_batch_size,epoch,X_Train,Y_Train):

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


                result = self.forward_propagate(xbatch,self.scores,self.hidden_outputs,self.output)
                self.back_prop(ybatch,result)

                self.gradients["Gr3"] += np.dot(self.hidden_outputs[1].T,self.deltas[2]) / mini_batch_size
                self.gradients["Gr2"] += np.dot(self.hidden_outputs[0].T,self.deltas[1]) / mini_batch_size
                self.gradients["Gr1"] += np.dot(xbatch.T,self.deltas[0]) / mini_batch_size

                self.bias_gradients["BiasGr2"] = np.mean(self.deltas[2],axis=0)
                self.bias_gradients["BiasGr1"] = np.mean(self.deltas[1],axis=0)
                self.bias_gradients["BiasGr0"] = np.mean(self.deltas[0],axis=0)



                self.weights["W3"] -= self.gradients["Gr3"] * (learning_rate )
                self.weights["W2"] -= self.gradients["Gr2"] * (learning_rate )
                self.weights["W1"] -= self.gradients["Gr1"] * (learning_rate )

                self.biases["B2"] -= self.bias_gradients["BiasGr2"] * (learning_rate )
                self.biases["B1"] -= self.bias_gradients["BiasGr1"] * (learning_rate )
                self.biases["B0"] -= self.bias_gradients["BiasGr0"] * (learning_rate )

                self.gradients["Gr1"] = np.zeros((n_input_neurons,n_1st_hidden))
                self.gradients["Gr2"] = np.zeros((n_1st_hidden,n_2nd_hidden))
                self.gradients["Gr3"] = np.zeros((n_2nd_hidden,n_output_neurons))


                self.bias_gradients["BiasGr0"] = np.zeros(n_1st_hidden)
                self.bias_gradients["BiasGr1"] = np.zeros(n_2nd_hidden)
                self.bias_gradients["BiasGr2"] = np.zeros(n_output_neurons)

                y_pred_classes = np.argmax(result, axis=1)
                y_true_classes = np.argmax(ybatch, axis=1)
                correct += np.sum(y_pred_classes == y_true_classes)

            print(f"Epoch: {i + 1}/{epoch}, Accuracy: {(correct*100) / (mini_batch_size):.4f}%")


    def test(self,x_test,y_test):
        accuracy = 0
        x_test = x_test.values
        y_test = y_test.values

        result = self.forward_propagate(x_test,self.scores, self.hidden_outputs, self.output)
        y_pred_classes = np.argmax(result, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        accuracy += np.sum(y_pred_classes == y_true_classes)
        
        print("Test Accuracy: ",(accuracy*100) / len(x_test),"%")


n_input_neurons = 25
n_1st_hidden = 40
n_2nd_hidden = 40
n_output_neurons = 27         

X_Train,x_test,Y_Train,y_test = data_prepration("Spotify_Features.csv",test_size=0.2)

learning_rate = 0.01
type_of_non = "tanh"
mini_batch_size = 16384
epoch = 100

start = timeit.default_timer()

nn = NeuralNetwork(n_input_neurons,n_1st_hidden,n_2nd_hidden,n_output_neurons,type_of_non)



nn.train(learning_rate,mini_batch_size,epoch,X_Train,Y_Train)

nn.test(x_test,y_test)

stop = timeit.default_timer()
print('Time: ', stop - start)