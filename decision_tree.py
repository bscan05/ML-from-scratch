import pandas as pd
import numpy as np
from collections import Counter
import timeit

def data_prepration(name,test_size):

    data_file = pd.read_csv(name)
    data_file = data_file.sample(frac = 1)

    data_file.pop("artist_name")
    data_file.pop("track_name")
    data_file.pop("track_id")
    data_file.pop("time_signature")

    genre_data = data_file.pop("genre")

    genre_categorical = genre_data.astype('category')
    genre_encoded = genre_categorical.cat.codes + 1

    data_file = pd.get_dummies(data_file,columns=["mode","key"],dtype=int)
    data_file = (data_file - data_file.mean()) / data_file.std()

    index = round((1 - test_size) * len(data_file)) 
    X_Train = data_file[:index]
    x_test = data_file[index:]

    Y_Train = genre_encoded[:index]
    y_test = genre_encoded[index:]


    Y_Train=Y_Train.values
    y_test = y_test.values
    X_Train= X_Train.values
    x_test = x_test.values

    Y_Train = Y_Train.reshape(-1, 1)

    data_train = np.concatenate([X_Train,Y_Train],axis=1)



    return data_train,x_test,y_test




class Node():

    def __init__(self, left_child=None, right_child=None, feature_index=None, threshold=None, information_gain=None,predicted_value=None):
        self.left_child = left_child
        self.right_child = right_child
        self.feature_index = feature_index
        self.threshold = threshold
        self.information_gain = information_gain
        self.predicted_value = predicted_value
        self.root = None




class Tree():

    def __init__(self,max_depth,min_info_gain=0):
        self.max_depth = max_depth
        self.min_info_gain = min_info_gain



    def entropy(self, y_col):

        a, occurance = np.unique(y_col, return_counts=True)
        pi = occurance / len(y_col)

        return -np.sum(pi * np.log2(pi))
    
        

    def Information_Gain(self,parent,left_child,right_child):

        left_w = len(left_child) / len(parent)
        right_w = len(right_child) / len(parent)

        return self.entropy(parent) - ((self.entropy(left_child)*left_w) + (right_w * self.entropy(right_child)))



    def split_data(self,data,feature,treshold):


        left = data[data[:,feature] <= treshold]
        right = data[data[:,feature] > treshold]

        return left,right
    


    def divide_node(self,data,num_of_iter):

        index = -1
        gain = -1
        left_data = []
        right_data = []
        treshold = 0
        max_info_gain = -float("inf")
        for i in range(data.shape[1]-1):
            
            column = data[:,i]
            unq_column = np.unique(column)
            sorted_unique = np.sort(unq_column)
        
            

            if len(sorted_unique)>(3*num_of_iter):
                jump = round(len(sorted_unique) / num_of_iter)
                sampled_thresholds = sorted_unique[::jump]
            else:
                sampled_thresholds = sorted_unique

            for j in sampled_thresholds:

                left, right = self.split_data(data,i,j)
                
                if len(left) and len(right):
                    information_gain = self.Information_Gain(data[:,-1],left[:,-1],right[:,-1])
                    if information_gain > max_info_gain:

                        index = i
                        gain = information_gain
                        left_data = left
                        right_data = right
                        treshold = j  
                        max_info_gain = information_gain
        

        
        return index,gain,left_data,right_data,treshold



    def create_tree(self,data,num_of_iter,curr_depth):
        

        y_train = data[:,-1]

        index,gain,left_data,right_data,treshold = self.divide_node(data,num_of_iter)


        if curr_depth < self.max_depth:


            if gain > self.min_info_gain:
                #print("building the tree1")
                left_child = self.create_tree(left_data,num_of_iter,curr_depth+1)
                right_child = self.create_tree(right_data,num_of_iter,curr_depth+1)
                #print("building the tree2")
                
                return Node(left_child,right_child,index,treshold,gain)
            
        value_of_node = self.frequent_occuring_label(y_train)
        
        return  Node(predicted_value=value_of_node)
         
    
    def frequent_occuring_label(self,y):
        a = max(set(y), key=list(y).count)
        return a
        

    def predict_single(self,x_row,node):

        node = node
        while True:

            if node.predicted_value != None:
                return node.predicted_value
            
            threshold= node.threshold
            index_of_prmt = node.feature_index
            if x_row[index_of_prmt] <= threshold:
                node = node.left_child
            else:
                node = node.right_child


    def test_data(self,x_test,y_test,root_node):

        accuracy = 0
        for i in range(len(x_test)):

            if self.predict_single(x_test[i],root_node) == y_test[i]:
                accuracy += 1
        
        return (accuracy * 100) / (len(x_test))





"""

start = timeit.default_timer()


data_train,x_test,y_test = data_prepration("Spotify_Features.csv",test_size=0.2)



decision_tree = Tree(5)



root = decision_tree.create_tree(data_train,num_of_iter=10,curr_depth=0)

stop = timeit.default_timer()
print('Time: ', stop - start)



accuracy = decision_tree.test_data(x_test,y_test,root)

print("Accuracy = ", accuracy,"%")
"""
