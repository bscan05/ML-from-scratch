import pandas as pd
import numpy as np
from collections import Counter
import timeit
from data_prep import data_prepration



class Node():

    def __init__(self, left_child=None, right_child=None, feature_index=None, threshold=None, information_gain=None, predicted_value=None):
        self.left_child = left_child
        self.right_child = right_child
        self.feature_index = feature_index
        self.threshold = threshold
        self.information_gain = information_gain
        self.predicted_value = predicted_value




class DecisionTree():

    def __init__(self,max_depth):
        self.max_depth = max_depth



    def entropy(self, y_col):

        a, occurance = np.unique(y_col, return_counts=True)
        pi = occurance / len(y_col)

        return -np.sum(pi * np.log2(pi))
    
        

    def Information_Gain(self,parent,left_child,right_child):

        parent_entropy = self.entropy(parent)
        left_entropy = self.entropy(left_child)
        right_entropy = self.entropy(right_child)

        left_weight = len(left_child) / len(parent)
        right_weight = len(right_child) / len(parent)

        return parent_entropy - ((left_entropy*left_weight) + (right_weight * right_entropy))



    def split(self,data,feature,treshold):


        left = data[data[:,feature] <= treshold]
        right = data[data[:,feature] > treshold]

        return left,right
    


    def best_split(self,data):

        best_split = {"index":-1,"info_gain":-1,"left_data":[],"right_data":[],"treshold":0}
        index = -1
        gain = -1
        left_data = []
        right_data = []
        treshold = 0
        max_info_gain = -float("inf")
        for i in range(data.shape[1]-1):
            
            column = data[:,i]
            unq_column = np.unique(column)
            sorted_unique = np.sort(unq_column)  # Sort the unique values
        
            

            if len(sorted_unique)>300:
                jump = round(len(sorted_unique) / 100)
                sampled_thresholds = sorted_unique[::jump]
            else:
                sampled_thresholds = sorted_unique

            for j in sampled_thresholds:

                left, right = self.split(data,i,j)
                
                if len(left) and len(right):
                    information_gain = self.Information_Gain(data[:,-1],left[:,-1],right[:,-1])
                    if information_gain > max_info_gain:

                        index = i
                        gain = information_gain
                        left_data = left
                        right_data = right
                        treshold = j  
                        max_info_gain = information_gain
        
        best_split["index"] = index
        best_split["info_gain"] = gain
        best_split["left_data"] = left_data
        best_split["right_data"] = right_data
        best_split["treshold"] = treshold
        
        return best_split



    def build_tree(self,data,curr_depth):
        

        y_train = data[:,-1]

        best_split = self.best_split(data)

        if curr_depth < self.max_depth:

            if best_split["info_gain"] > 0:
                print("building the tree1")
                left_child = self.build_tree(best_split["left_data"],curr_depth+1)
                right_child = self.build_tree(best_split["right_data"],curr_depth+1)
                print("building the tree2")
                
                return Node(left_child,right_child,best_split["index"],best_split["treshold"],best_split["info_gain"])
        
        value_of_node = self.find_max_occuring_label(y_train)
        print("building the tree")
        return Node(predicted_value=value_of_node)
        
    
    def find_max_occuring_label(self,y):
        return max(set(y), key=list(y).count)
        

    def predict():
        pass
        




start = timeit.default_timer()


X_Train,x_test,Y_Train,y_test = data_prepration("Spotify_Features.csv",test_size=0.2)

dataset = np.concatenate([X_Train.values,Y_Train.values],axis=1)



decision_tree = DecisionTree(2)

root = decision_tree.build_tree(dataset,0)
print(root.left_child.predicted_value)
stop = timeit.default_timer()
print('Time: ', stop - start)  