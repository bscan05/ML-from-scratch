from decision_tree import DecisionTree
import pandas as pd
import numpy as np
from decision_tree import data_prepration
from collections import Counter



class RandomForest():

    def __init__(self):
        pass

    def prep_bootstrap_data(self,data_train,sample_size,feature_count):
        
        np.random.shuffle(data_train)
        y_train = data_train[:,-1]
        x_train = data_train[:,0:-1]

        num_rows, num_cols = x_train.shape

        selected_columns = np.random.randint(low=0, high=num_cols, size=feature_count, dtype='int')

        #select some random features, they can repeat

        selected_rows = np.random.randint(low=0, high=num_rows, size=sample_size, dtype='int')
        #select some random rows they can repeat

        x_train = x_train[selected_rows][:,selected_columns]
        y_train = y_train[selected_rows]

        y_train = y_train.reshape(-1, 1)

        new_data = np.concatenate([x_train,y_train],axis=1)

        return new_data,selected_columns



    def train(self,data,number_of_trees,sample_ratio,feature_count,max_depth,jump_factor):

        sample_size = round(len(data)*sample_ratio)
        trees = []
        columns_list= []
        for i in range(number_of_trees):
            print(i+1, " tree training")

            dt = DecisionTree(max_depth)
            bootstrapped_data,selected_columns = self.prep_bootstrap_data(data,sample_size,feature_count)
            dt.root = dt.build_tree(bootstrapped_data,0,jump_factor)
            trees.append(dt)
            columns_list.append(selected_columns)

        
        
        return trees,columns_list
    
    def vote_single_row(self,trees,columns_list,x_row):

  
        mylist = []
        
        for i,tree in enumerate(trees):        
            new_x = x_row[columns_list[i]]
            mylist.append(tree.predict_single(new_x, tree.root))
        predictions = np.array(mylist)

        values, counts = np.unique(predictions, return_counts=True)

        results = np.argmax(counts)

        election_winner = values[results]
        
        return election_winner


    def test(self,X_test,Y_test,trees,columns_list):

        accuracy = 0


        for m in range(len(X_test)):

            
            if self.vote_single_row(trees,columns_list,X_test[m]) == Y_test[m]:
                print("Predicted:",self.vote_single_row(trees,columns_list,X_test[m])," Real: ", Y_test[m])
                accuracy += 1
        
        return (accuracy * 100) / (len(X_test))



data_train,x_test,y_test = data_prepration("Spotify_Features.csv",test_size=0.1)


random_forest = RandomForest()

trees,columns_list = random_forest.train(data_train,100,0.05,10,8,500)

accuracy = random_forest.test(x_test,y_test,trees,columns_list)

print("Accuracy", accuracy, "%")
        
"""
parameters = {"test_size":[0.1,0.2,0.3],"number_of_trees":[100,150,200],"sample_ratio":[0.1,0.05,0.01],"feature_count":[7,8,9,10],"max_depth":[6,7,8],"jump_factor":[500,1000,2000]}

def cross_validation(parameters,data,k):

    # for each parameter combination make cross validation
    # take one validation set out each time, calculate avarage for each combination
    #take most 

    #create batches
    batches = []
    split_size = round(len(data) / k)
    for i in range(k-1):
        array = data[i*split_size:(i+1)*split_size]
        batches.append(array)
    array = data[(k-1)*split_size:]
    batches.append(array)

    #for each batches try each possible scenerio, and take avarage
    for batch in batches:



    pass
"""
