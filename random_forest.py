from decision_tree import Tree
import pandas as pd
import numpy as np
from decision_tree import data_prepration
from collections import Counter
import timeit


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

            dt = Tree(max_depth)
            bootstrapped_data,selected_columns = self.prep_bootstrap_data(data,sample_size,feature_count)
            dt.root = dt.create_tree(bootstrapped_data,jump_factor,0)
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
            
            print(m)
            
            if self.vote_single_row(trees,columns_list,X_test[m]) == Y_test[m]:
                accuracy += 1
        
        return (accuracy * 100) / (len(X_test))



data_train,x_test,y_test = data_prepration("Spotify_Features.csv",test_size=0.2)


start = timeit.default_timer()
random_forest = RandomForest()

trees,columns_list = random_forest.train(data=data_train,number_of_trees=10,sample_ratio=0.005,feature_count=13,max_depth=7,jump_factor=200)

accuracy = random_forest.test(x_test,y_test,trees,columns_list)

print("Accuracy", accuracy, "%")
        




stop = timeit.default_timer()
print('Time: ', stop - start)