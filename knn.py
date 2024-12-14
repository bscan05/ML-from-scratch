import numpy as np
import random
import time
from tqdm import tqdm  
from data_prep import data_prepration


def knn(k, X_train, x_test, Y_train,y_test):

    accuracy = 0
    start_time = time.time()
    for i in tqdm(range(len(x_test)), desc="Evaluating KNN"):

        distances = np.sqrt(np.sum((X_train - x_test[i]) ** 2, axis=1))
        closest = np.argsort(distances)[:k]

        results = Y_train[closest]
        sum_results = np.sum(results,axis=0)

        vote_winner = np.argmax(sum_results)

        real_winner = np.argmax(y_test[i])

        if(vote_winner == real_winner):
            accuracy+=1

    print("Accuracy: ", accuracy * 100 / len(x_test), "%")
    print(f"Time taken: {time.time() - start_time:.3f} seconds")





X_Train,x_test,Y_Train,y_test = data_prepration("Spotify_Features.csv",test_size=0.2)

X_Train = X_Train.values
x_test = x_test.values
Y_Train = Y_Train.values
y_test = y_test.values

knn(201,X_Train,x_test,Y_Train,y_test)


