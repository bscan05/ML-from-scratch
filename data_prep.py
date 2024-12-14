import pandas as pd
import numpy as np

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