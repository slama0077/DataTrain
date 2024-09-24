import csv
import numpy as np

#the loadData function converts a csv file containing features and output to numpy arrays. The csv file must be have columns of features
#and the last column must be output (result). The loadData function returns a numpy array of features (x_train), a numpy array of output(y_train), 
#and a list of features. The load data function is for the training data. For instance, if the csv file is:

#size, time, results
#3, 2, 0
#10, 9, 1

#x-train = [[3, 2], [10, 9]]
#y_train = [0, 1]
#features = [size, time]



#the loadInputData function is for applying the trained model to a sample data. The sample data must also be .csv file and should only contain features. 
#This function returns a numpy array of features(x_check). 





def loadData():
    while True: 
        try:    
            filename = input("Provide the name of your file. The file must be a CSV file: ")
            istream = open(filename, mode = 'r', encoding = 'utf-8')
        except: 
            print(f"There is no such file called {filename}. Or maybe you have not provided the extension. Try again!")
        else:
            print("The file name provided is correct. Thank you!")
            break
    
    csv_input = csv.reader(istream)
    data = list(csv_input)
    feature_length = len(data[0])
    i = 0
    x_train = [[]]
    while i < len(data):
        x = []
        j = 0
        while j < feature_length:
            x.append(float(data[i][j]))
            j += 1
        x_train.append(x)
        i += 1
    x_train = x_train[1:]
    istream.close()
    x_train = np.array(x_train)
    return (x_train)

def loadInputData():
    while True: 
        try:    
            filename = input("Provide the name of your file that you want to apply this model to. The file must be a CSV file: ")
            istream = open(filename, mode = 'r', encoding = 'utf-8')
        except: 
            print(f"There is no such file called {filename}. Or maybe you have not provided the extension. Try again!")
        else:
            print("The file name provided is correct. Thank you!")
            break
    csv_input = csv.reader(istream)
    data = list(csv_input)
    feature_length = len(data[0])
    i = 1
    x_train = [[]]
    while i < len(data):
        x = []
        j = 0
        while j < feature_length:
            x.append(float(data[i][j]))
            j += 1
        x_train.append(x)
        i += 1
    x_train = x_train[1:]
    istream.close()
    x_train = np.array(x_train)
    return x_train