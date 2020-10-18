# Imports
import numpy as np
import pandas as pd
import csv
import random
import keras
import sklearn
import pandas
import pickle
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, LSTM, RepeatVector, Lambda, Activation, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from keras import backend as K
from tensorflow.keras.models import Sequential
# Import from other code files
import anom_gen as anomaly
import timestamp as ts


# Support Vector Machine Class
class SVM:
    def __init__(self):
        """Initializes an SVM model"""
        self.log_data = []

    def getData(self, csv_file):
        """Get data from csv_file"""
        with open(csv_file, "r") as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                self.log_data.append(row[4:-1])
        
        logs_file = pd.read_csv(csv_file)
        self.results = logs_file["Result"]

    def getselectedpartialData(self, csv_file, anomalies=0.001, num=50000):
        """Gets the selected partial data with only a percentage of anomalies"""
        with open(csv_file, "r") as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                self.log_data.append(row[:-1])

        logs_file = pd.read_csv(csv_file)
        self.results = logs_file["Result"]

        self.anomaly_data = self.log_data[:50000]
        self.anomaly_results = self.results[:50000]
        self.anomaly_data, self.anomaly_results = sklearn.utils.shuffle(self.anomaly_data, self.anomaly_results)
        a_num = int(anomalies * num)
        self.anomaly_data = self.anomaly_data[:a_num]
        self.anomaly_results = self.anomaly_results[:a_num]

        self.log_data = self.log_data[50000:]
        self.results = self.results[50000:]
        self.log_data, self.results = sklearn.utils.shuffle(self.log_data, self.results)
        self.log_data = self.log_data[:(num-a_num)]
        self.results = self.results[:(num-a_num)]
        print(self.results)
        print(self.anomaly_results)
        self.log_data = list(self.log_data) + list(self.anomaly_data)
        self.results = list(self.results) + list(self.anomaly_results)
        return self.log_data, self.results

    def processData(self):
        """Process data and make sure everything is in the form of integers"""
        # Loops through the logs
        print("Start processing Data")
        for i in range(len(self.log_data)):
            for j in range(len(self.log_data[0])):
                # Checks to see if it is not null
                if self.log_data[i][j] != '':
                    try:
                        # Makes the number from a string to integer representation
                        self.log_data[i][j] = int(self.log_data[i][j])
                    except ValueError:
                        print(self.log_data[i][j])
                else:
                    # Converts null value to 0
                    self.log_data[i][j] = 0

    def shuffleData(self):
        """Shuffles the Data and Results in the same order"""
        print("Start shuffling Data")
        self.log_data, self.results = sklearn.utils.shuffle(self.log_data, self.results)

    def fitTransform(self):
        """Fits and then transforms the data to be represented by numbers from 0-1"""
        self.scaler = MinMaxScaler()
        self.scaled_logs = self.scaler.fit_transform(self.log_data)
    
    def partialData(self, n):
        """Select how many logs you want to train and test the model with"""
        print("Getting partial data")
        self.scaled_logs = self.scaled_logs[:n]
        self.results = self.results[:n]

    def crossValidationModel(self, k="rbf", C=8, cv=10):
        """Runs a cross validation model"""
        print("Running model")
        scoring = {'acc': 'accuracy',
                'prec': 'precision',
                'rec': 'recall'}
        self.classification_model = svm.SVC(kernel=k, C=C)
        scores = cross_validate(self.classification_model, self.scaled_logs, self.results, cv=cv, scoring=scoring)
        return scores

    def traintestSplit(self, testsize=0.2):
        """Prepares training and testing data for ML model"""
        X = self.scaled_logs
        Y = self.results

        self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(X,Y,test_size=testsize)

    def model(self, k="rbf", C_val=6):
        """Create the SVM model"""
        print("Start model")
        self.classification_model = svm.SVC(kernel=k, C=C_val)
        self.classification_model.fit(self.x_train, self.y_train)

    def saveModel(self, path_to_file="model.pkl"):
        """Save the current model"""
        self.model_file = path_to_file
        with open(model_file, 'wb') as file:
            pickle.dump(self.classification_model, file)

    def getModel(self, path_to_file):
        """Get the model at the given path"""
        with open(path_to_file, 'rb') as file:
            pickle_model = pickle.load(file)
        return pickle_model

    def testModel(self):
        """Test the model with the testing data"""
        print("Start testing model")
        self.y_prediction = self.classification_model.predict(self.x_test)

        self.classification_model_accuracy = metrics.accuracy_score(self.y_test, self.y_prediction)
        mismatch = 0
        match = 0
        for i in range(len(self.y_prediction)):
            if list(self.y_test)[i] != self.y_prediction[i]:
                #print(self.x_test[i])
                mismatch += 1
            else:
                match += 1

        print("The number of correct matches: {0} and the number of incorrect matches: {1}".format(match, mismatch))
        return self.classification_model_accuracy

    def testRows(self, n, anomaly_true=True):
        """Test specific types and number of rows"""
        if anomaly_true:
            y_predictions = []
            timestamps = anomaly.genanomalyDateTimeStamps(n)
            for timestamp in timestamps:
                row = [timestamp.year, timestamp.month, timestamp.day, timestamp.day_num, timestamp.hour, timestamp.minute, timestamp.second]
                row = [1, 18, 3, 4, 2020, 1, 1, 3, 10, 50, 30]
                print("Row: ", row)
                print()
                row = [row] + self.log_data[:-1]
                print(np.shape(row))
                row = self.scaler.fit_transform(row)
                print("Scaled row: ", row[0])
                print()
                y_predictions.append(self.classification_model.predict([row[0]])[0])
        else:
            """TO-DO: Implement testing normal rows"""
        print(y_predictions)


# Sequential AutoEncoder Class
class autoEncoder:
    models = []
    def __init__(self, name="AutoEncoder"):
        """Initializes the sequential autoencoder object."""
        # Creates the sequential model (Initializes)
        self.model = Sequential()
        # Creates an empty list of log data that will be populated from a csv file later
        self.log_data = []
        # Gives the model a unique name to differentiate between other models created
        self.name = name
        # Adds the model to the list of models of this class
        autoEncoder.models.append(self)

    def getData(self, csv_file, system=True):
        """Get log data from the gien csv file with the choice to include the system in the log data"""
        # Opens the csv file
        with open(csv_file, "r") as file:
            csv_reader = csv.reader(file)
            # Skips the header line
            next(csv_reader)
            for row in csv_reader:
                if system:
                    self.log_data.append(row[:4])
                else:
                    self.log_data.append(row[1:4])

        # Saves whether the system was included in row
        self.system = system
    
    def processData(self):
        """Convert all the numbers in the log data to integers. Will cause error if any element is not a number"""
        for i in range(len(self.log_data)):
            for j in range(len(self.log_data[0])):
                self.log_data[i][j] = int(self.log_data[i][j])
    
    def createAnomaly(self):
        """
        Creates anomalies in the testing data.
        ***THIS IS TEMPORARY UNTIL I CREATE A FUNCTION THAT WILL GENERATE ANOMALY USER/OPERATION/OBJECT TRIPLETS***
        """
        """
        if self.system:
            self.log_data[9998] = [1, 15, 15, 15]
            self.log_data[9999] = [1, 1, 3, 15]
        else:
            self.log_data[9998] = [15, 15, 15]
            self.log_data[9999] = [1, 3, 15]
        """
        self.scaled_logs = self.log_data
        print(np.shape(self.log_data))
        print(np.shape(anomalies))
        self.log_data = self.log_data + anomalies
        print(np.shape(self.log_data))
        
    def scaleData(self):
        """Scale the data so that the model reads data better and has a better accuracy score"""
        scaler = MinMaxScaler()
        self.scaled_logs = scaler.fit_transform(self.log_data)

    def traintestData(self, num=10000, percent=0.2):
        """
        Separates the train, test data based on num and percent.
        - Num: number of logs you want to partition in total
        - Percent: the percent of the total logs that will be the testing data
        """
        # Determines the number of logs for the testing data
        num_test_data = int(num * (1-percent))
        self.x_train = self.scaled_logs[0:num_test_data]
        print("Length of x_train: ", len(self.x_train))
        self.x_test = self.scaled_logs[num_test_data:num]
        print("Length of x_test: ", len(self.x_test))

    def startModel(self, hidden_layer=25, activation="relu", lstm=False):
        """
        Starts the sequential autoencoding model.
        - Takes the training data, initial hidden layer and the initial activation function.
        - Creates the first hidden layer.
        """
        if lstm:
            self.model.add(LSTM(hidden_layer, 
                            input_shape=np.shape(self.x_train)[1], 
                            activation=activation))
        else:
            self.model.add(Dense(hidden_layer, 
                                input_dim=np.shape(self.x_train)[1], 
                                activation=activation))

    def addLayer(self, hidden_layer, activation="relu", lstm=False):
        """
        Adds a layer to the model
        - Make sure to add the same layer again at the very end in reverse order to create the encoder/decoder.
        """
        if lstm:
            self.model.add(LSTM(hidden_layer, activation=activation))
        else:
            self.model.add(Dense(hidden_layer, 
                                activation=activation, use_bias=True)) 

    def compileModel(self, loss="mean_squared_error", optimizer='adam', lstm=False):
        """
        Adds the layer with the original number of inputs and compiles the model with the given loss function and optimizer.
        ***THIS SHOULD ONLY BE DONE AFTER ALL THE HIDDEN LAYERS HAVE BEEN ADDED***
        """
        #opt = keras.optimizers.Adam(learning_rate=0.05)
        if lstm:
            self.model.add(TimeDistributed(Dense((np.shape(self.x_train)[1]))))
        else:
            self.model.add(Dense(np.shape(self.x_train)[1]))
        self.model.compile(loss=loss, 
                            optimizer=optimizer,
                            )

    def fitModel(self, verbose=1, epochs=200):
        """
        Fits the model with the given verbose and epochs.
        ***THIS SHOULD ONLY BE DONE AFTER THE MODEL HAS COMPILED (SEE def compileModel(args))
        """
        self.model.fit(self.x_train, self.x_train, verbose=verbose, epochs=epochs, 
                        batch_size=80, 
                        validation_data=(self.x_test, self.x_test),
                        shuffle=True,
                        )

    def modelResults(self, anomalies, scale=100):
        """
        Tests the sequential autoencoding model on:
        1. Training data
        2. Testing data (minus the anomalies)
        3. Anomalies in the testing data
        - A smaller number means that it has encoded properly (not an anomaly)
        - A bigger number means that the autoencoder had a hard time encoding/decoding properly (an anomaly)
        """
        # Forms the predicted decoded results
        pred = self.model.predict(self.x_train)
        # Determines the score of the autoencoding process through sqrt of the MSE
        ## The number is multiplied by 100 to make the results into non only decimals

        score1 = pow(metrics.mean_absolute_error(pred, self.x_train), 2) * scale
        #score1 = metrics.mean_squared_error(pred, self.x_train) * scale
        #score1 = np.sqrt(metrics.mean_squared_error(pred, self.x_train)) * scale
        
        
        pred = self.model.predict(self.x_test)

        score2 = pow(metrics.mean_absolute_error(pred, self.x_test), 2) * scale
        #score2 = metrics.mean_squared_error(pred, self.x_test) * scale
        #score2 = np.sqrt(metrics.mean_squared_error(pred, self.x_test)) * scale
        
        print("Scaled logs: ", self.scaled_logs[50000:])
        print("On Trained Normal Data: ", score1)
        print("On Untrained Normal Data: ", score2)
        self.scaled_logs = self.scaled_logs[50000:]
        pred = self.model.predict(self.scaled_logs)

        anomaly_scores_s = pow(metrics.mean_absolute_error(pred, self.scaled_logs), 2) * scale
        #anomaly_scores_s = metrics.mean_squared_error(pred, self.scaled_logs) * scale
        #anomaly_scores_s = (np.sqrt(metrics.mean_squared_error(pred, self.scaled_logs)) * scale)
        
        print("On Untrained Anomaly Data: ", anomaly_scores_s)

        anomaly_scores = []
        for i in range(len(self.scaled_logs) - 1):
            logs = self.scaled_logs[i:i+2]
            pred = self.model.predict(logs)

            anomaly_scores.append(pow(metrics.mean_absolute_error(pred, logs), 2) * scale)
            #anomaly_scores.append(metrics.mean_squared_error(pred, logs) * scale)
            #anomaly_scores.append(np.sqrt(metrics.mean_squared_error(pred, logs)) * scale)
        
        print()
        print("On Trained Normal Data: ", score1)
        print("On Untrained Normal Data: ", score2)
        return anomaly_scores, score1, score2, anomaly_scores_s
        # Prints the results


"""
# *** THIS IS THE MODEL FOR THE TIMESTAMPS***
model = SVM()
model.getData("logs/mixed_preprocess_logs.csv")
model.processData()
model.shuffleData()
model.fitTransform()
model.partialData(10000)
model.traintestSplit()
model.model()
print(model.testModel())
model.testRows(1)
"""
'''
anomalies = []
with open("logs/both_anomaly_preprocessed_logs.csv", "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        if int(row[2]) != 1 and int(row[2]) != 2:
            anomalies.append(row[1:4])

for i in range(len(anomalies)):
    for j in range(len(anomalies[0])):
        anomalies[i][j] = int(anomalies[i][j])
list_of_above_scores = []
list_of_below_scores = []
for _ in range(6):
    autoencoder = autoEncoder()
    autoencoder.getData("logs/normal_preprocessed_logs.csv", system=False)
    autoencoder.processData()
    autoencoder.createAnomaly()
    autoencoder.scaleData()
    autoencoder.traintestData(num=50000)
    autoencoder.startModel(48)
    autoencoder.addLayer(24)
    autoencoder.addLayer(12)
    autoencoder.addLayer(6)
    autoencoder.addLayer(3, activation="linear")
    autoencoder.addLayer(6)
    autoencoder.addLayer(12)
    autoencoder.addLayer(24)
    autoencoder.addLayer(48, activation="sigmoid")
    autoencoder.compileModel(optimizer="adam", lstm=False)
    autoencoder.fitModel(verbose=1, epochs=200)
    anomaly_scores, training_score, testing_score = autoencoder.modelResults(anomalies=anomalies, scale=100)
    above_score = []
    below_score = []
    anomaly_scores = sorted(anomaly_scores)
    for i, anomaly in enumerate(anomaly_scores):
        if anomaly < testing_score:
            below_score.append([anomaly, i, anomalies[i]])
            print("Below score Anomaly #{0}: {1}".format(i, anomalies[i]))
        else:
            above_score.append([anomaly, i, anomalies[i]])
            print("Above score Anomaly #{0}: {1}".format(i, anomalies[i]))
    print(len(above_score))
    list_of_above_scores.append(len(above_score))
    print(len(below_score))
    list_of_below_scores.append(len(below_score))

with open("scores_model.csv", "w") as file:
    csv_writer = csv.writer(file)
    for i in range(len(list_of_above_scores)):
        row = [list_of_above_scores[i], list_of_below_scores[i]]
        csv_writer.writerow(row)

'''

'''
anomalies = []
with open("logs/both_anomaly_preprocessed_logs.csv", "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        if int(row[2]) != 1 and int(row[2]) != 2:
            anomalies.append(row[1:4])

for i in range(len(anomalies)):
    for j in range(len(anomalies[0])):
        anomalies[i][j] = int(anomalies[i][j])

# *** THIS IS THE MODEL FOR THE SYSTEM/USER/OPERATION/OBJECTS ***
layers = [[48, 24, 12, 6, 3, 6, 12, 24, 48]]
#
max_above_score = 0
min_below_score = 0
max_layer = []
rows = []
count = 0
for layer in layers:
    print("Count: ", count)
    for _ in range(15):
        print("Count: ", count)
        autoencoder = autoEncoder()
        autoencoder.getData("logs/normal_preprocessed_logs.csv", system=False)
        autoencoder.processData()
        autoencoder.createAnomaly()
        autoencoder.scaleData()
        autoencoder.traintestData(num=50000)
        autoencoder.startModel(hidden_layer=layer[0])
        for i in range(1, len(layer)):
            if i == len(layer) - 1:
                print(layer[i])
                autoencoder.addLayer(layer[i], lstm=False, activation="sigmoid")
            elif layer[i] == 3:
                autoencoder.addLayer(layer[i], lstm=False, activation="linear")
            else:
                autoencoder.addLayer(layer[i], lstm=False)

        autoencoder.compileModel(optimizer="adam", lstm=False)
        autoencoder.fitModel(verbose=1, epochs=200)
        anomaly_scores, training_score, testing_score, anomaly_score = autoencoder.modelResults(anomalies=anomalies, scale=100)
        above_score = []
        below_score = []
        anomaly_scores = sorted(anomaly_scores)
        for i, anomaly in enumerate(anomaly_scores):
            if anomaly < testing_score:
                below_score.append([anomaly, i, anomalies[i]])
                print("Below score Anomaly #{0}: {1}".format(i, anomalies[i]))
            else:
                above_score.append([anomaly, i, anomalies[i]])
                print("Above score Anomaly #{0}: {1}".format(i, anomalies[i]))
        rows.append([layer, training_score, testing_score, anomaly_score, len(above_score), len(below_score)])
        """
        if len(above_score) > max_above_score:
            max_layer = layer
            max_above_score = len(above_score)
            min_below_score = len(below_score)
            print("Above number: ", max_above_score)
            print("Below number: ", min_below_score)
        """
    count += 1  


print(max_above_score)
print(min_below_score)
print(max_layer)

with open("model_data.csv","w") as file:
    csv_writer = csv.writer(file)
    for row in rows:
        file_row = row
        csv_writer.writerow(file_row)
'''