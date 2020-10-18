# Imports
import numpy as np
import pandas as pd
import csv
import random
import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
import model as m

model = m.SVM()
log_data, results = model.getselectedpartialData("logs/mixed_preprocess_logs.csv", anomalies=0.001, num=10000)
model.processData()
model.shuffleData()
model.fitTransform()
#model.partialData(50000)
scores = model.crossValidationModel(k="linear", C=2, cv=10)
print(scores)
print(scores["test_acc"].mean())
print(scores["test_prec"].mean())
print(scores["test_rec"].mean())
exit()
#model.traintestSplit(testsize=0.2)
#model.model(k="rbf", C_val=20)
#print(model.testModel())
#model.testRows(1)


"""
log_data = []
with open("logs/normal_preprocessed_logs.csv", "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        log_data.append(row)

for i in range(len(log_data)):
    for j in range(len(log_data[0])):
            log_data[i][j] = int(log_data[i][j])


anomalies = []
with open("logs/both_anomaly_preprocessed_logs.csv", "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        if int(row[2]) != 1 and int(row[2]) != 2:
            anomalies.append(row)

for i in range(len(anomalies)):
    for j in range(len(anomalies[0])):
        anomalies[i][j] = int(anomalies[i][j])

log_data = log_data + anomalies

df = pd.DataFrame(log_data)

scaler = MinMaxScaler()
scaled_logs = scaler.fit_transform(log_data)
anomaly_scaled_logs = scaled_logs[50000:]
training_samples = int(len(scaled_logs) * 0.80)
test_samples = int(len(scaled_logs) * 0.20)
X_train = scaled_logs[:training_samples]
X_test = scaled_logs[training_samples: training_samples + test_samples]

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
    filepath='log_model-testing.h5',
    monitor='val_loss',
    save_best_only=True
    )
]

input_dim = 7
encoding_dim = 4
hidden_dim = 2


nb_epoch = 30
batch_size = 128
learning_rate = 0.1

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
encoder = Dense(hidden_dim, activation="relu")(encoder)
decoder = Dense(encoding_dim, activation='relu')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.summary()

autoencoder.compile(optimizer='adam',
            loss='mean_squared_error',
            metrics=['accuracy'])

autoencoder.fit(X_train, X_train,
        epochs=50,
        batch_size=256,
        #callbacks=callbacks_list,
        validation_data=(X_test, X_test),
        shuffle=True
        )


normal_predicted_logs = autoencoder.predict(X_test)
anomaly_predicted_logs = autoencoder.predict(anomaly_scaled_logs)
mse_normal = np.mean(np.power(X_test - normal_predicted_logs, 2), axis=1)
mse_anomaly = np.mean(np.power(anomaly_scaled_logs - anomaly_predicted_logs, 2), axis=1)

print(np.mean(mse_normal))
print(np.mean(mse_anomaly))
below_avg = 0
above_avg = 0
for anomaly in mse_anomaly:
    if anomaly < np.mean(mse_normal):
        below_avg += 1
    else:
        above_avg += 1

print(above_avg)
print(below_avg)
print((above_avg / (above_avg + below_avg)))
exit()

print(min(mse))
print(max(mse))
print("mse of the anomaly", mse[-1])
mse = sorted(mse, reverse=True)
print("greatest mse: ", mse[0])
print("second greatest mse: ", mse[1])
print(mse[:5])

print(scaled_logs[-1])
"""