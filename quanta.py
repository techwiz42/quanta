import numpy as np
import pandas as pd
import requests
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliFeatureMap
from qiskit_algorithms.optimizers import ADAM
from qiskit.circuit import Parameter
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms import VQR
# First, we fetch the data using the historical Data API endpoint provided by Financial Modeling Prep as follows:

api_url = "https://financialmodelingprep.com/api/v3/historical-price-full/AAPL?apikey=ebb4b855e1bdd6c131ea21dc938988eb"

# Make a GET request to the API
response = requests.get(api_url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
   # Parse the response JSON
   data = response.json()
else:
   print(f"Error: Unable to fetch data. Status code: {response.status_code}")

df = pd.json_normalize(data, 'historical', ['symbol']) #convert into a datframe
df.tail()

#From this plethora of data, we are going to use open price as our temporal variable and we will work with 500 data points each representing daily open prices, and our window size for prediction would be 2.

final_data = df[['open', 'date']][0:500] #forming filtered dataframe
input_sequences = []
labels = []

#Creating input and output data for time series forecasting
for i in range(len(final_data['open'])):
   if i > 1:
       labels.append(final_data['open'][i])
       input_sequences.append(final_data['open'][i-1:i+1].tolist())
      
#creating train test split
x_train = np.array(input_sequences[0:400])
x_test = np.array(input_sequences[400:])
y_train = np.array(labels[0:400])
y_test = np.array(labels[400:])

#Now plot the data

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Plotting the time series data
plt.figure(figsize=(10, 6))
plt.plot(df['date'][0:500], df['open'][0:500], marker='o', linestyle='-')

# Adding labels and title
plt.xlabel('date')
plt.ylabel('open')
plt.title('Time Series Data')

# Display the plot
plt.grid(True)
plt.show()

#Now comes the Quantum stuff
num_features =  2
feature_map = PauliFeatureMap(feature_dimension = num_features, reps = 2)
optimizer = ADAM(maxiter = 100)

def ans(n, depth):
   qc = QuantumCircuit(n)
   for j in range(depth):
       for i in range(n):
           param_name = f'theta_{j}_{i}'
           theta_param = Parameter(param_name)
           qc.rx(theta_param, i)
           qc.ry(theta_param, i)
           qc.rz(theta_param, i)
   for i in range(n):
       if i == n-1:
           qc.cx(i, 0)
       else:
           qc.cx(i, i+1)
   return qc

#Initializing the ansatz circuit
ansatz = ans(num_features, 5) #anstaz(num_qubits=num_features, reps=5)

#creating train test split
x_train = np.array(input_sequences[0:400])
x_test = np.array(input_sequences[400:])
y_train = np.array(labels[0:400])
y_test = np.array(labels[400:])

vqr = VQR(
   feature_map = feature_map,
   ansatz = ansatz,
   optimizer = optimizer,
)

vqr.fit(x_train,y_train)
vqr_mse = mean_squared_error(y_test, vqr.predict(x_test))

# Calculate root mean squared error
vqr_rmse = np.sqrt(vqr_mse)

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(x_train, y_train, epochs = 20, batch_size = 32, validation_data = (x_test,y_test))

loss = model.evaluate(x_test, y_test)
prediction = model.predict(x_test)

ann_mse = mean_squared_error(y_test, prediction.flatten())
ann_rmse = np.sqrt(ann_mse)

