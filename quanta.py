"""
    Demo script adapted from this article
    https://medium.datadriveninvestor.com/stock-price-prediction-with-quantum-
        machine-learning-in-python-54948a3da389
    to compare classical and quantum machine learning algorithms
"""
import multiprocessing
import requests
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
import  matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliFeatureMap
from qiskit.circuit import Parameter
from qiskit_machine_learning.algorithms import VQR
from qiskit_algorithms.optimizers import ADAM

API_KEY = "ebb4b855e1bdd6c131ea21dc938988eb"
BASE_URL = "financialmodelingprep.com/api/v3/historical-price-full"

"""
    The main entry point to the program. Gets data, plots it in a separate
    process, then classifies it first using a classical algorithm and then
    using a quantum algorithm
"""
def run_main(tkr):
    """
    The main entry point to the program. Gets data, plots it in a separate
    process, then classifies it first using a classical algorithm and then
    using a quantum algorithm
    """

    # First, we fetch the data using the historical Data API endpoint provided
    # by Financial Modeling Prep as follows:

    api_url = f"https://{BASE_URL}/{tkr}?apikey={API_KEY}"

    df = get_data(api_url)
    (x_train, y_train, x_test, y_test) = process_data(df)

    plotting_job = multiprocessing.Process(target=plot_data, kwargs={"df":df, "tkr":tkr})
    plotting_job.start()
    classical_classifier(x_train, y_train, x_test, y_test)
    #quantum_classifier(x_train, y_train, x_test, y_test)

def get_data(api_url):
    """
        Retrieves stock data from the URL
    """
    # Make a GET request to the API
    response = requests.get(api_url, timeout=5)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the response JSON
        data = response.json()
    else:
        print(f"Error: Unable to fetch data. Status code: {response.status_code}")

    df = pd.json_normalize(data, 'historical', ['symbol']) #convert into a datframe
    return df

def process_data(df):
    """
    From this plethora of data, we are going to use open price as our temporal 
    variable and we will work with 500 data points each representing daily open 
    prices, and our window size for prediction would be 2.
    """
    df.tail()
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
    return (x_train, y_train, x_test, y_test)

def plot_data(df, tkr):
    """
        Displas a plot of the data. Runs in a separate process
        in order not to delay the execution of the remainder of the code.
    """
    plt.style.use('ggplot')

    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Plotting the time series data
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'][0:500], df['open'][0:500], marker='o', linestyle='-')

    # Adding labels and title
    plt.xlabel('date')
    plt.ylabel('open')
    plt.title(f"Time Series Data for {tkr}")

    # Display the plot
    plt.grid(True)
    plt.show()

def quantum_classifier(x_train, y_train, x_test, y_test):
    """
        Classifies the trained data using Qiskit's VQR classifier
    """
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
    vqr = VQR(
        feature_map = feature_map,
        ansatz = ansatz,
        optimizer = optimizer,
    )
    # vqr.fit() runs indefinitely and consumes all the resources on the computer.
    vqr.fit(x_train,y_train)
    prediction = vqr.predict(x_test)
    vqr_mse = mean_squared_error(y_test, prediction)

    # Calculate root mean squared error
    vqr_rmse = np.sqrt(vqr_mse)
    print(f"Quantum root mean squared error: {vqr_rmse}")

def classical_classifier(x_train, y_train, x_test, y_test):
    """
        Classifies the data using TensorFlow's Sequential model
    """
    model = Sequential()
    model.add(Dense(64,activation = 'relu', input_shape = (2,)))
    model.add(Dense(1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    model.fit(x_train, y_train, epochs = 20, batch_size = 32, validation_data = (x_test,y_test))

    prediction = model.predict(x_test)
    flattened = prediction.flatten()
    ann_mse = mean_squared_error(y_test, flattened)
    ann_rmse = np.sqrt(ann_mse)
    print(f"Classical root mean square error: {ann_rmse}")

if __name__ == "__main__":
    ticker = input("Enter a ticker symbol: ")
    run_main(ticker)
