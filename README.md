Code for Seizure Prediction project included in this repo

Files:

Baseline Model - Contains 2 layer ANN model, training and testing script also included

LSTM Model - Contains LSTM model and train/test script

Conv Model - Convolutional NN that takes in FFT transformed data from all 19 channels

Conv Model Single - Takes in single channel data using the same convolutional network

Testing.py - Converts time series data to fourier transformed data

Siezure_CNN - Jupyter notebook that contains a more descriptive version of prior models

X_Test/train_power - train/test values of input data. Uses the power spectrum of the raw data (FFT transformed data)

Y_test/train_power - test/train labels. Take values 0, 1, 2, 3
