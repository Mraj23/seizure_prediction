# Predicting Seizure Onset using Deep Learning 

## Code for Seizure Prediction project included in this repo

Files:

Baseline Model - Contains 2 layer ANN model, training and testing script also included

LSTM Model - Contains LSTM model and train/test script

Conv Model - Convolutional NN that takes in FFT transformed data from all 19 channels

Conv Model Single - Takes in single channel data using the same convolutional network

Testing.py - Converts time series data to fourier transformed data

Siezure_CNN - Jupyter notebook that contains a more descriptive version of prior models

X_Test/train_power - train/test values of input data. Uses the power spectrum of the raw data (FFT transformed data)

Y_test/train_power - test/train labels. Take values 0, 1, 2, 3

## Report 

### Abstract

In this paper, we apply deep learning methods on EEG data to classify three different kinds of seizures (Complex Partial Seizure, Electrographic Seizures, Video-detected Seizures with no visual change over EEG) and no seizure occurrence. We used an ANN as our baseline model, which received an accuracy of 86%, but was unable to correctly classify video detected seizures. Our improved model, a CNN, had an overall higher accuracy of 95%, and was able to correctly classify the video detected seizures. We also compared the performance of single channels and created a CNN model that was able to obtain an accuracy comparable to all 19 channels (93%) while  using only 6 channels. 

### Introduction 

Around 1.2% of the US population had active epilepsy, which can lead to sudden seizures. With advancements in medical detection technology, technologies for specific prediction of seizure onset are becoming increasingly popular for epileptic patients (for example EEG detection while operating a vehicle [1]).
Machine learning has been applied to EEG data in the past [2], especially for the development of Brain Computer Interfaces[3]. There exists some research on seizure detection as well, which we learned during background research.

EEG recording hardware uses multiple electrodes (between 1 to 20) placed on a person's head (Fig 1) to record electrical activity between neurons in various regions of the brain. When a seizure is occuring in an individual, the electrical activity of the brain is noticeably different. 

Most prior work in seizure classification uses the CHB-MIT database, a database curated by MIT and the Childrens Hospital of Boston. This dataset is very old and only monitors one type of seizure. As part of our project, we decided to fill gaps in research by experimenting on newer datasets that contained seizures labeled as part of one of four classes, and experimenting with single channel data.

![image1](https://user-images.githubusercontent.com/12428554/225136430-081e4cbb-54a7-49f2-a7de-9d333c1f301c.png)

### Prior Work 


