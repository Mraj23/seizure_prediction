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

Originally, we planned on using a dataset hosted on a machine learning dataset repository created by the University of California Irvine. This dataset contained single channel EEG data with samples at 178 Hz, and had binary labels. When performing background research, we saw that simple models achieved very high classification accuracy on this dataset. We then decided to use more complex models such as LSTM to possibly achieve higher accuracy, but we discovered that this approach had already been taken as well. These more complex models did not improve accuracy by much, as the accuracy had already been quite high when using basic models. 

The gap we aim to fill by our project is to use more complex models that effectively deal with time series data (such as RNNs, LSTMs or Multi layered CNN’s with engineered features) to specify not only if a seizure is occuring, but the type of seizure a person may be experiencing as well. To do so we will be using a different dataset that is multi channeled, multi labeled, and high resolution. The new dataset that we will be using was published very recently (just over a year ago) and as of now we have not found any other papers that cite this dataset. 

### Data

The dataset we used was curated by the American University of Beirut (owned by Dr. Wassim Nasereddine) and contains EEG signals for 6 patients monitored at various periods over the course of a year. 19 channels of data are recorded, and the sampling rate is 500Hz. There are 35 instances of seizures of varying duration, and they are of three categories: Complex Partial Seizure, Electrographic Seizures, Video-detected Seizures with no visual change over EEG. The aim of our project is to classify a one second interval of data which is represented by a matrix of size 19x500 (19 channels and 500 data points per second) and is labeled as part of one of three seizure categories, or as normal (no seizure).

There were many advantages associated with using this particular dataset apart from its newer nature:  

- The data was preprocessed, and the test-train ratio could easily be reshaped 
- All null values had been dealt with, and data was well labeled 
- 19 channels at 500 hertz gave the data high resolution 
- Data could be transformed to a frequency domain via fast fourier transform (FFT). 
- Data was publicly available, and well documented. 


4. Methods 

The first step of the analysis was to maintain the temporal integrity of the data by keeping it in a time domain, and running it through models that may be able to capture the sequential dependence of the signaals such as LSTMs and RNNs. However, we found that it took a long time, and was performing poorly as defined by the state of the art classifiers on other datasets. Hence, we performed a fast fourier transform on our data, and ran it through a simple two-layer artificial neural network. By using a FFT and keeping signals within the 1 Hz - 70 Hz range, we reshaped the input dimension from 19x500 to 19x70. The output dimensions remained the same - 1x4. Subsequently, we augmented our ANN with two, 1-D convolutional layers, since current literature shows 1-D convolutions work well to featurize signals, including audio signals, EEG signals, PPGa signals etc. Finally, we tested each channel of data individually to see their impact on classification accuracy. We used the same CNN model created before, however, we reshaped some of the hidden layers in order to make them work correctly with the altered input dimension (input changed from 19x70 to 1x70). 

We designed our neural network models using pyTorch. Fig 2 shows the architecture of our models, procured using the torch summary package. Since this project was exploratory in nature, we tried to keep our networks as basic as possible. 


Fig 2: ANN and CNN summary
![image2](https://user-images.githubusercontent.com/12428554/225136882-5b17f103-58d1-449a-b479-38a1261effe3.png)

Our computational analysis tools included sklearn’s test-train split to creature custom split ratios, and its confusion matrix to visualize the accuracy, recall, and precision. We trained all our models on 10 epochs to generate consistent results across our neural networks, and used the accuracy metric with microsoft excel to create a bubble chart to indicate the predictive power of individual channels (more in results section). 

### Results 
Our three main results include our ANN, our CNN, and our single channel analysis. For all architectures and models, we used a learning rate of 0.001, and a batch size of 1 for the sake of consistency. Our ANN model functioned as our baseline model. It performed reasonably well. Fig 4 shows the confusion matrix received from the network. The overall accuracy was 86%. 




