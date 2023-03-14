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


### Methods 

The first step of the analysis was to maintain the temporal integrity of the data by keeping it in a time domain, and running it through models that may be able to capture the sequential dependence of the signaals such as LSTMs and RNNs. However, we found that it took a long time, and was performing poorly as defined by the state of the art classifiers on other datasets. Hence, we performed a fast fourier transform on our data, and ran it through a simple two-layer artificial neural network. By using a FFT and keeping signals within the 1 Hz - 70 Hz range, we reshaped the input dimension from 19x500 to 19x70. The output dimensions remained the same - 1x4. Subsequently, we augmented our ANN with two, 1-D convolutional layers, since current literature shows 1-D convolutions work well to featurize signals, including audio signals, EEG signals, PPGa signals etc. Finally, we tested each channel of data individually to see their impact on classification accuracy. We used the same CNN model created before, however, we reshaped some of the hidden layers in order to make them work correctly with the altered input dimension (input changed from 19x70 to 1x70). 

We designed our neural network models using pyTorch. Fig 2 shows the architecture of our models, procured using the torch summary package. Since this project was exploratory in nature, we tried to keep our networks as basic as possible. 

![image2](https://user-images.githubusercontent.com/12428554/225136882-5b17f103-58d1-449a-b479-38a1261effe3.png)
![image3](https://user-images.githubusercontent.com/12428554/225137674-e46437b5-67fb-424a-94d3-77f6471123fa.png)


Our computational analysis tools included sklearn’s test-train split to creature custom split ratios, and its confusion matrix to visualize the accuracy, recall, and precision. We trained all our models on 10 epochs to generate consistent results across our neural networks, and used the accuracy metric with microsoft excel to create a bubble chart to indicate the predictive power of individual channels (more in results section). 

### Results 
Our three main results include our ANN, our CNN, and our single channel analysis. For all architectures and models, we used a learning rate of 0.001, and a batch size of 1 for the sake of consistency. Our ANN model functioned as our baseline model. It performed reasonably well. Fig 4 shows the confusion matrix received from the network. The overall accuracy was 86%. After adding convolutional layers to our model, we received a far higher accuracy of around 96% (Fig 5). 
![image4](https://user-images.githubusercontent.com/12428554/225137757-9bdc09d6-b26b-4dd6-ba85-69bfe67f6463.png)
![image5](https://user-images.githubusercontent.com/12428554/225137772-05da958b-7aa1-4e89-b85e-d2ca670b89c2.png)

Finally, Fig 6 shows the positions of each of the electrodes on the brain, and Fig 7. Is a similarly laid out bubble chart of the corresponding single channels in classification of the dataset. 

![image6](https://user-images.githubusercontent.com/12428554/225137894-27d40fa3-0d70-4ce7-a89f-a9db1eaf5e3e.png)

As seen, the channels with the most predictive power are those that record data from the central gyrus, the temporal lobe and the frontal lobe. More specifically, areas A1, A2, T3, C4, T6, F8 had very high classification power. Individually, the channels had an accuracy that ranged between 70%-80%.

![image7](https://user-images.githubusercontent.com/12428554/225137964-a4a3612d-8072-43d7-9faa-61de727ad1f8.png)

Finally, for our last step, we built a model that used only the best 6 channels that had a high spatial coverage to incorporate signals from all areas of the brain. Using this model we received an accuracy of 93% (Fig 8). 

![image8](https://user-images.githubusercontent.com/12428554/225137984-9ce8007d-b6d2-442e-aa20-5f7cb0fbf088.png)

### Discussion 

#### ANN

From our baseline model, we had pretty good accuracy going forward. However, from figure 4, it can be seen that most of the misclassifications occur when classifying video detected seizures as normal EEG signals. This is expected, since video detected seizures have no visible change in the EEG signal, and to the human eye they look very similar to normal EEG signals. Hence, a standard model misclassifies it around 50% of the time as normal/no seizure.

#### CNN

Although our main motivation for selecting 1-D convolutions was their prevalence in literature for classifying similar signals, we also noticed that handcrafted features (FFT) were not sufficient for dealing with video detected seizures. CNN’s have the ability to generate latent features using filters and convolutions, so we expected it to be able to pick up finer differences in signals that allowed it to discriminate between normal EEG data and video detected seizures. We were proven correct, since accuracy specifically for video detected seizures with no visible change in EEG went from 56% to 88%. It is also interesting to notice that video detected seizures were never misclassified as normal cases anymore. This is very useful, since it is better to classify seizure instances as a different kind of seizure than no seizure for diagnostic purposes. 

#### Single Channel Analysis 

By performing a single channel analysis, we can interpret the results of our model to apply them to a clinically relevant setting. 
 
These results can give us an insight into where in the brain seizures occur. They suggest that significant differences in electrical activity occur in the central gyrus, temporal lobe, and frontal lobe during a seizure.  Furthermore, if we had to build a network with data from hardware that used fewer channels, we would now know where to place these electrodes to maximize classification accuracy of the model.

#### Shortcomings 
There were a few shortcomings with our dataset and model. The first was that we only trained our models for 10 epochs for the sake of time, and the accuracy was still increasing and not yet converging. Secondly, the dataset contained far more data points that were normal data than seizure data as there were only 35 occurences of seizures. So, we should have used some sort of oversampling methods to account for this and to prevent the model from being biased.

When creating a model that used fewer channels, we decided to use the 6 top performing channels that had a high spatial coverage. However, there may be another combination of channels that when used together provide an even greater classification accuracy. To verify this, we should do further experimentation.
Conclusion
Our best performing model was able to classify high resolution EEG data quite accurately (a maximum of 95%), and achieve a similar accuracy using lower dimensional data. For seizure classification tasks, we believe that FFT/CNN methods are more effective than time series/RNN methods, although the clinical significance of working with seizure data is more skewed toward predictive tasks that require sequential methods. 

Hence, the first area of future work is in the prediction of seizures. Knowing whether or not a seizure is occurring in real-time provides valuable information, however it is more useful to know whether one will occur in the near future in order to take preemptive action. Along with this, hardware improvements could lead to even higher resolution data with fewer channels. 
### References

Dataset: Epileptic EEG Dataset - Mendeley Data 

GitHub Repository: https://github.com/Mraj23/seizure_prediction.git 

[1] Gastaut, Henri, and Benjamin G. Zifkin. “The Risk of Automobile Accidents with Seizures Occurring While Driving: Relation to Seizure Type.” Neurology, vol. 37, no. 10, 1 Oct. 1987, pp. 1613–1613, n.neurology.org/content/37/10/1613, 10.1212/WNL.37.10.1613. Accessed 8 May 2022.

[2] Kim, Taeho, et al. “Epileptic Seizure Detection and Experimental Treatment: A Review.” Frontiers in Neurology, vol. 11, 21 July 2020, 10.3389/fneur.2020.00701.

[3] M. -P. Hosseini, H. Soltanian-Zadeh, K. Elisevich and D. Pompili, "Cloud-based deep learning of big EEG data for epileptic seizure prediction," 2016 IEEE Global Conference on Signal and Information Processing (GlobalSIP), 2016, pp. 1151-1155, doi: 10.1109/GlobalSIP.2016.7906022.

[4] Obeid, I., & Picone, J. (2016). The Temple University Hospital EEG Data corpus. Frontiers in Neuroscience, 10. https://doi.org/10.3389/fnins.2016.00196 






