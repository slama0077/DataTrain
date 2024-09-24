# DataTrain
This repo contains working with EEG data, including things like calculating bandpower, using Common Spatial Pattern, and applying classifier models like Linear Discriminant Analysis and Principal Component Analysis.

The main.py file helps to classify two EEG dataaset using CSP, LDA, and LR. You can use any two files from the storage directory that you want to classify.

The utils folder contains FFT.py file which contains functions that help us to convert EEG data from time domain to frequency domain. I've used multitaper function more than Welch. 

The yasa_plot.py file helps us to plot a topographical plot of our brain. I've used the beta bandpower associated with each channel to plot. You can change from beta to any bandpower as you desire by making a few simple changes to the code.


