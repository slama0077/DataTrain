import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as signal
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper as multitaper
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def bandpower_welch(data, sf, band, window_sec, rel = False):   #window_sec = 4 sec being Welch time window
    freqs, psd = signal.welch(data, sf, nperseg = window_sec * sf) 
    plt.plot(freqs, psd)
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1]) #band we are intersted in
    bandpower = simps(psd[idx_band], freqs[idx_band], dx = freqs[1] - freqs[0])
    if rel == True:
        total_power = simps(psd, dx = freqs[1] - freqs[0])
        rel_bandpower = (bandpower/total_power)
        return rel_bandpower
    
    return bandpower



def bandpower_multitaper(data, sf, band, rel = False):
    psd, freqs = multitaper(data, sf, adaptive=True, normalization= 'full', verbose= 0)  #y value, in this case, psd is returned first
    idx_band = np.logical_and(freqs >= band[0] , freqs <= band[1])
    #plt.fill_between(freqs, psd, where=idx_band, color='skyblue')
    bandpower = simps(psd[idx_band], freqs[idx_band], dx = freqs[1] - freqs[0])
    if rel == True:
        total_power = simps(psd, dx = freqs[1] - freqs[0])
        rel_bandpower = (bandpower/total_power)
        return rel_bandpower
    
    return bandpower



# class bandPower():
#     def __init__(self, data, sf, band, window_sec = 0):
#         self.data = data
#         self.sf = sf
#         self.band = band
#         self.window_sec = window_sec
    
#     def welch(self):
#         freqs, psd = signal.welch(self.data, self.sf, nperseg = self.window_sec * self.sf)
#         idx_band = np.logical_and(freqs >= self.band[0], freqs <= self.band[1])
#         fig, ax1 = plt.subplots(1,1)
#         ax1.plot(freqs, psd)
#         deltapower = simps(psd[idx_band], freqs[idx_band], dx = freqs[1] - freqs[0]) 
#         return deltapower
    
#     def multitaper(self):
#         psd, freqs  = multitaper(self.data, self.sf, adaptive = True, normalization = 'full', verbose = 0)
#         fig, ax1 = plt.subplots(1,1)
#         idx_band = np.logical_and(freqs >= self.band[0], freqs <= self.band[1])
#         ax1.plot(freqs, psd)
#         deltapower = simps(psd[idx_band], freqs[idx_band], dx = freqs[1] - freqs[0]) 
#         return deltapower
