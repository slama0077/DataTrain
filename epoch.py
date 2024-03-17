import numpy as np
import LoadData as ld
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import mne
from time import perf_counter


def createEpoch(feature_data, epoch_size, epoch_window):
    '''
    '''
    epochs = []
    n_samples = feature_data.shape[1]
    for i in range(0, n_samples, epoch_window):
        if n_samples - i < epoch_size:
            break

        epoch_data = feature_data[:, i: i + epoch_size]
        epochs.append(epoch_data)
    
    
    epochs_array = np.array(epochs)

    return epochs_array





def createChannelInfo(no_channels):
    '''This function creates an info object 
    that is required later to convert the eeg data from np data
    to mne.io.raw data.

    no_channels: number of channels used in the eeg data

    returns info object
    '''
    ch_names = []
    for i in range(no_channels):
        ch_names.append('Channel' +"-"+ str(i))   #for this code I have just created dummy channel names. We can have actual channel names
    info = mne.create_info(ch_names = ch_names, sfreq = 250, ch_types='eeg') 
    return info 

def bandPassFilterReshape(raw_data, info):
    '''
    First converts the np data into mne.io.raw data and
    uses createEpoch funcn to convert the data into epochs
    raw_data : eeg data
    info: info object we created before which has information about the channel
    
    returns epoch data which will be applied for CSP
    '''
    raw_data = np.transpose(raw_data)
    features = mne.io.RawArray(raw_data, info)
    features.filter(7.0, 40.0)
    features_data = features.get_data()
    features_data = createEpoch(features_data, 750, 64)
    return features_data


start = perf_counter()     #just to measure the runtime
raw_data_hand = ld.loadData()   
raw_data_feet = ld.loadData()    #loading eeg data of any two movements of ur choosing

info = createChannelInfo(22)     #we will need the info object to convert the eeg data from np data to raw_eeg data which we will need for bandpass filtering

features_hand = bandPassFilterReshape(raw_data_hand, info)   #this converts the eeg data into epochs, which will be needed for CSP
features_feet = bandPassFilterReshape(raw_data_feet, info)
features = np.concatenate([features_hand, features_feet])   #concatenating eeg data of two different movements


y_train = np.concatenate([np.zeros(features_hand.shape[0]), np.ones(features_feet.shape[0])])   #creating a target object (0 rrepresenting the first movement and 1 representing the second movement)

csp = mne.decoding.CSP(n_components = 4, reg = None, log = True, norm_trace = False)   #creating a csp object with no. of features = 4 (can be changed)
features_transform = csp.fit_transform(features, y_train)           #applying CSP to the features


lda = LinearDiscriminantAnalysis()    #creating LDA classifier (we can specify the dimension, but if only two movements are classified, it is going to be automatically 1 dimension)
lda.fit_transform = lda.fit(features_transform, y_train)  #applying LDA classifier to the data transformed by CSP (u can also just do fit but with fit_transform we can also generate plots)
time = perf_counter() - start

print(f"The time required was {time}")

test_data_hand = ld.loadData()   #loading test data for the first movement
features_test_hand = bandPassFilterReshape(test_data_hand, info)  #applying bandpass filter and changing it into epochs
test_data_feet = ld.loadData()
features_test_feet = bandPassFilterReshape(test_data_feet, info)
features_test_hand = csp.transform(features_test_hand)      #transforming the data using CSP
features_test_feet = csp.transform(features_test_feet)


test_predict_handd = lda.predict(features_test_hand)   #predicting hand movement in this case (but any other movements can be used instead)
test_predict_feett = lda.predict(features_test_feet)   #predicting feet movement in this case
test_predict = np.concatenate([test_predict_handd, test_predict_feett])   #concatenating the predicted values
y_actual = np.concatenate([np.zeros(test_predict_handd.shape[0]), np.ones(test_predict_feett.shape[0])])  #creating actual test_values
conf_matrix = confusion_matrix(y_actual, test_predict)
print("\n\n")
print("Confusion Matrix:")
print(conf_matrix)
accuracy = ((conf_matrix[0][0] + conf_matrix[1][1])/(sum(conf_matrix[0]) + sum(conf_matrix[1]))) * 100 #accuracy 

print("\n\n")
print("The accuracy of the classifier is: ")
print(accuracy)
print("\n\n")

plt.scatter(lda_transform, [0]*len(lda_transform), c = y_train, cmap='rainbow_r')  #plotting lda
plt.show()


 
pca = PCA(n_components=2)   #plotting pca for 2d visualization
pca_transform = pca.fit_transform(features_transform)
plt.scatter(pca_transform[:,0], pca_transform[:, 1], c = y_train, cmap = 'rainbow_r')
plt.show()