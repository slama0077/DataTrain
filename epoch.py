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
    features_data = createEpoch(features_data, 250, 64)
    return features_data


def filterProjection(lda_projections, y_train):

    '''This function takes lda projections and corresponding
    y_values and filters out the projections that is greater than
    -0.5 and lower than 0.5
    
    returns => lda_projections (//this is a column vector)
            => y_values (//this is a row vector)
    '''

    lda_projections = np.c_[lda_projections, y_train]

    rows = lda_projections.shape[0]

    delete_index = []

    for i in range(rows):
        if lda_projections[i][0] > -1.3 and lda_projections[i][0] < 1.0:
            delete_index.append(i)
            
    lda_projections = np.delete(lda_projections, delete_index, axis=0)

    lda_transform = np.c_[lda_projections[:, 0]] #converts the lda_transform from 1 dimension to 2 dimensiom

    return lda_transform, lda_projections[:, 1]





start = perf_counter()     #just to measure the runtime
raw_data_hand = ld.loadData()   
raw_data_feet = ld.loadData()    #loading eeg data of any two movements of ur choosing

info = createChannelInfo(22)     #we will need the info object to convert the eeg data from np data to raw_eeg data which we will need for bandpass filtering

features_hand = bandPassFilterReshape(raw_data_hand, info)   #this converts the eeg data into epochs, which will be needed for CSP
features_feet = bandPassFilterReshape(raw_data_feet, info)
features = np.concatenate([features_hand, features_feet])   #concatenating eeg data of two different movements


y_train = np.concatenate([np.zeros(features_hand.shape[0]), np.ones(features_feet.shape[0])])   #creating a target object (0 rrepresenting the first movement and 1 representing the second movement)

csp = mne.decoding.CSP(n_components = 21, reg = None, log = True, norm_trace = False)   #creating a csp object with no. of features = 4 (can be changed)
features_transform = csp.fit_transform(features, y_train)          #applying CSP to the features


lda = LinearDiscriminantAnalysis()    #creating LDA classifier (we can specify the dimension, but if only two movements are classified, it is going to be automatically 1 dimension)
lda_transform = lda.fit_transform(features_transform, y_train)  #applying LDA classifier to the data transformed by CSP (u can also just do fit but with fit_transform we can also generate plots)

lda_transform, y_train = filterProjection(lda_transform, y_train)

LR = LogisticRegression(multi_class= "multinomial")  #applying Logistic Regression to LDA transformed data (can use ovr too)
LR.fit(lda_transform, y_train)  #fitting LR

time = perf_counter() - start

print(f"The time required was {time}")

test_data_hand = ld.loadData()   #loading test data for the first movement
features_test_hand = lda.transform(csp.transform(bandPassFilterReshape(test_data_hand, info))) #transforming data
y_hand = np.zeros(features_test_hand.shape[0])
features_test_hand, y_hand = filterProjection(features_test_hand, y_hand)
test_data_feet = ld.loadData()
features_test_feet = lda.transform(csp.transform(bandPassFilterReshape(test_data_feet, info)))  #transforming data
y_feet = np.ones(features_test_feet.shape[0])
features_test_feet, y_feet = filterProjection(features_test_feet, y_feet)



test_predict_handd = LR.predict(features_test_hand)   #predicting hand movement in this case (but any other movements can be used instead)
test_predict_feett = LR.predict(features_test_feet)   #predicting feet movement in this case
test_predict = np.concatenate([test_predict_handd, test_predict_feett])   #concatenating the predicted values
y_actual = np.concatenate([y_hand, y_feet])  #creating actual test_values
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


 
# pca = PCA(n_components=2)   #plotting pca for 2d visualization
# pca_transform = pca.fit_transform(features_transform)
# plt.scatter(pca_transform[:,0], pca_transform[:, 1], c = y_train, cmap = 'rainbow_r')
# plt.show()