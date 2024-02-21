import numpy as np
import LoadData as ld
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import mne


def createChannelInfo(no_channels):
    ch_names = []
    for i in range(no_channels):
        ch_names.append('Channel' +"-"+ str(i))
    info = mne.create_info(ch_names = ch_names, sfreq = sf, ch_types='eeg')
    return info

def bandPassFilterReshape(raw_data, info, n_time_points):
    raw_data = np.transpose(raw_data)
    features = mne.io.RawArray(raw_data, info)
    features.filter(7.0, 30.0)
    features_data = np.transpose(features.get_data())
    n_samples, n_channels = features_data.shape
    n_trial = int(n_samples/n_time_points)
    features_data = features_data.reshape(n_trial, n_channels,n_time_points)
    return features_data


raw_data_hand = ld.loadData()
raw_data_feet = ld.loadData()
raw_data_hand = raw_data_hand[0:51000]
sf = 250

info = createChannelInfo(22)

n_time_points = 750
features_hand = bandPassFilterReshape(raw_data_hand, info, n_time_points)
features_feet = bandPassFilterReshape(raw_data_feet, info, n_time_points)
features = np.concatenate([features_hand, features_feet])


y_train = np.concatenate([np.zeros(features_hand.shape[0]), np.ones(features_feet.shape[0])])

csp = mne.decoding.CSP(n_components = 21, reg = None, log = True, norm_trace = False)
features_transform = csp.fit_transform(features, y_train)


lda = LinearDiscriminantAnalysis()
lda_transform = lda.fit_transform(features_transform, y_train)

# test_data = ld.loadData()
# features_test = bandPassFilterReshape(test_data, info, n_time_points)
# features_test = csp.transform(features_test)

# test_predict = lda.predict(features_test)
# print(test_predict)
# y_actual = np.ones(test_predict.shape[0])
# conf_matrix = confusion_matrix(y_actual, test_predict)
# accuracy = ((conf_matrix[0][0] + conf_matrix[1][1])/(sum(conf_matrix[0]) + sum(conf_matrix[1]))) * 100
# print(accuracy)
plt.scatter(lda_transform, [0]*len(lda_transform), c = y_train, cmap='rainbow_r')
plt.show()



pca = PCA(n_components=2)
pca_transform = pca.fit_transform(features_transform)
plt.scatter(pca_transform[:,0], pca_transform[:, 1], c = y_train, cmap = 'rainbow_r')
plt.show()