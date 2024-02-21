import numpy as np
import LoadData as ld
import Welchh
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import mne
from sklearn.metrics import confusion_matrix

def extractFeatures(rawData, sf, windowSize = 750, stepSize = 250):
    # deltapower = np.zeros(rawData.shape[1])
    # alphapower = np.zeros(rawData.shape[1])
    # betapower = np.zeros(rawData.shape[1])
    x_train = []
    i = 0
    while i < (rawData.shape[0] - windowSize):
        preList = []
        for j in range(rawData.shape[1]):
            myList = []
            for k in range(stepSize):
                myList.append(rawData[k+i][j])
            myList = np.array(myList)
            beta_alpha_power = Welchh.bandpower_multitaper(myList, sf, [7, 13])
            preList.append(beta_alpha_power)

            # deltapower[j] = Welchh.bandpower_multitaper(myList, sf, [0.5,4])
            # alphapower[j] = Welchh.bandpower_multitaper(myList, sf, [4,8])
            # betapower[j] = Welchh.bandpower_multitaper(myList, sf, [8, 12] )
            # if j == rawData.shape[1] - 1:
            #     x_train.append([np.mean(deltapower), np.mean(alphapower), np.mean(betapower)])
        x_train.append(preList)
        i = i + stepSize
    return np.array(x_train)



rawData = ld.loadData()
rawData2 = ld.loadData()
sf = 250
windowSize = 750
stepSize = 250  
features_hand = extractFeatures(rawData, sf, windowSize, stepSize)
features_hand = features_hand[0:51000]
features_feet = extractFeatures(rawData2, sf, windowSize, stepSize)

n_samples_hand, n_channels_hand = features_hand.shape
n_time_points = 3
n_trial_hand = int(n_samples_hand/n_time_points)
features_hand = features_hand.reshape(n_trial_hand, n_channels_hand,n_time_points)


n_samples_feet, n_channels_feet = features_feet.shape
n_trial_feet= int(n_samples_feet/n_time_points)
features_feet = features_feet.reshape(n_trial_feet, n_channels_feet, n_time_points )
y_train = np.concatenate([np.zeros(features_hand.shape[0]), np.ones(features_feet.shape[0])])
features = np.concatenate([features_hand, features_feet])


csp = mne.decoding.CSP(n_components = 21, reg = None, log = True, norm_trace = False)
x_train_csp = csp.fit_transform(features, y_train)


lda = LinearDiscriminantAnalysis()
lda_transform = lda.fit_transform(x_train_csp, y_train)


feet_test_data = ld.loadData()
feet_test_data = extractFeatures(feet_test_data, 250)
n_samples_test, n_channels_test = feet_test_data.shape
n_trial_test= int(n_samples_test/n_time_points)
feet_test_data = feet_test_data.reshape(n_trial_test, n_channels_test, n_time_points)

feet_test_transform = csp.transform(feet_test_data)
test_predict = lda.predict(feet_test_transform)

print(test_predict)
y_actual = np.ones(test_predict.shape[0])
conf_matrix = confusion_matrix(y_actual, test_predict)
accuracy = ((conf_matrix[0][0] + conf_matrix[1][1])/(sum(conf_matrix[0]) + sum(conf_matrix[1]))) * 100

print("Accuracy: ")
print(accuracy)
plt.scatter(lda_transform, [0]*len(lda_transform), c = y_train, cmap='rainbow_r')
plt.show()

# print("Delta\t\tAlpha\t\tBeta")
# print(features)

# pca = PCA(n_components=2)
# pca_transform = pca.fit_transform(x_train_csp)
# plt.scatter(pca_transform[:,0], pca_transform[:, 1], c = y_train, cmap = 'rainbow_r')
# plt.show()
