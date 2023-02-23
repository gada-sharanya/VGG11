input_folder =”NWPU_aerial” 
 
Splitfoplder.ratio(input_folder,output=”cell_images2”,seed=42,rat
io(.7,.2.1),group_prefix=None) 
 
# The output folder is mounted in drive and executed via Google 
colab 
 
from google.colab import drive 
drive.mount('/content/drive/') 
 
!pip install -q keras 
import keras 
from keras.applications.vgg16 import VGG16 
import tensorflow.compat.v2 as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import glob 
import cv2 
 
from keras.models import Model, Sequential 
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, In
put 
from keras.layers.normalization import batch_normalization 
import os 
import seaborn as sns 
SIZE = 224 
train_images = [] 
train_labels = [] 
for directory_path in glob.glob("/content/drive/MyDrive/Aer/train
/*"):label = directory_path.split("/")[-1] 
    print(label) 
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg
")): 
        print(img_path) 
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        img = cv2.resize(img, (SIZE, SIZE)) 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
        train_images.append(img) 
        train_labels.append(label) 
train_images = np.array(train_images) 
train_labels = np.array(train_labels) 
test_images = [] 
test_labels = [] 
for directory_path in glob.glob("/content/drive/MyDrive/Aer/val/*
"): 
    fruit_label = directory_path.split("/")[-1] 
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg
")): 
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        img = cv2.resize(img, (SIZE, SIZE)) 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
        test_images.append(img) 
        test_labels.append(fruit_label) 
#Convert lists to arrays 
test_images = np.array(test_images) 
test_labels = np.array(test_labels) 
from sklearn import preprocessing 
le = preprocessing.LabelEncoder() 
le.fit(test_labels) 
test_labels_encoded = le.transform(test_labels) 
le.fit(train_labels) 
train_labels_encoded = le.transform(train_labels) 
x_train, y_train, x_test, y_test = train_images, train_labels_enc
oded, test_images, test_labels_encoded 
 
#################################################################
## 
# Scale pixel values to between 0 and 1 
x_train, x_test = x_train / 225.0, x_test / 225.0
from keras.utils import to_categorical 
 
y_train_one_hot = to_categorical(y_train) 
y_test_one_hot = to_categorical(y_test) 
VGG_model = VGG16(weights='imagenet', include_top=False, input_sh
ape=(SIZE, SIZE, 3)) 
 
# Make loaded layers as non-
trainable. This is important as we want to work with pre-
trained weights 
for layer in VGG_model.layers: 
    layer.trainable = False 
 
VGG_model.summary()train_feature_extractor=VGG_model.predict(x_train) 
train_features = train_feature_extractor.reshape(train_feature_ex
tractor.shape[0], -1) 
#test features 
test_feature_extractor=VGG_model.predict(x_test) 
test_features = test_feature_extractor.reshape(test_feature_extra
ctor.shape[0], -1) 
 
print(train_features) 
#Printing training dataset features and testing dataset VGG16 
model pool5 features 
test_feature_extractor=VGG_model.predict(x_test) 
test_features = test_feature_extractor.reshape(test_feature_extra
ctor.shape[0], -1) #PCA and GMM 
kd = [16, 24] 
nc = [64, 128] 
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt 
from sklearn.mixture import GaussianMixture 
from sklearn.discriminant_analysis import LinearDiscriminantAnaly
sis 
#clf = LinearDiscriminantAnalysis() 
for k in kd: 
  for n in nc: 
    pc2 = PCA(n_components = k) 
    train_PCA = pc2.fit_transform(X_for_RF) 
    test_PCA = pc2.transform(test_features) 
 
  ##then throw everything in the knn 
    gm = GaussianMixture(n_components= n, random_state=0).fit(tra
in_PCA,test_labels) 
    y_pred = gm.predict(test_PCA) 
    from sklearn.metrics import confusion_matrix 
    cm = confusion_matrix(y_pred, test_labels_encoded) 
    sns.heatmap(cm, cmap="Blues", annot=True, cbar=False ) 
    plt.title("vgg16 - pca gmm ") 
    plt.show()def CalModelFisherVector(f, kd, nc):  
    steps = [('pca', PCA(n_components=kd)), ('gmm', GMM(n_compone
nts=nc))]  
    gmm_model = Pipeline(steps)  
    gmm_model.fit(f)  
    pca_model, gmm = gmm_model  
    return gmm_model, pca_model.transform(f)  
sm1_16_64, s1 = CalModelFisherVector(X, 16, 64) 
sm2_16_128, s2 = CalModelFisherVector(X, 16, 128) 
sm3_24_64, s3 = CalModelFisherVector(X, 24, 64) #FISHER VECTOR CODES: 
def fisher_vector_calculation(xx, gmm):  
    xx = np.atleast_2d(xx)  
    N = xx.shape[0]  
    Q = gmm.predict_proba(xx)  
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N  
    Q_xx = np.dot(Q.T, xx) / N  
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N  
    # Compute derivatives with respect to mixing weights, means a
nd variances.  
    d_pi = Q_sum.squeeze() - gmm.weights_  
    d_mu = Q_xx - Q_sum * gmm.means_  
    d_sigma = (  
            - Q_xx_2  
            - Q_sum * gmm.means_ ** 2  
            + Q_sum * gmm.covariances_  
            + 2 * Q_xx * gmm.means_)  
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))# CODE FOR 1st FV 
L = pd.DataFrame(test_features).fillna(0)  
pca = PCA(n_components=16)  
pca.fit(L)  
ll = pca.transform(L)  
gmm = GMM(n_components=64, covariance_type='diag')  
gmm.fit(ll)  
fv = fisher_vector_calculation(ll, gmm)  
#CODE FOR 2ND FV: 
L = pd.DataFrame(test_features).fillna(0)  
pca1 = PCA(n_components=16)  
pca1.fit(L)  
ll1 = pca1.transform(L)  
gmm1 = GMM(n_components=128, covariance_type='diag',reg_covar=1e-
5)  
gmm1.fit(ll)  
fv1 = fisher_vector_calculation(ll1, gmm1)  
#CODE FOR 3RD FV: 
pca2 = PCA(n_components=24)  
pca2.fit(L)  
ll2 = pca2.transform(L)  
gmm2 = GMM(n_components=64, covariance_type='diag')  
gmm2.fit(ll2)  
fv2 = fisher_vector_calculation(ll2, gmm2)  
#CODE FOR 4TH FV: 
pca3 = PCA(n_components=24)  
pca3.fit(L)  
ll3 = pca3.transform(L)  
gmm3 = GMM(n_components=128, covariance_type='diag',reg_covar=1e-
4)  
gmm3.fit(ll3)  
fv3 = fisher_vector_calculation(ll3, gmm3)  from sklearn.svm import SVC 
 
# import metrics to compute accuracy 
from sklearn.metrics import accuracy_score 
def modelCreation(values, y_values):  
     
    # instantiate classifier with linear kernel and C=1.0 
    linear_svc=SVC(kernel='linear', C=1.0)  
    # fit classifier to training set linear_svc.fit(values,y_values) 
    # make predictions on test set 
    y_pred_test=linear_svc.predict(values) 
    # compute and print accuracy score 
    print('Model accuracy score with linear kernel and C=1.0 : {0
:0.4f}'. format(accuracy_score(y_values, y_pred_test)))  
    cm=confusion_matrix(y_values,y_pred_test) 
    sns.heatmap(cm,annot=True) 
    return y_values,y_pred_test
vg1=modelCreation(s1,test_labels)
#LDA CODE: 
# LDA 
from sklearn.discriminant_analysis import LinearDiscriminantAnaly
sis 
lda = LinearDiscriminantAnalysis(n_components=14) 
xx=lda.fit_transform(train_PCA,y_train) 
print('explained variance ratio (first 14 components): %s' 
% str(lda.explained_variance_ratio_))
