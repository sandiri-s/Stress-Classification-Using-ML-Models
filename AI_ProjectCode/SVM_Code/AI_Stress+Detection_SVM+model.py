
# coding: utf-8

# In[ ]:

import scipy.io as scy
import numpy as np
import pandas as pd
import random as rnd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.metrics import accuracy_score
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

import matplotlib.pyplot as plt
#import tables as tb
get_ipython().magic('matplotlib inline')



def readFile(filename):
    with open(filename, 'r') as csvfile:
        data = pd.read_csv(csvfile)
    #print(data)
    colsName = data.columns.values
    #print(colsName)
    featuresName = colsName[:len(colsName)-1]
    labelName = colsName[len(colsName)-1:len(colsName)]
    print(featuresName)

    x_data = data[featuresName].values
    y_data = data[labelName].values
    X_data = preprocessdata(x_data) #normalize
    #print (x_data)
    #print (y_data)
    return (X_data, y_data)

# Standardize the data since attributes are of varying scales
def preprocessdata(X):
    scaler = MinMaxScaler(feature_range=(0, 1))
    res_X = scaler.fit_transform(X)
    return(res_X)

def splitData(X, y, tstsize):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = tstsize)
    return(X_train, X_test, y_train, y_test)

def SelectFeactureUnivar(X,y):
    yy = y.flatten()
    selector = SelectKBest(f_classif, k = 3)
    selector.fit(X,yy)
    FeatureNames = []
    print(selector.pvalues_)
    X_new = selector.fit_transform(X, yy)
    #print(X_new.shape)
    return X_new
            
def SelectFeacturePCA(X,y):
    scaler = StandardScaler()
    res_X = scaler.fit_transform(X)
    X_norm, y_norm =res_X , y.flatten()
    #print(X_norm)
    pca = PCA(n_components=3) 
    fitval = pca.fit(X_norm)
    X_pca = pca.fit_transform(X_norm)
    varience = pca.explained_variance_
    print(varience)
    return X_pca
    

X, y = readFile('Dataset\All_subjects.csv')
X_new = SelectFeactureUnivar(X,y)
X_pca = SelectFeacturePCA(X,y)
OX_trn, OX_tst, Oy_trn, Oy_tst = splitData(X, y, tstsize = 0.2)
X_trn, X_tst, y_trn, y_tst = splitData(X_new, y, tstsize = 0.2)
PX_trn, PX_tst, Py_trn, Py_tst = splitData(X_pca, y, tstsize = 0.2)
#X_new2 = 




##"Next Cross Validate --split"
#"Normalize data using"
# Visualize training using PCA
# Feature selection http://scikit-learn.org/stable/modules/feature_selection.html



# In[ ]:

### Dataset Visualization using PCA


# In[ ]:


scaler = MinMaxScaler(feature_range=(0, 1))
res_X = scaler.fit_transform(OX_trn)
X_norm, y_norm =res_X ,Oy_trn.flatten()
print(X_norm.shape)
pca = PCA(n_components=2) 
fitval = pca.fit(X_norm)
pca_X = pd.DataFrame(pca.fit_transform(X_norm))
varience = pca.explained_variance_
print("Original shape:   ", X_trn.shape)
print("Transformed shape:", pca_X.shape)


# In[ ]:

plt.figure(figsize=(8, 6))
plt.scatter(pca_X[y_norm == 1][0], pca_X[y_norm == 1][1], s = 60, label='relax', c='navy')
plt.scatter(pca_X[y_norm == 2][0], pca_X[y_norm == 2][1], s = 60, label='physical', c='turquoise')
plt.scatter(pca_X[y_norm == 5][0], pca_X[y_norm == 5][1], s = 60, label='cognitive', c='darkorange')
plt.scatter(pca_X[y_norm == 7][0], pca_X[y_norm == 7][1], s = 60, label='emotional', c='red')

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.show()


# ### SVM with simplified SMO algorithm 

# In[ ]:

class SVMSMO_Model():
    def __init__(self, max_iter=10000, k='linear', C=1.0, tol=0.001):
        self.max_iter = max_iter
        self.C = C
        self.tol = tol 
        #self.kernel_
        if k == 'linear':
            self.kernel = self.kernel_linear

    def get_rnd_int(self, j,k,l):
        i = l
        c=0
        while i == l and c<1000:
            i = rnd.randint(j,k)
            c+=1
        return i
    
    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)
    
    def bias(self, X, y, w):
        b = np.mean(y - np.dot(w.T, X.T))
        return b
    
    def weights(self, alpha, y, X):
        w = np.dot(alpha * y, X)
        return w
    
    def fx_i(self, X, w, b):
        prb = np.dot(w.T, X.T) + b
        fx = np.sign(np.dot(w.T, X.T) + b).astype(int)
        return fx, prb
    
    def comLandH(self, C, oldalpha_i, oldalpha_j, yi, yj):
        if(yj != yi):
            return (max(0,  oldalpha_i -  oldalpha_j), min(C, C + oldalpha_i - oldalpha_i))
        else:
            return (max(0,  oldalpha_j +  oldalpha_i - C), min(C,  oldalpha_j +  oldalpha_i))
        
    def fit(self, X, y):
        m = X.shape[0]
        alphas = np.zeros((m))
        kernel = self.kernel
        passes = 0
        while True:
            passes += 1
            old_alphas = np.copy(alphas)
            for i in range(0, m):
                j = self.get_rnd_int(0, m-1, i) # Get random j
                #x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
                eta = kernel(X[j,:], X[j,:]) + kernel(X[i,:], X[i,:]) - 2 * kernel(X[j,:], X[i,:])
                
                if eta == 0:
                    continue
                    
                oldalpha_i, oldalpha_j = alphas[i], alphas[j]
                (L, H) = self.comLandH(self.C, oldalpha_i, oldalpha_j, y[i], y[j])

                # parameters
                self.w = self.weights(alphas, y, X)
                self.b  = self.bias(X, y, self.w)

                # Get E_i, E_j
                E_i = self.fx_i(X[i,:], self.w, self.b)[0] - y[i]
                E_j = self.fx_i(X[j,:], self.w, self.b)[0] - y[j]
                
                # update alphas
                alphas[i] = oldalpha_i + np.float(y[i] * (E_j - E_i))/eta
                alphas[i] = max(alphas[i], L)
                alphas[i] = min(alphas[i], H)

                alphas[j] =  oldalpha_j + y[i]*y[j] * ( oldalpha_i - alphas[i])

            #Are the alphas converging
            change = np.linalg.norm(alphas - old_alphas)
            if change < self.tol:
                break

            if passes >= self.max_iter:
                print("Reached a max of %d iterations" % (self.max_iter))
                return
            
        
        self.b = self.bias(X, y, self.w)
        self.w = self.weights(alphas, y, X)
        #print(self.w)
        
        #support vectors can be computed as follows
        alp_idx = np.where(alphas > 0)[0]
        support_vectors = X[alp_idx, :]
        
        return self.w, self.b
    
    def predict(self, X_tst):
        y, prb = self.fx_i(X_tst, self.w, self.b)
        return y, prb 
    
    


# ### Train four svm models 
# (One vs all wrapper)

# In[ ]:

def TrainModel(X_trn,y_trn):
    SVMs = []
    classes = [1,2,5,7]
    fits = []
    for i in range(len(classes)):
        #print ('\nThe %d/%dth classifier training...' % (i+1, len(classes)))
        y_train = deepcopy(y_trn)
        idx_i = y_train == i
        y_train[idx_i] = 1
        y_train[~idx_i] = -1
        model = SVMSMO_Model(max_iter=10000, k ='linear', C=1.0, tol=0.001)
        fit = model.fit(X_trn, y_train)
        fits.append(fit)
        SVMs.append(model)
    return SVMs, fits


# ### Predictions

# In[ ]:

def onevsall_pred(SVMs, X, classes):
    NumClasses = len(classes)
    print(NumClasses)
    predscore = np.zeros((NumClasses, X.shape[0]))
    #print(predscore)
    for i in range (NumClasses):
        svm = SVMs[i]
        predscore[i, :] = svm.predict(X)[1]
    pred = np.argmax(predscore, axis=0)
    y_pred = deepcopy(pred) 
    
    for i in range(len(y_pred)):
        if y_pred[i] == 0 :
            y_pred[i] = 1 
        elif y_pred[i] == 1:
            y_pred[i] = 2 
        elif y_pred[i] == 2:
            y_pred[i] = 5
        elif y_pred[i] == 1:
            y_pred[i] = 7 
    

    #print(y_pred)
    
    return y_pred


# ### Evaluation Functions

# In[ ]:

def accuracy(y, y_pred):
    acc = accuracy_score(y, y_pred)
    return acc
#function as illustrated in sklearn 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ### Models Evaluated  using 70-30 Split

# ### Model with all features

# In[ ]:

Oy_train = Oy_trn.flatten()
Oy_test = Oy_tst.flatten()
SVMs, losses = TrainModel(OX_trn, Oy_train)
Opred_ytrn = onevsall_pred(SVMs, OX_trn, classes= [1,2,5,7])
Oy_pred = onevsall_pred(SVMs, OX_tst, classes= [1,2,5,7])
print('Training  accuracy: %f' % (accuracy(Oy_train, Opred_ytrn)))
print ('Test accuracy: %f' % (accuracy(Oy_test, Oy_pred)))


# ### Confusion matrix 

# In[ ]:


cnf_matrix = confusion_matrix(Oy_test, Oy_pred)
np.set_printoptions(precision=2)
class_names = ['relaxed', 'physical', 'cognitive',  'emotional']
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# ### Model with 3 Best features from Univariate Feature Selection

# In[ ]:

y_train = y_trn.flatten()
y_test = y_tst.flatten()
SVMs, losses = TrainModel(X_trn, y_train)
pred_ytrn = onevsall_pred(SVMs, X_trn, classes= [1,2,5,7])
#print(pred_train_one_vs_all)
y_pred = onevsall_pred(SVMs, X_tst, classes= [1,2,5,7])
print('Training dataset accuracy: %f' % (accuracy(y_train, pred_ytrn)))
print ('Test datast accuracy: %f' % (accuracy(y_tst, y_pred)))


# In[ ]:

cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
class_names = ['relaxed', 'physical', 'cognitive',  'emotional']
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# ### Model with 3 features from PCA 

# In[ ]:

Py_train = Py_trn.flatten()
Py_test = Py_tst.flatten()

SVMs, losses = TrainModel(PX_trn, Py_train)
Ppred_ytrn = onevsall_pred(SVMs, PX_trn, classes= [1,2,5,7])
Py_pred = onevsall_pred(SVMs, PX_tst, classes= [1,2,5,7])
print('Training dataset accuracy: %f' % (accuracy(Py_train, pred_ytrn)))
print ('Test datast accuracy: %f' % (accuracy(Py_test, Py_pred)))


# In[ ]:

cnf_matrix = confusion_matrix(Py_test, Py_pred)
np.set_printoptions(precision=2)
class_names = ['relaxed', 'physical', 'cognitive',  'emotional']
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# ### Models evaluated using Leave one Subject out Cross validation 

# ### Model with All Features

# In[ ]:

groupsO = np.array([1, 1, 1 , 1 , 2 , 2 , 2 , 2 , 3 , 3 , 3 , 3 , 4 , 4 , 4 , 4 ,
     5 , 5 , 5 , 5 ,  6 ,  6 ,  6 ,  6 ,  7 ,  7 ,  7 ,  7 ,  8 ,  8 ,  8 ,  8 ,
      9 ,  9 ,  9 ,  9 ,  10 ,  10 ,  10 ,  10 ,  11 ,  11 ,  11 ,  11 ,  12 ,  12 ,  12 ,  12 ,
      13 ,  13 ,  13 ,  13 ,  14 ,  14 ,  14 ,  14 ,  15 ,  15 ,  15 ,  15 ,  16 ,  16 ,  16 ,  16])

loso = LeaveOneGroupOut()
OTrain_acc = []
OTest_acc =[]
for train_index, test_index in loso.split(X, y, groupsO):
    OX_train, OX_test = X[train_index], X[test_index]
    Oy_train, Oy_test = y[train_index], y[test_index]
    Oy_train = Oy_train.flatten()
    Oy_test = Oy_test.flatten()
    OSVMs, Olosses = TrainModel(OX_train, Oy_train)
    Opred_ytrn = onevsall_pred(OSVMs, OX_train, classes= [1,2,5,7])
    Oy_pred = onevsall_pred(OSVMs, OX_test, classes= [1,2,5,7])
    Otrnacc = accuracy(Oy_train, Opred_ytrn)
    Otstacc = accuracy(Oy_test, Oy_pred)
    OTrain_acc.append(Otrnacc)
    OTest_acc.append(Otstacc)

Otrain = np.array(OTrain_acc)
OTrnAcc=np.mean(Otrain)

Otest = np.array(OTest_acc)
OTstAcc=np.mean(Otest)

print('Training dataset accuracy: %f' % (OTrnAcc))
print ('Test datast accuracy: %f' % (OTstAcc))


# In[ ]:

### Model with 3 Best features from Univariate Feature Selection


# In[ ]:

groups = np.array([1, 1, 1 , 1 , 2 , 2 , 2 , 2 , 3 , 3 , 3 , 3 , 4 , 4 , 4 , 4 ,
     5 , 5 , 5 , 5 ,  6 ,  6 ,  6 ,  6 ,  7 ,  7 ,  7 ,  7 ,  8 ,  8 ,  8 ,  8 ,
      9 ,  9 ,  9 ,  9 ,  10 ,  10 ,  10 ,  10 ,  11 ,  11 ,  11 ,  11 ,  12 ,  12 ,  12 ,  12 ,
      13 ,  13 ,  13 ,  13 ,  14 ,  14 ,  14 ,  14 ,  15 ,  15 ,  15 ,  15 ,  16 ,  16 ,  16 ,  16])

loso = LeaveOneGroupOut()
Train_acc = []
Test_acc =[]
for train_index, test_index in loso.split(X_new, y, groups):
    X_trn, X_tst = X[train_index], X[test_index]
    y_trn, y_test = y[train_index], y[test_index]
    y_train = y_trn.flatten()
    y_test = y_tst.flatten()
    SVMs, Olosses = TrainModel(X_trn, y_train)
    pred_ytrn = onevsall_pred(SVMs, X_trn, classes= [1,2,5,7])
    y_pred = onevsall_pred(SVMs, X_tst, classes= [1,2,5,7])
    trnacc = accuracy(y_train, pred_ytrn)
    tstacc = accuracy(y_test, y_pred)
    Train_acc.append(trnacc)
    Test_acc.append(tstacc)

train = np.array(Train_acc)
TrnAcc=np.mean(train)

test = np.array(Test_acc)
TstAcc=np.mean(test)

print('Training dataset accuracy: %f' % (OTrnAcc))
print ('Test datast accuracy: %f' % (OTstAcc))


# In[ ]:

### Model with 3 features from PCA 


# In[ ]:

groupsP = np.array([1, 1, 1 , 1 , 2 , 2 , 2 , 2 , 3 , 3 , 3 , 3 , 4 , 4 , 4 , 4 ,
     5 , 5 , 5 , 5 ,  6 ,  6 ,  6 ,  6 ,  7 ,  7 ,  7 ,  7 ,  8 ,  8 ,  8 ,  8 ,
      9 ,  9 ,  9 ,  9 ,  10 ,  10 ,  10 ,  10 ,  11 ,  11 ,  11 ,  11 ,  12 ,  12 ,  12 ,  12 ,
      13 ,  13 ,  13 ,  13 ,  14 ,  14 ,  14 ,  14 ,  15 ,  15 ,  15 ,  15 ,  16 ,  16 ,  16 ,  16])

loso = LeaveOneGroupOut()
PTrain_acc = []
PTest_acc =[]
for train_index, test_index in loso.split(X_pca, y, groupsP):
    PX_trn, PX_tst = X[train_index], X[test_index]
    Py_trn, Py_tet = y[train_index], y[test_index]
    Py_train = Py_trn.flatten()
    Py_test = Py_tst.flatten()
    PSVMs, Plosses = TrainModel(PX_trn, Py_train)
    Ppred_ytrn = onevsall_pred(PSVMs, PX_trn, classes= [1,2,5,7])
    Py_pred = onevsall_pred(SVMs, PX_tst, classes= [1,2,5,7])
    Ptrnacc = accuracy(Py_train, Ppred_ytrn)
    Ptstacc = accuracy(Py_test, Py_pred)
    PTrain_acc.append(Ptrnacc)
    PTest_acc.append(Ptstacc)

Ptrain = np.array(PTrain_acc)
PTrnAcc=np.mean(Ptrain)

Ptest = np.array(PTest_acc)
PTstAcc=np.mean(Ptest)

print('Training dataset accuracy: %f' % (PTrnAcc))
print ('Test datast accuracy: %f' % (PTstAcc))


# In[ ]:

#SVM from a package
from sklearn import svm
from sklearn.metrics import accuracy_score

#Datasets
OXlib= PX_trn
OXlibtst= PX_tst

#tranforming input binary for fast model training
#X_trn = X_train.where((X_train > 0), 1)
#print (X_trn)

#X = TrainData.ix[:,1:785].values

#Y_new = np.array(Y_trn.value)
Oylib = Py_trn.flatten()
Oylibtst = Py_tst.flatten()


#print(X.shape)
#print(y.shape)
#X_test = 

C = 0.1 # regularization parameter
t = 0.001 # tolerance
maxiter = 10000 #maximum iteration


#Rbf svm
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C, tol=t, max_iter=maxiter).fit(OXlib, Oylib)

#Polynomial svm
poly_svc = svm.SVC(kernel='poly', degree=3, C=C, max_iter=maxiter).fit(OXlib, Oylib)

#sigmoid svm
sig_svc = svm.SVC(kernel='sigmoid', gamma=0.7, C=C, tol=t, max_iter=maxiter).fit(OXlib, Oylib)

pred_svcLK = svc_Lin_kernel.predict(OXlibtst)
pred_rbf_svc = rbf_svc.predict(OXlibtst)
pred_poly_svc = poly_svc.predict(OXlibtst)
pred_lin_svc = lin_svc.predict(OXlibtst)
pred_sig_svc = sig_svc.predict(OXlibtst)

acc_svcLK = accuracy_score(Oylibtst, pred_svcLK)
acc_lin_svc = accuracy_score(Oylibtst, pred_lin_svc)
acc_rbf_svc = accuracy_score(Oylibtst, pred_rbf_svc )
acc_poly_svc = accuracy_score(Oylibtst, pred_poly_svc)
acc_sig_svc = accuracy_score(Oylibtst, pred_sig_svc)


print("RBF accuracy", acc_rbf_svc)
print("Poly accuracy",acc_poly_svc)
print("Sigmoid accuracy",acc_sig_svc)


