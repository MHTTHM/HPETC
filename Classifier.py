import pandas as pd
import joblib
import sys
import numpy as np
from prettytable import PrettyTable
import Models_Classifier as models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn import metrics
import time
from conf import select_features, pl_iat
from conf import torpath, norpath, modelpath

# normal samples
nor_num = 5000

# online or offline
featrues_use = select_features
#featrues_use = pl_iat

tor = pd.read_csv(torpath, low_memory=False)
norl = pd.read_csv(norpath, low_memory=False)
nor = norl.sample(nor_num)

data = pd.concat([tor, nor])
data = data.replace(' ', 0)

# choose features and labels
#data = data[[*pl_iat,'class1']]
#data.loc[data['class1']!='normal','class1']='tor'

data = data[[*featrues_use, 'class1']]

# get train set
dataset = data.values
features = dataset[::, 0:-1]
label = dataset[::, -1]

x_train,x_test,y_train,y_test = train_test_split(features,label,test_size=0.1)

# select models
modeldict = {'c45':models.c45, 'cart':models.cart,'knn':models.knn,'lrc':models.lrc,'rf10':models.rf10,'rf20':models.rf20,'rf30':models.rf30,'gbdt':models.gbdt,'AdaBoost':models.AdaBoost,'gnb':models.gnb,'lda':models.lda,'qda':models.qda,'svm':models.svm}

modeldict = {'c45':models.c45,'knn':models.knn,'rf30':models.rf30,'gbdt':models.gbdt}


tabel = PrettyTable(['num','model','accuracy','precision','act\\predict','predict tor','predict nor'])
n = 1
for model in modeldict:
    clf = modeldict[model]()
    start = time.time()
    #clf.fit(x_train,y_train)
    clf.fit(features, label)
    # save models
    joblib.dump(clf, modelpath+'/'+model+'.m')

    predict = clf.predict(x_test)
    end = time.time()
    
    #pd_predict = pd.DataFrame({'predict':predict})
    #x_test.index = np.arange(len(x_test))
    #pd_predict.index = np.arange(len(pd_predict))
    #data = pd.concat([x_test,pd_predict], axis=1)
    
    result = accuracy_score(y_test, predict)
    precision = precision_score(y_test, predict, average='macro')
    
    print('------------------------------------------------------------------------------------------------')
    print(model, end-start)
    print(precision)
    confusion_matrix_result = metrics.confusion_matrix(y_test, predict, labels=['audio','mail','p2p','vedio','browser','voip','normal'])
    print(confusion_matrix_result)
    print('------------------------------------------------------------------------------------------------')

    '''
    a = 0
    for i in range(len(precision)):
        a = a+precision[i]
    precision = a/(len(precision))
    '''
    '''
    confusion_matrix_result = metrics.confusion_matrix(y_test, predict)
    tabel.add_row([n,model,result,precision,'tor',confusion_matrix_result[0][0],confusion_matrix_result[0][1]])
    tabel.add_row(['','','','','nor',confusion_matrix_result[1][0],confusion_matrix_result[1][1]])
    '''

    n = n+1

print(tabel)


# features rank
'''
clf = models.rf30()
clf.fit(x_train,y_train)
importamce = clf.feature_importances_ / np.max(clf.feature_importances_)
print(sorted(zip(map(lambda x: round(x, 4), importamce), numfeature.pop()), reverse=True))

indices = np.argsort(importamce)[::-1]
a = []
for f in range(x_train.shape[1]):
    a.append(numfeature[indices[f]])
    print("%2d) %-*s %f" % (f + 1, 30, numfeature[indices[f]], importamce[indices[f]]))

print(a)
'''