import pandas as pd
import sys,time, copy, os
import joblib
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from imblearn import over_sampling, under_sampling
import math
from collections import Counter
import warnings
from conf import select_features,pl_iat, modelpath, adaboostpath
from conf import adaboostout, ensembleout, ensemble_strategy
from ensemble import naive_esb,slightly_esb, strong_esb

def O2OS_classify(x_test, y_test, modeldict1, modeldict2):
    print('start')
    z_test = copy.deepcopy(y_test)
    table = PrettyTable(['num','time','all-accuracy','all-precision','all-recall','model-1','acc-1','pre-1','rec-1','model-2','acc-2','pre-2','rec-2'])
    online = []
    offline = []

    if(ensembleout> 0):
        esmb = pd.DataFrame(y_test.values)
        esmb.columns = ['label']
    n = 1
    # online identification
    for model1 in modeldict1:
        clf_1 = joblib.load(modelpath +'/' + model1)
        start_1 = time.time()
        predict_1 = clf_1.predict(x_test[pl_iat])
        end_1 = time.time()

        # evaluate models
        acc_1 = accuracy_score(z_test, predict_1)
        rec_1 = recall_score(z_test, predict_1, pos_label='tor')
        pre_1 = precision_score(z_test, predict_1, pos_label='tor')
        cfs_mtrx_1 = confusion_matrix(z_test, predict_1, labels=['tor', 'normal'])
        tmp1 = [n, model1, cfs_mtrx_1[0][0],cfs_mtrx_1[0][1], acc_1, pre_1,rec_1]
        online.append(tmp1)
        tmp1 = ['', '', cfs_mtrx_1[1][0],cfs_mtrx_1[1][1], '', '','']
        online.append(tmp1)

        #print(model1, " ", pre_1)
        
        # --------------------------------------------------------------------------------------------
        # output misclassification results
        if(adaboostout > 0):
            z_ar = np.array(z_test)
            Findex1 = np.where(z_ar!=predict_1)
            orgdata = pd.concat([x_test, y_test], axis=1)
            Fdata1 = orgdata.iloc[Findex1]
            Fdata1.to_csv(adaboostpath+'\\'+model1+'.csv', index=False)
        # --------------------------------------------------------------------------------------------

        m_test = x_test.iloc[np.where(predict_1=='tor')]
        n_test = y_test.iloc[np.where(predict_1=='tor')]

        # offline classification
        for model2 in modeldict2:
            clf_2 = joblib.load(modelpath +'/' +model2)
            start_2 = time.time()
            predict_2 = clf_2.predict(m_test[select_features])
            #print(Counter(predict_2))
            end_2 = time.time()

            # evaluate models
            acc_2 = accuracy_score(n_test, predict_2)
            rec_2 = recall_score(n_test, predict_2, average='weighted')
            pre_2 = precision_score(n_test, predict_2, average='weighted')
            cfs_mtrx_2 = confusion_matrix(n_test, predict_2, labels=['audio','mail','p2p','message','vedio','browser','voip','normal'])
            tmp2 = [n,model1,model2,cfs_mtrx_2[0][0],cfs_mtrx_2[0][1],cfs_mtrx_2[0][2],cfs_mtrx_2[0][3],cfs_mtrx_2[0][4],cfs_mtrx_2[0][5],cfs_mtrx_2[0][6],cfs_mtrx_2[0][7]]
            offline.append(tmp2)
            
            # ----------------------------------------------------------------------------------------------------------------
            # output misclassification results
            if(adaboostout > 0):
                n_ar = np.array(n_test)
                Findex2 = np.where(n_ar!=predict_2)
                Fdata2 = orgdata.iloc[Findex2]
                Fdata2.to_csv(adaboostpath+'\\'+model2+'.csv', index=False)
            # ----------------------------------------------------------------------------------------------------------------
           
           # ----------------------------------------------------------------------------------------------------------------
            if(ensembleout > 0):
                p = pd.DataFrame(predict_1, columns=['p'])
                q = pd.DataFrame(predict_2, columns=['q'])
                idx = p[p['p']=='tor'].index
                qq = q.values
                q = pd.DataFrame(qq, index=idx, columns=[model1+model2])
                esmb = esmb.join(q[model1+model2])
                
           # ----------------------------------------------------------------------------------------------------------------

            table.add_row([n, end_1-start_1+end_2-start_2, 'all-accuracy',pre_2,rec_1*rec_2,model1,acc_1,pre_1,rec_1,model2,acc_2,pre_2,rec_2])
            n += 1
    if(ensembleout > 0):
        esmb.to_csv('ensemble.csv', index=False)
        #print(esmb)
    return table, online, offline

def ensemble(ensemblecsv):
    esmb = pd.read_csv(ensemblecsv, low_memory=False)
    columns = ['label','0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
    esmb.columns = columns
    esmb = esmb.fillna('non')
    online_label = []
    offline_label = []
    for i in (range(len(esmb))):
        # online
        row = esmb[i:i+1][['0', '4', '8', '12']].values[0]
        row[row!='non']='tor'
        row[row!='tor']='normal'
        counter = list(Counter(row).most_common())
        if ensemble_strategy==1:
            label1 = naive_esb(counter)
        elif ensemble_strategy==2:
            label1 = slightly_esb(counter)
        elif ensemble_strategy==3:
            label1 = strong_esb(counter)
        online_label.append(label1)
        if(label1=='normal'):
            label2 = 'non'

        offline_label.append(label2)

    online = pd.DataFrame(online_label)
    esmb['online']=online
    offline = pd.DataFrame(offline_label)
    esmb['offline']=offline

    label = copy.deepcopy(esmb.iloc[:, 0])
    label[label!='normal']='tor'
    mtrx = confusion_matrix(label, esmb['online'],labels=['tor','normal'])

    tordata = esmb.iloc[np.where(esmb['online']=='tor')]
    mtrx2 = confusion_matrix(tordata['label'], tordata['offline'],labels=['audio','mail','p2p','message','vedio','browser','voip','normal'])
    
    return mtrx, mtrx2

# Generate a dataset Iterator, and generate a dataset according to the ratio of normal traffic and Tor traffic
# A is the lower limit of proportion, b is the upper limit of proportion, nor_ Tor_ Ratio is the number of proportional increments
def get_tor_normal(tor,normal, a=1, b=2, ratio=1):
    while(a<=b):
        tor_num = len(tor)
        nor_num = math.ceil(tor_num*a)
        print('a: ', a, ' normal_num: ', nor_num)

        # 下采样
        nor = normal.sample(nor_num)
        data = pd.concat([tor, nor])
        yield a, data
        a += ratio