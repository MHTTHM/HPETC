import pandas as pd
import joblib
import copy
import numpy as np
from collections import Counter
import Models_Classifier as models
from imblearn import over_sampling
from conf import select_features,pl_iat, torpath, norpath, modelpath

normal_class = 1 # Whether the normal class is treated as a separate class

Online_model = {'c45':models.c45,'knn':models.knn,'rf30':models.rf30,'gbdt':models.gbdt}
Offline_model = {'c45':models.c45,'knn':models.knn,'rf30':models.rf30,'gbdt':models.gbdt}

# tor_train 27985 samles
nor_num_1 = 30000 #how many normal sampels in online
nor_num_2 = 5000 #how many normal sampels in offline

# generate bordlerline samples
borderline = 0

# output adaboost samples
adaboostout = 0
adaboostpath = r''

def fitmodel(online_dict, offline_dict):
    tor = pd.read_csv(torpath, low_memory=False)
    norl = pd.read_csv(norpath, low_memory=False)
    norl = norl.replace(' ', 0)
    nor = norl.sample(nor_num_1)

    data = pd.concat([tor, nor])
    data = data.replace(' ', 0)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # generate borderline samples during online stage
    if(borderline > 0):
        {'browser': 5000, 'mail': 5000, 'p2p': 5000, 'voip': 5000, 'message': 3797, 'vedio': 2514, 'audio': 1674}

        x = data.iloc[::, 0:-1]
        y = data.iloc[::, -1]
        # improve recall
        #Blsmo = over_sampling.BorderlineSMOTE(kind='borderline-1',sampling_strategy={'browser': 5000, 'mail': 5000, 'p2p': 5000, 'voip': 5000, 'message': 3797, 'vedio': 2514, 'audio': 1674, 'normal': 10000},random_state=42)
        # improve precision
        Blsmo = over_sampling.BorderlineSMOTE(kind='borderline-1',sampling_strategy=online_dict,random_state=42)

        m, n = Blsmo.fit_resample(x, y)
        columns = data.columns.values
        data = pd.concat([m,n], axis=1)
        data.columns = columns
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    features_1 = data[pl_iat]

    label2 = data['class1']

    label1 = copy.deepcopy(label2)
    label1[label1!='normal']='tor'

    print("1阶段: ",Counter(label1))
    for model1 in Online_model:
        # fit online model
        print(model1)
        clf_1 = Online_model[model1]()
        clf_1.fit(features_1,label1)
        # save online model
        joblib.dump(clf_1, modelpath+'/1_'+model1+'.m')
        if(adaboostout > 0):
            predict_1 = clf_1.predict(features_1)
            arr_label1 = np.array(label1)
            index1 = np.where(arr_label1!=predict_1)
            data1 = data.iloc[index1]
            predict_1 = clf_1.predict(norl[pl_iat])
            index1 = np.where(norl['class1']!=predict_1)
            data2 = norl.iloc[index1]
            data1 = pd.concat([data1, data2])
            data1.to_csv(adaboostpath+'\\1_'+model1+'.csv', index=False)

    nor = nor.sample(nor_num_2)
    data = pd.concat([tor, nor])
    data = data.replace(' ', 0)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if(borderline > 0):
        {'browser': 5000, 'mail': 5000, 'p2p': 5000, 'voip': 5000, 'message': 3797, 'vedio': 2514, 'audio': 1674}

        x = data.iloc[::, 0:-1]
        y = data.iloc[::, -1]
        
        #Blsmo = over_sampling.BorderlineSMOTE(kind='borderline-1',sampling_strategy={'browser': 5000, 'mail': 5000, 'p2p': 5000, 'voip': 5000, 'message': 3797, 'vedio': 2514, 'audio': 1674, 'normal': 10000},random_state=42)
        Blsmo = over_sampling.BorderlineSMOTE(kind='borderline-1',sampling_strategy=offline_dict,random_state=42)

        m, n = Blsmo.fit_resample(x, y)
        columns = data.columns.values
        data = pd.concat([m,n], axis=1)
        data.columns = columns
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    features_2 = data[select_features]
    label2 = data['class1']

    print("2阶段: ",Counter(label2))
    for model2 in Offline_model:
        print(model2)
        clf_2 = Offline_model[model2]()
        clf_2.fit(features_2, label2)
        # save the model
        joblib.dump(clf_2, modelpath+'/2_'+model2+'.m')
        if(adaboostout > 0):
            predict_2 = clf_2.predict(features_2)
            arr_label2 = np.array(label2)
            index2 = np.where(arr_label2!=predict_2)
            data1 = data.iloc[index2]
            predict_2 = clf_2.predict(norl[select_features])
            index2 = np.where(norl['class1']!=predict_2)
            data2 = norl.iloc[index2]
            data1 = pd.concat([data1, data2])
            data1.to_csv(adaboostpath+'\\2_'+model2+'.csv', index=False)

def cirfitmodel():
    onlinedict = {'browser': 5000, 'mail': 5000, 'p2p': 5000, 'voip': 5000, 'message': 3797, 'vedio': 2514, 'audio': 1674, 'normal':nor_num_1}
    offlinedict = {'browser': 5000, 'mail': 5000, 'p2p': 5000, 'voip': 5000, 'message': 3797, 'vedio': 2514, 'audio': 1674, 'normal':nor_num_2}
    for i in range(0,10):
        for j in range(0,10):
            offlinedict['normal']+=(j*0.1*nor_num_2)
        onlinedict['browser']+=500
        onlinedict['mail']+=500
        onlinedict
            
    fitmodel(onlinedict, offlinedict)

cirfitmodel()