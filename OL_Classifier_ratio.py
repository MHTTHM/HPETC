import pandas as pd
import sys,time, copy, os
import joblib
import numpy as np
from prettytable import PrettyTable
from sklearn import ensemble
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from imblearn import over_sampling, under_sampling
import math
from collections import Counter
import warnings
from conf import torpath, norpath, modelpath
from function import O2OS_classify, get_tor_normal

warnings.filterwarnings("ignore")

tor = torpath
nor = norpath

lowbound = 1
uppbound = 1000
increaseratio = 10

# output misclassification sampels# dont use!
adaboostout = 0

# output prediction labels for ensemble tech.
ensembleout = 0

if __name__ == '__main__':
    tor_data = pd.read_csv(tor ,low_memory=False, delimiter=',')
    tor_data = tor_data.replace(' ', 0)
    nor_data = pd.read_csv(nor ,low_memory=False, delimiter=',')
    nor_data = nor_data.replace(' ', 0)
    
    modelnames = os.listdir(modelpath)
    modeldict1 = [model for model in modelnames if model.find('1_')>=0]
    modeldict2 = [model for model in modelnames if model.find('2_')>=0]

    online_result = []
    offline_result = []

    data_iter = get_tor_normal(tor_data,nor_data, lowbound, uppbound, increaseratio)
    while True:
        try:
            ratio, data = next(data_iter)
            features = data.iloc[:, :-1]
            label = data.iloc[:, -1]
            result, online, offline = O2OS_classify(features, label, modeldict1, modeldict2)
            print(online)

            for i in online:
                i.insert(1,'') if i[0]=='' else i.insert(1, ratio)
                online_result.append(i)
            for i in offline:
                i.insert(1,'') if i[0]=='' else i.insert(1, ratio)
                offline_result.append(i)

            with open('result.txt', 'a') as f:
                f.write('\n')
                f.write('--------------------------------------------------------------------------------------\n')
                ratioline = 'a: '+str(ratio)+'------------------------------------------------------------------------\n'
                f.write(ratioline)
                f.write(str(result))
            
        except StopIteration:
            writer = pd.ExcelWriter('data.xlsx')# pylint: disable=abstract-class-instantiated
            df1 = pd.DataFrame(online_result, columns=['num','ratio','model1','tor','normal','accuracy','precision','recall'])
            df2 = pd.DataFrame(offline_result, columns=['num','ratio','model1','model2','audio','browser','vedio','p2p','message','mail','voip','normal'])
            df1.to_excel(writer,sheet_name='online',index=False)
            df2.to_excel(writer,sheet_name='offline',index=False)
            writer.save()
            sys.exit()
