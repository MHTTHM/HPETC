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
from conf import modelpath, tor_testpath, nor_testpath
from function import O2OS_classify, ensemble, get_tor_normal

warnings.filterwarnings("ignore")

tor = tor_testpath
nor = nor_testpath

# ratio of normal to tor, r
lowbound = 1000
uppbound = 1001
increaseratio = 2

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

    #data_iter = get_dataset(data, 1, 501, 100)
    data_iter = get_tor_normal(tor_data,nor_data, lowbound, uppbound, increaseratio)
    while True:
        try:
            ratio, data = next(data_iter)
            features = data.iloc[:, :-1]
            label = data.iloc[:, -1]
            result, online, offline = O2OS_classify(features, label, modeldict1, modeldict2)
            onlinemtrx, offlinemtrx = ensemble('ensemble.csv')
            print('online:', online)
            print('onlinemtrx:', onlinemtrx)
            #print('offline:', offline)
            #print('offlinemtrx:', offlinemtrx)
            
            if(len(online_result)>0):
                onlinemtrx[0][0] = int(onlinemtrx[0][0]) + int(online_result[-2][0])
                onlinemtrx[0][1] = int(onlinemtrx[0][1]) + int(online_result[-2][1])
                onlinemtrx[1][0] = int(onlinemtrx[1][0]) + int(online_result[-1][0])
                onlinemtrx[1][1] = int(onlinemtrx[1][1]) + int(online_result[-1][1])

            if(len(offline_result)>0):
                for j in range(8):
                    for k in range(8):
                        offlinemtrx[j][k] = int(offlinemtrx[j][k]) + offline_result[j-8][k]

            for i in onlinemtrx:
                online_result.append(i)
            for i in offlinemtrx:
                offline_result.append(i)

            with open('result.txt', 'a') as f:
                f.write('\n')
                f.write('--------------------------------------------------------------------------------------\n')
                ratioline = 'a: '+str(ratio)+'------------------------------------------------------------------------\n'
                f.write(ratioline)
                f.write(str(result))
            
        except StopIteration:
            writer = pd.ExcelWriter('data.xlsx')# pylint: disable=abstract-class-instantiated
            # df1 = pd.DataFrame(online_result, columns=['num','ratio','model1','tor','normal','accuracy','precision','recall'])
            # df2 = pd.DataFrame(offline_result, columns=['num','ratio','model1','model2','audio','browser','vedio','p2p','message','mail','voip','normal'])
            df1 = pd.DataFrame(online_result, columns=['tor', 'normal'])
            df2 = pd.DataFrame(offline_result, columns=['audio','browser','vedio','p2p','message','mail','voip','normal'])
            df1.to_excel(writer,sheet_name='online',index=False)
            df2.to_excel(writer,sheet_name='offline',index=False)
            writer.save()
            sys.exit()
