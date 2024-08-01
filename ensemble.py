import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import copy
from conf import ensemble_strategy

file = 'ensemble.csv'

# It can be simplified in the future
# naive, According to the principle of majority voting. if 2:2, select normal.
def naive_esb(counter):
    lenght = len(counter)
    if(lenght==1):
        result = counter[0][0]
    elif(lenght==2):
        if(counter[0][1]==3):
            result = counter[0][0]
        else:#(2:2)
            if((counter[0][0]=='non') or (counter[1][0]=='non') or (counter[0][0]=='normal') or (counter[1][0]=='normal')):
                result = 'normal'
            else:
                result = counter[0][0]
    elif(lenght==3):
        result = counter[0][0]
    else:
        result = 'normal'
    return result

# The weak negation method, as long as there is a normal classification, is normal, and others are classified according to original.
def slightly_esb(counter):
    lenght = len(counter)
    if(lenght==1):
        result = counter[0][0]
    elif(lenght==2):
        if(counter[0][1]==3):
            result = counter[0][0]
        else:#(2:2)
            if((counter[0][0]=='non') or (counter[1][0]=='non') or (counter[0][0]=='normal') or (counter[1][0]=='normal')):
                result = 'normal'
            else:
                result = counter[0][0]
        if((counter[0][0]=='normal') or (counter[1][0]=='normal') or (counter[0][0]=='non') or (counter[1][0]=='non')):
            result = 'normal'
    elif(lenght==3):
        if((counter[0][0]=='normal') or (counter[1][0]=='normal') or (counter[2][0]=='normal') or (counter[0][0]=='non') or (counter[1][0]=='non') or (counter[2][0]=='non')):
            result = 'normal'
        else:
            result = counter[0][0]
    else:
        result = 'normal'
    return result

# The strong negation method is classified as normal as long as there are significant differences.
def strong_esb(counter):
    lenght = len(counter)
    if(lenght==1):
        result = counter[0][0]
    elif(lenght==2):
        if(counter[0][1]==3):
            result = counter[0][0]
        else:
            result = 'normal'
    else:
        result = 'normal'
    return result

if __name__ == '__main__':
    data = pd.read_csv(file, low_memory=False)
    orgcolumns = data.columns
    columns = ['label','0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
    data.columns = columns
    data = data.fillna('non')
    online_label = []
    offline_label = []
    for i in (range(len(data))):
        # online
        row = data[i:i+1][['0', '4', '8', '12']].values[0]
        #row[row=='non']='normal'
        #row[row!='normal']='tor'
        row[row!='non']='tor'
        row[row!='tor']='normal'
        counter = list(Counter(row).most_common())
        label1 = naive_esb(counter)
        #label1 = slightly_esb(counter)
        #label1 = strong_esb(counter)
        online_label.append(label1)
        if(label1=='normal'):
            label2 = 'non'
        else:
            for j in range(0, 13, 4):
                if(data[i:i+1][columns[j+1]].values[0]!='non'):
                    if ensemble_strategy==1:
                        label2 = naive_esb(list(Counter(data[i:i+1][[columns[j+1], columns[j+2],columns[j+3],columns[j+4]]].values[0]).most_common()))
                    elif ensemble_strategy==2:
                        label2 = slightly_esb(list(Counter(data[i:i+1][[columns[j+1], columns[j+2],columns[j+3],columns[j+4]]].values[0]).most_common()))
                    elif ensemble_strategy==3:
                        label2 = strong_esb(list(Counter(data[i:i+1][[columns[j+1], columns[j+2],columns[j+3],columns[j+4]]].values[0]).most_common()))
                    continue
        offline_label.append(label2)
    
    # save the results
    online = pd.DataFrame(online_label)
    data['online']=online
    offline = pd.DataFrame(offline_label)
    data['offline']=offline
    data.to_csv('data.csv', index=False)

    # ouput online results
    label = copy.deepcopy(data.iloc[:, 0])
    label[label!='normal']='tor'
    acc = accuracy_score(label, data['online'])
    pre = precision_score(label, data['online'], pos_label='tor')
    recall = recall_score(label, data['online'], pos_label='tor')
    mtrx = confusion_matrix(label, data['online'],labels=['tor','normal'])
    print("precision: ", pre)
    print('recall: ', recall)
    print("accuracy: ", acc)
    print("confusion matrix: \n", mtrx)

    # output offline results
    tordata = data.iloc[np.where(data['online']=='tor')]
    print(len(tordata))
    mtrx2 = confusion_matrix(tordata['label'], tordata['offline'],labels=['audio','mail','p2p','message','vedio','browser','voip','normal'])
    print("confusion matrix2: \n", mtrx2)