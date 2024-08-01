import pandas as pd
import os, joblib
from sklearn.metrics import confusion_matrix
from conf import select_features, pl_iat
from conf import tor_testpath, nor_testpath, modelpath

# Load a model to identify baseline results

tor = tor_testpath
nor = nor_testpath

if __name__ == '__main__':
    tor_data = pd.read_csv(tor ,low_memory=False, delimiter=',')
    tor_label = tor_data['class1']
    tor_data = tor_data[select_features]
    tor_data = tor_data.replace(' ', 0)

    nor_data = pd.read_csv(nor ,low_memory=False, delimiter=',')
    nor_label = nor_data['class1']
    nor_data = nor_data[select_features]
    nor_data = nor_data.replace(' ', 0)

    data = pd.concat([tor_data, nor_data])
    label = pd.concat([tor_label, nor_label])    

    modelnames = os.listdir(modelpath)
    n=1
    sheet = []
    for model in modelnames:
        clf = joblib.load(modelpath +'/' + model)
        predict = clf.predict(data)
        cfs_mtrx = confusion_matrix(label, predict, labels=['audio','mail','p2p','message','vedio','browser','voip','normal'])
        tmp = [n, model, cfs_mtrx[0][0], cfs_mtrx[0][1], cfs_mtrx[0][2], cfs_mtrx[0][3], cfs_mtrx[0][4], cfs_mtrx[0][5], cfs_mtrx[0][6], cfs_mtrx[0][7]]
        sheet.append(tmp)
        for i in range(1, 8):
            tmp = ['','',cfs_mtrx[i][0],cfs_mtrx[i][1],cfs_mtrx[i][2],cfs_mtrx[i][3],cfs_mtrx[i][4],cfs_mtrx[i][5],cfs_mtrx[i][6],cfs_mtrx[i][7]]
            sheet.append(tmp)
    
    writer = pd.ExcelWriter('data.xlsx')
    df = pd.DataFrame(sheet, columns=['num', 'model', 'audio','browser','vedio','p2p','message','mail','voip','normal'])
    df.to_excel(writer, index=False)
    writer.save()