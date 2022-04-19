import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import os
from sklearn.preprocessing import StandardScaler
from math import log
def __populate(nnarray,n, y,y_label):
    T = 0
    for i in range(n):
        label = y[nnarray[i]]
        if (y_label != label):
            T = 1  # 1代表为重叠实例
            break
    number=0
    for i in range(n):
        label=y[nnarray[i]]
        if(y_label==label):
            number=number+1

    return number,T
def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 样本数
    labelCounts = {}  # 该数据集每个类别的频数
    for featVec in dataSet:  # 对每一行样本
        currentLabel = featVec[-1]  # 该样本的标签
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 计算p(xi)
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt


if __name__ == '__main__':

    g = os.walk(r"E:/gln/C/myclassoverlap/Data/new/all/")
    scar=StandardScaler()
    ratio = []
    Name = []
    ratio1=[]
    EPV=[]
    size=[]

    for path, dir_list, file_list in g:
        for file_name in file_list:
            s = os.path.join(path, file_name)
            df = pd.read_csv(s)
            print(file_name)
            Name.append(file_name)
            s1='E:/gln/C/myclassoverlap/Data/new/all/'+file_name
            df1=pd.read_csv(s1)
            laplabel_3=np.zeros(shape=[df.shape[0], 1])
            laplabel_4 = np.zeros(shape=[df.shape[0], 1])
            laplabel_5 = np.zeros(shape=[df.shape[0], 1])
            # laplabel1 = np.zeros(shape=[df.shape[0], 1])
            # print(df.columns)
            X = df.iloc[:, :df.shape[1]-2].values
            X=normalize(X)
            # print(X)
            y = df.iloc[:, df.shape[1]-1].values
            df_defect=df[df["label"]==1]
            if(df_defect.shape[0]>0):
                nbrs1 = NearestNeighbors(n_neighbors=5, algorithm="auto", metric="euclidean")
                nbrs1.fit(X)
                nnarray1 = nbrs1.kneighbors(X)[1]
                # print(nnarray1.shape,df.shape)
                for i in range(nnarray1.shape[0]):
                    y_label = y[i]
                    num, T = __populate(nnarray1[i], 5, y, y_label)
                    if (num < 3):
                        laplabel_3[i] = 1
                    if(num<4):
                        laplabel_4[i] = 1
                    if (num < 5):
                        laplabel_5[i] = 1

                # if (num<3 and y[i]==0):
                #     laplabel[i] = 1

            # df["removelabel"]=laplabel
            df["laplabel_3"]=laplabel_3
            df["laplabel_4"] = laplabel_4
            df["laplabel_5"] = laplabel_5
            #df["loc"]=df1["loc"]
            ss = 'E:/gln/C/myclassoverlap/RQ1/KNN/' + file_name
            df.to_csv(ss, index=False)






