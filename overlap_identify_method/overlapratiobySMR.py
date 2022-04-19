import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.metrics import pairwise

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

if __name__ == '__main__':

    g = os.walk(r"E:/gln/C/myclassoverlap/Data/new/all/")
    ratio = []
    Name = []
    ratio1=[]
    EPV=[]
    size=[]
    ss="E:/gln/C/myclassoverlap/RQ1/SMR/"

    for path, dir_list, file_list in g:
        for file_name in file_list:
            print(file_name)

            Name.append(file_name)
            s = os.path.join(path, file_name)
            df = pd.read_csv(s)
            print(df.shape[0])
            print(df.shape)
            # print(df.columns)
            X = df.iloc[:, :df.shape[1]-2].values
            # print(X)
            X=normalize(X)
            print(X.shape)
            y = df.iloc[:, df.shape[1]-1].values
            df_defect=df[df["label"]==1]
            svm = SVC(kernel="linear", C=6.55)
            if(df_defect.shape[0]>=1):
                svm.fit(X, y)
                #rbfs = pairwise.rbf_kernel(X, svm.support_vectors_, gamma=svm.gamma)
                #SMR = np.dot(rbfs, svm.dual_coef_.T) + svm.intercept_
                SMR = np.dot(X, svm.coef_.T) + svm.intercept_
                print(SMR)
            else:
                SMR=np.ones(shape=[df.shape[0],1])
            print(SMR.shape)
            print(df.shape)
            df["SMR"] = SMR
            print(df.shape)
            df_defec = df[df["label"] == 1]
            print(df_defec.shape[0], df.shape[0])
            ratio1.append(df_defec.shape[0] / df.shape[0])
            EPV.append(df_defec.shape[0] / (df.shape[1] - 1))
            size.append(df.shape[0])
            sss = os.path.join(ss, file_name)
            df.to_csv(sss, index=False)


    data_df = pd.DataFrame(np.column_stack((Name,ratio1,EPV,size)))

    # for i in range (df.shape[1]):
    #     data_df.index[i]=df.columns[2]

    data_df.columns = ['Project', 'class ratio','EPV','Size']
    # ss = 'E:\\MyCGAN\\DANN\\RF\\' + file_name
    data_df.to_csv("E:/gln/C/myclassoverlap/RQ1/SMR/all.csv", index=False)