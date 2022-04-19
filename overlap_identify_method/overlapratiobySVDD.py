import numpy as np
import pandas as pd
import os
from sklearn import svm

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

if __name__ == '__main__':
    g = os.walk(r"E:/gln/C/myclassoverlap/Data/new/all/")
    ratio = []
    Name = []
    ratio1=[]
    EPV=[]
    size=[]
    ss="E:/gln/C/myclassoverlap/RQ1/SVDD_new1/"

    for path, dir_list, file_list in g:
        for file_name in file_list:
            print(file_name)
            Name.append(file_name)
            svgg=[]
            s = os.path.join(path, file_name)
            df = pd.read_csv(s)
            print(df.shape)
            X=df.iloc[:, :df.shape[1]-2].values
            y=df.iloc[:,df.shape[1]-1].values
            X=normalize(X)
            df_new=pd.DataFrame(X)
            df_new["label"]=df["label"]
            df_defect = df_new[df_new["label"] == 1]
            df_clean = df_new[df_new["label"] == 0]
            X_defect = df_defect.iloc[:, :df_defect.shape[1] - 1].values
            X_clean = df_clean.iloc[:, :df_clean.shape[1] - 1].values
            clf_defect=svm.OneClassSVM(nu=0.1,kernel='rbf',gamma=0.1)
            if(df_defect.shape[0]>0):
                clf_defect.fit(X_defect)
            clf_clean=svm.OneClassSVM(nu=0.1,kernel='rbf',gamma=0.1)
            clf_clean.fit(X_clean)
            for i in range(df.shape[0]):
                if((df_defect.shape[0]>0) and(clf_defect.decision_function(X[i,:].reshape(1, -1))>0) and (clf_clean.decision_function(X[i,:].reshape(1, -1)))>0):
                    svgg.append(1)
                else:
                    svgg.append(0)
                # if(y[i]==0):
                #
                #     if((df_defect.shape[0]>0) and (clf_defect.decision_function(X[i,:].reshape(1, -1))>0)):
                #         svgg.append(1)
                #     else:
                #         svgg.append(0)
                # elif(y[i]==1):
                #     if(clf_clean.decision_function(X[i,:].reshape(1, -1))>0):
                #         svgg.append(1)
                #     else:
                #         svgg.append(0)
            df["svgg"] = svgg
            df_defec = df[df["label"] == 1]
            print(df_defec.shape[0], df.shape[0])
            ratio1.append(df_defec.shape[0] / df.shape[0])
            EPV.append(df_defec.shape[0] / (df.shape[1] - 1))
            size.append(df.shape[0])
            sss = os.path.join(ss, file_name)
            df.to_csv(sss, index=False)

    data_df = pd.DataFrame(np.column_stack((Name, ratio1, EPV, size)))

    # for i in range (df.shape[1]):
    #     data_df.index[i]=df.columns[2]

    data_df.columns = ['Project', 'class ratio', 'EPV', 'Size']
    # ss = 'E:\\MyCGAN\\DANN\\RF\\' + file_name
    data_df.to_csv("E:/gln/C/myclassoverlap/RQ1/SVDD_new1/all.csv", index=False)
