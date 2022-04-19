import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import os

def flipdata(df,df_defect,df_clean):
   X_defect=df_defect.iloc[:,:df_defect.shape[1]-1].values
   X_clean = df_clean.iloc[:, :df_clean.shape[1] - 1].values
   X=df.iloc[:,:df.shape[1]-1].values
   y=df.iloc[:,df.shape[1]-1].values
   nbrs1 = NearestNeighbors(n_neighbors=3)
   nbrs1.fit(X)
   nnarray1 = nbrs1.kneighbors(X_defect)[1]
   print(nnarray1)
   for i in range(nnarray1.shape[0]):
       for j in range(3):
           if(y[nnarray1[i][j]]==0):
               y[nnarray1[i][j]]=1

   nnarray2 = nbrs1.kneighbors(X_clean)[1]
   print(nnarray2)
   for i in range(0,nnarray2.shape[0]):
       for j in range(0,3):
           if (y[nnarray2[i][j]] == 1):
               y[nnarray2[i][j]] = 0
   df["label"] = y
   return df

# def flipdata(df,df_clean):
#    X_clean = df_clean.iloc[:, :df_clean.shape[1] - 1].values
#    X=df.iloc[:,:df.shape[1]-1].values
#    y=df.iloc[:,df.shape[1]-1].values
#    nbrs1 = NearestNeighbors(n_neighbors=5)
#    nbrs1.fit(X)
#    nnarray2 = nbrs1.kneighbors(X_clean)[1]
#    print(nnarray2)
#    for i in range(0,nnarray2.shape[0]):
#        for j in range(0,5):
#            if (y[nnarray2[i][j]] == 1):
#                y[nnarray2[i][j]] = 0
#    df["label"] = y
#    return df

def generatedata(df,p):
    df_defect = df[df["label"] == 1]
    df_clean = df[df["label"] == 0]
    # print(df_defect.shape[0])
    n=int(df_defect.shape[0]*p/2)
    boot_defect_no= np.random.choice(df_defect.shape[0], n, replace=False)
    df_defect_select= df_defect.iloc[boot_defect_no]
    # oob_defect = [x for x in [i for i in range(0, df_defect.shape[0])] if x not in boot_defect_no]  # testing data
    # df_defect_left=df_defect.iloc[oob_defect]
    # df_defect_select['label'] = 0
    # for i in range(n):
    #     df_defect.iloc[boot_defect_no[i]]['label']=0
    #     print(df_defect.iloc[boot_defect_no[i]])
    # df_defect.iloc[boot_defect_no]['label']=0
    n_clean = int(df_clean.shape[0] * p)
    boot_clean_no = np.random.choice(df_clean.shape[0],n_clean, replace=False)
    df_clean_select = df_clean.iloc[boot_clean_no]
    oob = [x for x in [i for i in range(0, df_clean.shape[0])] if x not in boot_clean_no]  # testing data
    df_clean_left= df_clean.iloc[oob]
    df_clean_select['label']=1
    df_new=pd.DataFrame([])
    df_new=df_new.append(df_clean_left)
    df_new = df_new.append(df_clean_select)
    # df_new_defect=df_new[df_new['label']==1]
    # print(df_new_defect.shape[0])
    df_new=df_new.append(df_defect)
    # df_new = df_new.append(df_defect_select)
    # df_new_defect = df_new[df_new['label'] == 1]
    # print(df_new_defect.shape[0])
    # print(df_new['label'])
    # df_new=flipdata(df,df_defect_select,df_clean_select)
    return df_new

if __name__ == '__main__':
    # g = os.walk(r"E:/gln/C/myclassoverlap/generateoverlapratio/data/derby/train/")
    # i=0
    # for path, dir_list, file_list in g:
    #     for file_name in file_list:
    #         s = os.path.join(path, file_name)
            df = pd.read_csv("E:/gln/C/myclassoverlap/generateoverlapratio/data/prop1/train/prop_0.csv")
            p = [1,5,10,15,18]
            # p=[5,10,20,30,40,50,60,70,80]
            for k in range(9):
                for j in range(0,100):
                    df_generate = generatedata(df, p[k]/100)
                    df_new_defect = df_generate[df_generate['label'] == 1]
                    print(df_new_defect.shape[0])
                    s = "E:/gln/C/myclassoverlap/generateoverlapratio/data/overlapratio1/prop1/0/" + str(
                        p[k])+ "/prop_" + str(j) + ".csv"
                    df_generate.to_csv(s, index=False)







