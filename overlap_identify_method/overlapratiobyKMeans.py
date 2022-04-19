import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn import preprocessing
import os
def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def count(df, label):
    nlist = np.zeros(shape=[len(label), 1], dtype=np.float64)
    for i in range(df.shape[0]):
        for j in range(len(label)):
            if (df.iloc[i, df.shape[1] - 1] == label[j]):
                nlist[j] = nlist[j] + 1
    return nlist

def calcluster(df):
    number = df.shape[0]
    p=df.iloc[:,df.shape[1]-1].values*100

    if number<=20:
        return df

    # print(df.shape[0])
    k = int(df.shape[0] / 20 + 1)

    # x = df.iloc[:, :df.shape[1] - 2]
    x = df.iloc[:, 1:df.shape[1]-7]
    x = preprocessing.MinMaxScaler().fit_transform(x)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(x)
    list=kmeans.labels_
    for q in range(len(list)):
        list[q]=list[q]+1
    df["clusterlabel"] = list + p
    # print(df["clusterlabel"])
    label = df["clusterlabel"].unique()
    number1 = count(df, label)
    # print(label)
    # print(number1)
    # print(len(number1))
    # print(k)
    # print(number)
    for i in range(0,len(number1)):
        # print(i)
        if(number1[i]==number):
            break
        else:
            df[df["clusterlabel"] == label[i]] = calcluster(df[df["clusterlabel"] == label[i]])
    return df

if __name__ == '__main__':
    g = os.walk(r"E:/gln/C/myclassoverlap/Data/new/gooddata/data1/")
    for path, dir_list, file_list in g:
        for file_name in file_list:
            # Name.append(file_name)
            print(file_name)
            s = os.path.join(path, file_name)
            df = pd.read_csv(s)
            p_banlance=df[df['label']==1].shape[0]/df[df['label']==0].shape[0]
            print(df.shape)
            n_label = np.zeros(shape=[df.shape[0], 1], dtype=np.uint64)
            #s1='E:/gln/C/myclassoverlap/Data/new/all/'+file_name
            #df1=pd.read_csv(s1)
            #df['loc']=df1['loc']
            df['clusterlabel'] = n_label
            df = calcluster(df)
            print(df.shape)
            df_new=pd.DataFrame([])

            label = df["clusterlabel"].unique()
            #print(label)
            number1 = 0
            number2=0

            for i in range(len(label)):
                clus = df[df["clusterlabel"] == label[i]]
                print(clus.shape)
                defec1 = clus[clus["label"] == 1]
                print(defec1.shape)
                clean1 = clus[clus["label"] == 0]
                print(clean1.shape)

                if(clus[clus["label"]==1].shape[0]==0 or clus[clus["label"]==0].shape[0]==0):
                    clus["laplabel"]=np.zeros(shape=[clus.shape[0], 1])
                    lab = np.zeros(shape=[defec1.shape[0], 1])
                    defec1["removelabel"] = lab
                    lab1 = np.zeros(shape=[clean1.shape[0], 1])
                    clean1["removelabel"] = lab1
                else:
                    clus["laplabel"] = np.ones(shape=[clus.shape[0], 1])
                    p = defec1.shape[0] / clean1.shape[0]
                    if(p>=p_banlance):
                        lab = np.zeros(shape=[defec1.shape[0], 1])
                        defec1["removelabel"] = lab
                        lab1 = np.ones(shape=[clean1.shape[0], 1])
                        clean1["removelabel"] = lab1
                    else:
                        lab = np.ones(shape=[defec1.shape[0], 1])
                        defec1["removelabel"] = lab
                        lab1 = np.zeros(shape=[clean1.shape[0], 1])
                        clean1["removelabel"] = lab1
                    # else:
                    #     lab = np.zeros(shape=[defec1.shape[0], 1])
                    #     defec1["removelabel"] = lab
                    #     lab1 = np.ones(shape=[clean1.shape[0], 1])
                    #     clean1["removelabel"] = lab1

                df_new = df_new.append(defec1, ignore_index=True)
                print(df_new.shape)
                df_new=df_new.append(clean1, ignore_index=True)
                print(df_new.shape)
                df_new.sort_values("index", inplace=True)


            ss = 'E:/gln/C/myclassoverlap/Data/new/gooddata/data1/' + file_name
            df_new.to_csv(ss, index=False)




