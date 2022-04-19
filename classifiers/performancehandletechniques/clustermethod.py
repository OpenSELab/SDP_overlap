import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
import os
from sklearn.ensemble import RandomForestClassifier
from hyperopt import fmin, tpe, hp
from hyperopt import Trials
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import recall_score

def CE_score(SL, defnum,prob):
    df = pd.DataFrame(np.column_stack((SL, defnum, prob)))
    df.columns = ['CountLineCode','bugs', 'prob']
    df["den"] = df["bugs"] / df["CountLineCode"]


    b1 = df.sort_values(by=['prob', 'CountLineCode'], ascending=[False,True])
    op = df.sort_values(by=['den', 'CountLineCode'], ascending=[False,True])

    b = df.sort_values(by=['den', 'CountLineCode'], ascending=[True,False])

    sm = df['CountLineCode'].sum()
    count = df['bugs'].sum()
    print(count)

    x = []
    y = []
    line = 0
    defec = 0
    for i in range(df.shape[0]):
        line = line + b.iloc[i, 0] / sm
        defec = defec + b.iloc[i, 1] / count
        x.append(line)
        y.append(defec)

    x1 = []
    y1 = []
    line = 0
    defec = 0
    for i in range(df.shape[0]):
        line = line + b1.iloc[i, 0] / sm
        defec = defec + b1.iloc[i, 1] / count
        x1.append(line)
        y1.append(defec)

    x_op = []
    y_op = []
    line = 0
    defec = 0
    for i in range(df.shape[0]):
        line = line + op.iloc[i, 0] / sm
        defec = defec + op.iloc[i, 1] / count
        x_op.append(line)
        y_op.append(defec)

    are_m = np.trapz(y1, x1)
    are_op = np.trapz(y_op, x_op)
    are_rom = np.trapz(y, x)
    ce = (are_m - are_rom) / (are_op - are_rom)
    print(ce)

    return ce

def rand(df):
    boot = np.random.choice(df.shape[0], df.shape[0], replace=True)
    oob = [x for x in [i for i in range(0, df.shape[0])] if x not in boot]
    df1=df.iloc[boot]
    df2=df.iloc[oob]

    defe = df1[df1["label"] == 1]
    clean = df1[df1["label"] == 0]

    defe1 = df2[df2["label"] == 1]
    clean1 = df2[df2["label"] == 0]

    if (defe.shape[0]!=0 and clean.shape[0]!=0 and defe1.shape[0]!=0 and clean1.shape[0]!=0):
        return boot
    else:
        rand(df)

    return boot

def calmedian(x):
    featuremedian=np.zeros(shape=[x.shape[1],1])
    kmedian=np.zeros(shape=[x.shape[0],1])
    for j in range(x.shape[1]):
        featuremedian[j]=x.iloc[:,j].median()
    for i in range(x.shape[0]):
        m=0
        for j in range(x.shape[1]):
            if(x.iloc[i,j]>featuremedian[j]):
                m=m+1
        kmedian[i]=m
    #print(featuremedian,kmedian)
    #kmedian1 = np.array(kmedian)
    cluster = np.unique(kmedian)
    cluster = np.sort(cluster)
    k = int(len(cluster) / 2)
    thred = cluster[k]
   # print(thred)
    y_pred=np.zeros(shape=[x.shape[0],1])
    #print(thred)
    for i in range(x.shape[0]):
        if (kmedian[i] >= thred):
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return featuremedian,kmedian,y_pred
def calfeature(x):
    features=x.columns
    featuremedian, kmedian, y_pred=calmedian(x)
  #  print(y_pred)
    selectfeature=[]
    prop=[]
    for j in range(x.shape[1]):
        m=0
        for i in range(0,x.shape[0]):
            if(y_pred[i]==0):
                if(x.iloc[i,j]>featuremedian[j]):
                    m = m + 1
            if(y_pred[i]==1):
                if(x.iloc[i,j]<featuremedian[j]):
                    m = m + 1

        prop.append(m/x.shape[0])
 #   print(prop)
    minprop=min(prop)
    for j in range(0,x.shape[1]):
        if(prop[j]==minprop):
            selectfeature.append(j)
    #print(selectfeature)
    selectfeatures=[]
    for t in range(len(selectfeature)):
        selectfeatures.append(features[selectfeature[t]])
   # print(selectfeatures)
    x_feature = x.loc[:, selectfeatures]
    x_select=pd.DataFrame([])
    y_test=[]
    for i in range(x_feature.shape[0]):
        m=0
        for j in range(0,x_feature.shape[1]):
           # print(featuremedian[selectfeature[j]])
            if (y_pred[i] == 0):
                if (x_feature.iloc[i, j] > featuremedian[selectfeature[j]]):
                    m=m+1
            if (y_pred[i] == 1):
                if (x_feature.iloc[i, j]<featuremedian[selectfeature[j]]):
                   m=m+1
        if(m==0):
            x_select=x_select.append(x_feature.iloc[i,:])
            y_test.append(y_pred[i])
   # print(x_select)
    return x_select,y_test,selectfeatures

if __name__ == '__main__':
    file = ['eclipse34_debug.csv','prop-1-44.csv','prop-1-92.csv','prop-1-164.csv','prop-2-256.csv','prop-3-318.csv','prop-4-355.csv','prop-5-4.csv','prop-5-40.csv','prop-5-85.csv','prop-5-121.csv','prop-5-157.csv','prop-5-185.csv','xalan-2.5.csv','xalan-2.6.csv']

    for i in range(len(file)):
        s = 'E:/gln/C/myclassoverlap/Data/new/all/' + file[i]
        g = os.walk(s)
        trials = Trials()

        auc_cla = np.zeros(shape=[100,1])
        recall_cla =np.zeros(shape=[100,1])
        brier_cla = np.zeros(shape=[100,1])
        pop_cla = np.zeros(shape=[100,1])
        pf_cla=np.zeros(shape=[100,1])

        auc_RF_clmi = np.zeros(shape=[100,1])
        recall_RF_clmi =np.zeros(shape=[100,1])
        brier_RF_clmi = np.zeros(shape=[100,1])
        pop_RF_clmi = np.zeros(shape=[100,1])
        pf_RF_clmi=np.zeros(shape=[100,1])

        df = pd.read_csv(s)

        for t in range(100):
            boot = rand(df)
            train = df.iloc[boot]


            x_train_clmi = train.iloc[:, :train.shape[1] - 2]



            oob = [x for x in [i for i in range(0, df.shape[0])] if x not in boot]  # testing data
            test = df.iloc[oob]
            n_test_clean=test[test['label']==0].shape[0]

            SL = test['loc'].values
            defnum = test['bugs'].values
            x_test_cla=test.iloc[:, :test.shape[1] - 2]

            x_test = test.iloc[:, :test.shape[1] - 2].values
            y_test = test.iloc[:, test.shape[1] - 1].values


            _,kmedian,y_pred=calmedian(x_test_cla)

            auc_cla[t] = roc_auc_score(y_test, y_pred)
            recall_cla[t] = recall_score(y_test, y_pred)
            brier_cla[t] = brier_score_loss(y_test, y_pred)
            pop_cla[t] = CE_score(SL, defnum, kmedian)
            k = 0
            for p in range(test.shape[0]):
                if ((y_test[p] == 0) and (y_pred[p] == 1)):
                    k = k + 1
            pf_cla[t] = k / n_test_clean

            x_select,y_train,selectfeatures=calfeature(x_train_clmi)
            x_test=x_test_cla.loc[:,selectfeatures].values


           #  def hyperopt_model_score_RF(params):
           #      clf = RandomForestClassifier(**params, n_jobs=5)
           #      return cross_val_score(clf, x_select, y_train, scoring="roc_auc",cv=2).mean()
           #
           #  n_feature=int(math.sqrt(x_select.shape[1]))+1
           #  print(n_feature)
           #  n_es=int(x_select.shape[0]/2)
           #  #n=int(train.shape[0]/2)
           #  space_RF = {
           #      'max_depth': 1 + hp.randint('max_depth', 10),
           #      #'max_features': 1 + hp.randint("max_features", n_feature),
           #      'n_estimators': 50+  hp.randint('n_estimators', n_es),
           #      'criterion': hp.choice('criterion', ["gini", "entropy"])
           #  }
           #
           #
           #  def fn_RF(params):
           #      acc = hyperopt_model_score_RF(params)
           #      return -acc
           #
           #
           #  trials = Trials()
           #
           #  best = fmin(
           #      fn=fn_RF, space=space_RF, algo=tpe.suggest, max_evals=10, trials=trials)
           #  print("Best: {}".format(best))
           #  if (best['criterion'] == 0):
           #      criterion = "gini"
           #  else:
           #      criterion = "entropy"
           #
           #  best_max_depth = best['max_depth']
           # # best_max_features = best['max_features']
           #  best_n_estimators = best['n_estimators']
           #  if (best['max_depth'] == 0):
           #      best_max_depth=best['max_depth']+1
           # # if (best['max_features'] == 0):
           # #     best_max_features = best['max_features'] + 1
           #  if (best['n_estimators'] == 0):
           #      best_n_estimators = best['n_estimators'] +1

            rf_euclidean = RandomForestClassifier()#criterion=criterion, max_depth=best_max_depth,max_features=x_select.shape[1], n_estimators=best_n_estimators

            rf_euclidean.fit(x_select, y_train)
            rf_euclidean.predict(x_test)
            y_pred_rf_clmi = rf_euclidean.predict(x_test)
            y_prob_rf_clmi = rf_euclidean.predict_proba(x_test)
            auc_RF_clmi[t]=roc_auc_score(y_test, y_prob_rf_clmi[:, 1])
            recall_RF_clmi[t]=recall_score(y_test, y_pred_rf_clmi)
            brier_RF_clmi[t]=brier_score_loss(y_test, y_prob_rf_clmi[:,1])
            pop_RF_clmi[t]=CE_score(SL,defnum,y_prob_rf_clmi[:,1])
            k=0
            for p in range(test.shape[0]):
                if((y_test[p]==0) and (y_pred_rf_clmi[p]==1)):
                    k=k+1
            pf_RF_clmi[t]=k/n_test_clean




        data_RF = pd.DataFrame(
            np.column_stack((auc_cla, recall_cla, brier_cla, pop_cla, pf_cla, auc_RF_clmi, recall_RF_clmi, brier_RF_clmi, pop_RF_clmi, pf_RF_clmi)))
        data_RF.columns = ['AUC_cla', 'Recall_cla', 'Brier_cla', 'Pop_cla', 'pf_cla', 'AUC_clmi', 'Recall_clmi', 'Brier_clmi', 'Pop_clmi', 'pf_clmi']
        s3 = "E:/gln/C/myclassoverlap/rebuttal/results/performance/cluster/" + file[i] + ".csv"
        data_RF.to_csv(s3, index=False)
