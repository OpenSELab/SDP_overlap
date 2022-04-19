import numpy as np
import pandas as pd
from sklearn import metrics
import os
from sklearn.svm import SVC
from hyperopt import fmin, atpe, hp
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

if __name__ == '__main__':
    file = ['camel-1.2.csv','derby-10.2.1.6.csv','derby-10.3.1.4.csv','eclipse34_debug.csv','prop-1-44.csv','prop-1-92.csv','prop-1-164.csv','prop-2-256.csv','prop-3-318.csv','prop-4-355.csv','prop-5-4.csv','prop-5-40.csv','prop-5-85.csv','prop-5-121.csv','prop-5-157.csv','prop-5-185.csv','xalan-2.5.csv','xalan-2.6.csv']
    min_max_scaler_original = MinMaxScaler()
    min_max_scaler_KNN = MinMaxScaler()
    min_max_scaler_Kmeans = MinMaxScaler()
    min_max_scaler_SMR = MinMaxScaler()
    min_max_scaler_SVDD = MinMaxScaler()
    #min_max_scaler_grid = MinMaxScaler()
    min_max_scaler_combine = MinMaxScaler()
    for i in range(len(file)):
        s = 'E:/gln/C/myclassoverlap/Data/new/gooddata/data/' + file[i]
        g = os.walk(s)
        trials = Trials()

        auc_RF_original = np.zeros(shape=[100,1])
        recall_RF_original =np.zeros(shape=[100,1])
        brier_RF_original = np.zeros(shape=[100,1])
        pop_RF_original = np.zeros(shape=[100,1])
        pf_RF_original=np.zeros(shape=[100,1])

        auc_RF_KNN = np.zeros(shape=[100,1])
        recall_RF_KNN =np.zeros(shape=[100,1])
        brier_RF_KNN = np.zeros(shape=[100,1])
        pop_RF_KNN = np.zeros(shape=[100,1])
        pf_RF_KNN=np.zeros(shape=[100,1])

        auc_RF_Kmeans = np.zeros(shape=[100,1])
        recall_RF_Kmeans =np.zeros(shape=[100,1])
        brier_RF_Kmeans = np.zeros(shape=[100,1])
        pop_RF_Kmeans = np.zeros(shape=[100,1])
        pf_RF_Kmeans=np.zeros(shape=[100,1])

        auc_RF_SMR = np.zeros(shape=[100,1])
        recall_RF_SMR =np.zeros(shape=[100,1])
        brier_RF_SMR = np.zeros(shape=[100,1])
        pop_RF_SMR = np.zeros(shape=[100,1])
        pf_RF_SMR=np.zeros(shape=[100,1])

        auc_RF_SVDD = np.zeros(shape=[100, 1])
        recall_RF_SVDD = np.zeros(shape=[100, 1])
        brier_RF_SVDD = np.zeros(shape=[100, 1])
        pop_RF_SVDD = np.zeros(shape=[100, 1])
        pf_RF_SVDD = np.zeros(shape=[100, 1])

        # auc_RF_grid = np.zeros(shape=[100, 1])
        # recall_RF_grid = np.zeros(shape=[100, 1])
        # brier_RF_grid = np.zeros(shape=[100, 1])
        # pop_RF_grid = np.zeros(shape=[100, 1])
        # pf_RF_grid = np.zeros(shape=[100, 1])

        auc_RF_combine = np.zeros(shape=[100, 1])
        recall_RF_combine = np.zeros(shape=[100, 1])
        brier_RF_combine = np.zeros(shape=[100, 1])
        pop_RF_combine = np.zeros(shape=[100, 1])
        pf_RF_combine = np.zeros(shape=[100, 1])

        df = pd.read_csv(s)

        for t in range(100):
            boot = rand(df)
            train = df.iloc[boot]

            x_train_original = train.iloc[:, :train.shape[1] - 7].values
            print(x_train_original.shape[1])
            y_train_original = train.iloc[:, train.shape[1] - 6].values
     #       print(y_train_original)
            x_train_original = min_max_scaler_original.fit_transform(x_train_original)


            train_KNN=train[train['KNN_overlap']==0]
            x_train_KNN = train_KNN.iloc[:, :train_KNN.shape[1] - 7].values
            print(x_train_KNN.shape[1])
            y_train_KNN = train_KNN.iloc[:, train_KNN.shape[1] - 6].values
      #      print(y_train_KNN)
            x_train_KNN = min_max_scaler_KNN.fit_transform(x_train_KNN)

            oob = [x for x in [i for i in range(0, df.shape[0])] if x not in boot]  # testing data
            test = df.iloc[oob]
            n_test_clean=test[test['label']==0].shape[0]

            SL = test['loc'].values
            defnum = test['bugs'].values

            x_test = test.iloc[:, :test.shape[1] - 7].values
            y_test = test.iloc[:, test.shape[1] - 6].values
            x_test_original = min_max_scaler_original.transform(x_test)
            x_test_KNN = min_max_scaler_KNN.transform(x_test)

            train_Kmeans = train[train['Kmeans_overlap'] == 0]
            x_train_Kmeans = train_Kmeans.iloc[:, :train_Kmeans.shape[1] - 7].values
            y_train_Kmeans = train_Kmeans.iloc[:, train_Kmeans.shape[1] - 6].values
            #print(y_train_manhattan)
            x_train_Kmeans = min_max_scaler_Kmeans.fit_transform(x_train_Kmeans)
            x_test_Kmeans = min_max_scaler_Kmeans.transform(x_test)

            train_SMR = train[train['SMR_overlap'] == 0]
            x_train_SMR = train_SMR.iloc[:, :train_SMR.shape[1] - 7].values
            y_train_SMR = train_SMR.iloc[:, train_SMR.shape[1] - 6].values
            #print(y_train_SMR)
            x_train_SMR = min_max_scaler_SMR.fit_transform(x_train_SMR)
            x_test_SMR = min_max_scaler_SMR.transform(x_test)

            train_SVDD = train[train['SVDD_overlap'] == 0]
            x_train_SVDD = train_SVDD.iloc[:, :train_SVDD.shape[1] - 7].values
            y_train_SVDD = train_SVDD.iloc[:, train_SVDD.shape[1] - 6].values
       #     print(y_train_SVDD)
            x_train_SVDD = min_max_scaler_SVDD.fit_transform(x_train_SVDD)
            x_test_SVDD = min_max_scaler_SVDD.transform(x_test)

            train_combine = train[train['combine_overlap_label'] == 0]
            x_train_combine = train_combine.iloc[:, :train_combine.shape[1] - 7].values
            y_train_combine = train_combine.iloc[:, train_combine.shape[1] - 6].values
         #   print(y_train_combine)
            x_train_combine = min_max_scaler_combine.fit_transform(x_train_combine)
            x_test_combine = min_max_scaler_combine.transform(x_test)


            n_train_defect_KNN=train_KNN[train_KNN['label']==1].shape[0]
            n_train_defect_Kmeans = train_Kmeans[train_Kmeans['label'] == 1].shape[0]
            n_train_defect_SMR = train_SMR[train_SMR['label'] == 1].shape[0]
            n_train_defect_SVDD = train_SVDD[train_SVDD['label'] == 1].shape[0]
           # n_train_defect_grid = train_grid[train_grid['label'] == 1].shape[0]
            n_train_defect_combine = train_combine[train_combine['label'] == 1].shape[0]


            if(n_train_defect_KNN==0 or n_train_defect_Kmeans==0 or n_train_defect_SMR==0 or n_train_defect_SVDD==0 or n_train_defect_combine==0 or n_train_defect_KNN==1 or n_train_defect_Kmeans==1 or n_train_defect_SMR==1 or n_train_defect_SVDD==1 or n_train_defect_combine==1):
                continue


            def hyperopt_model_score_SVM(params):
                clf = SVC(**params)
                return cross_val_score(clf, x_train_original, y_train_original, scoring="roc_auc", cv=2).mean()


            space_SVM = {
                'C': hp.uniform('C', 0.1, 10),
                'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'rbf', 'sigmoid']),
                'gamma': hp.uniform('gamma', 0.001, 10)
            }


            def fn_SVM(params):
                acc = hyperopt_model_score_SVM(params)
                return -acc


            trials = Trials()

            best = fmin(
                fn=fn_SVM, space=space_SVM, algo=atpe.suggest, max_evals=100, trials=trials)
            print("Best: {}".format(best))
            if (best['kernel'] == 0):
                kernel = 'linear'
            else:
                if (best['kernel'] == 1):
                    kernel = 'sigmoid'
                else:
                    if (best['kernel'] == 2):
                        kernel = 'poly'
                    else:
                        kernel = 'rbf'
            #

            rf_original = SVC(C=best['C'], kernel=kernel, gamma=best['gamma'], probability=True)

            rf_original.fit(x_train_original , y_train_original )
            rf_original.predict(x_test_original )
            y_pred_rf_original  = rf_original .predict(x_test_original )
            y_prob_rf_original  = rf_original .predict_proba(x_test_original)
            auc_RF_original[t]=roc_auc_score(y_test, y_prob_rf_original[:, 1])
            recall_RF_original[t]=recall_score(y_test, y_pred_rf_original)
            brier_RF_original[t]=brier_score_loss(y_test, y_prob_rf_original[:,1])
            pop_RF_original[t]=CE_score(SL,defnum,y_prob_rf_original[:,1])
            k=0
            for p in range(test.shape[0]):
                if((y_test[p]==0) and (y_pred_rf_original[p]==1)):
                    k=k+1
            pf_RF_original[t]=k/n_test_clean


            def hyperopt_model_score_SVM_KNN(params):
                clf = SVC(**params)
                return cross_val_score(clf, x_train_original, y_train_original, scoring="roc_auc", cv=2).mean()


            space_SVM_KNN = {
                'C': hp.uniform('C', 0.1, 10),
                'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'rbf', 'sigmoid']),
                'gamma': hp.uniform('gamma', 0.001, 10)
            }


            def fn_SVM_KNN(params):
                acc = hyperopt_model_score_SVM_KNN(params)
                return -acc


            trials = Trials()

            best = fmin(
                fn=fn_SVM_KNN, space=space_SVM_KNN, algo=atpe.suggest, max_evals=100, trials=trials)
            print("Best: {}".format(best))
            if (best['kernel'] == 0):
                kernel = 'linear'
            else:
                if (best['kernel'] == 1):
                    kernel = 'sigmoid'
                else:
                    if (best['kernel'] == 2):
                        kernel = 'poly'
                    else:
                        kernel = 'rbf'
            #

            rf_KNN= SVC(C=best['C'], kernel=kernel, gamma=best['gamma'], probability=True)

            rf_KNN.fit(x_train_KNN, y_train_KNN)
            rf_KNN.predict(x_test_KNN)
            y_pred_rf_KNN = rf_KNN.predict(x_test_KNN)
            y_prob_rf_KNN = rf_KNN.predict_proba(x_test_KNN)
            auc_RF_KNN[t]=roc_auc_score(y_test, y_prob_rf_KNN[:, 1])
            recall_RF_KNN[t]=recall_score(y_test, y_pred_rf_KNN)
            brier_RF_KNN[t]=brier_score_loss(y_test, y_prob_rf_KNN[:,1])
            pop_RF_KNN[t]=CE_score(SL,defnum,y_prob_rf_KNN[:,1])
            k=0
            for p in range(test.shape[0]):
                if((y_test[p]==0) and (y_pred_rf_KNN[p]==1)):
                    k=k+1
            pf_RF_KNN[t]=k/n_test_clean


            def hyperopt_model_score_SVM_Kmeans(params):
                clf = SVC(**params)
                return cross_val_score(clf, x_train_original, y_train_original, scoring="roc_auc", cv=2).mean()


            space_SVM_Kmeans = {
                'C': hp.uniform('C', 0.1, 10),
                'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'rbf', 'sigmoid']),
                'gamma': hp.uniform('gamma', 0.001, 10)
            }


            def fn_SVM_Kmeans(params):
                acc = hyperopt_model_score_SVM_Kmeans(params)
                return -acc


            trials = Trials()

            best = fmin(
                fn=fn_SVM_Kmeans, space=space_SVM_Kmeans, algo=atpe.suggest, max_evals=100, trials=trials)
            print("Best: {}".format(best))
            if (best['kernel'] == 0):
                kernel = 'linear'
            else:
                if (best['kernel'] == 1):
                    kernel = 'sigmoid'
                else:
                    if (best['kernel'] == 2):
                        kernel = 'poly'
                    else:
                        kernel = 'rbf'
            #

            rf_Kmeans = SVC(C=best['C'], kernel=kernel, gamma=best['gamma'], probability=True)

            rf_Kmeans.fit(x_train_Kmeans, y_train_Kmeans)
            rf_Kmeans.predict(x_test_Kmeans)
            y_pred_rf_Kmeans = rf_Kmeans.predict(x_test_Kmeans)
            y_prob_rf_Kmeans = rf_Kmeans.predict_proba(x_test_Kmeans)
            auc_RF_Kmeans[t] = roc_auc_score(y_test, y_prob_rf_Kmeans[:, 1])
            recall_RF_Kmeans[t] = recall_score(y_test, y_pred_rf_Kmeans)
            brier_RF_Kmeans[t] = brier_score_loss(y_test, y_prob_rf_Kmeans[:, 1])
            pop_RF_Kmeans[t] = CE_score(SL, defnum, y_prob_rf_Kmeans[:, 1])
            k = 0
            for p in range(test.shape[0]):
                if ((y_test[p] == 0) and (y_pred_rf_Kmeans[p] == 1)):
                    k = k + 1
            pf_RF_Kmeans[t] = k / n_test_clean

            def hyperopt_model_score_SVM_SMR(params):
                clf = SVC(**params)
                return cross_val_score(clf, x_train_original, y_train_original, scoring="roc_auc", cv=2).mean()


            space_SVM_SMR= {
                'C': hp.uniform('C', 0.1, 10),
                'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'rbf', 'sigmoid']),
                'gamma': hp.uniform('gamma', 0.001, 10)
            }


            def fn_SVM_SMR(params):
                acc = hyperopt_model_score_SVM_SMR(params)
                return -acc


            trials = Trials()

            best = fmin(
                fn=fn_SVM_SMR, space=space_SVM_SMR, algo=atpe.suggest, max_evals=100, trials=trials)
            print("Best: {}".format(best))
            if (best['kernel'] == 0):
                kernel = 'linear'
            else:
                if (best['kernel'] == 1):
                    kernel = 'sigmoid'
                else:
                    if (best['kernel'] == 2):
                        kernel = 'poly'
                    else:
                        kernel = 'rbf'
            #

            rf_SMR= SVC(C=best['C'], kernel=kernel, gamma=best['gamma'], probability=True)

            rf_SMR.fit(x_train_SMR, y_train_SMR)
            rf_SMR.predict(x_test_SMR)
            y_pred_rf_SMR = rf_SMR.predict(x_test_SMR)
            y_prob_rf_SMR = rf_SMR.predict_proba(x_test_SMR)
            auc_RF_SMR[t] = roc_auc_score(y_test, y_prob_rf_SMR[:, 1])
            recall_RF_SMR[t] = recall_score(y_test, y_pred_rf_SMR)
            brier_RF_SMR[t] = brier_score_loss(y_test, y_prob_rf_SMR[:, 1])
            pop_RF_SMR[t] = CE_score(SL, defnum, y_prob_rf_SMR[:, 1])
            k = 0
            for p in range(test.shape[0]):
                if ((y_test[p] == 0) and (y_pred_rf_SMR[p] == 1)):
                    k = k + 1
            pf_RF_SMR[t] = k / n_test_clean


            def hyperopt_model_score_SVM_SVDD(params):
                clf = SVC(**params)
                return cross_val_score(clf, x_train_original, y_train_original, scoring="roc_auc", cv=2).mean()


            space_SVM_SVDD = {
                'C': hp.uniform('C', 0.1, 10),
                'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'rbf', 'sigmoid']),
                'gamma': hp.uniform('gamma', 0.001, 10)
            }


            def fn_SVM_SVDD(params):
                acc = hyperopt_model_score_SVM_SVDD(params)
                return -acc


            trials = Trials()

            best = fmin(
                fn=fn_SVM_SVDD, space=space_SVM_SVDD, algo=atpe.suggest, max_evals=100, trials=trials)
            print("Best: {}".format(best))
            if (best['kernel'] == 0):
                kernel = 'linear'
            else:
                if (best['kernel'] == 1):
                    kernel = 'sigmoid'
                else:
                    if (best['kernel'] == 2):
                        kernel = 'poly'
                    else:
                        kernel = 'rbf'
            #

            rf_SVDD = SVC(C=best['C'], kernel=kernel, gamma=best['gamma'], probability=True)

            rf_SVDD.fit(x_train_SVDD, y_train_SVDD)
            rf_SVDD.predict(x_test_SVDD)
            y_pred_rf_SVDD = rf_SVDD.predict(x_test_SVDD)
            y_prob_rf_SVDD = rf_SVDD.predict_proba(x_test_SVDD)
            auc_RF_SVDD[t] = roc_auc_score(y_test, y_prob_rf_SVDD[:, 1])
            recall_RF_SVDD[t] = recall_score(y_test, y_pred_rf_SVDD)
            brier_RF_SVDD[t] = brier_score_loss(y_test, y_prob_rf_SVDD[:, 1])
            pop_RF_SVDD[t] = CE_score(SL, defnum, y_prob_rf_SVDD[:, 1])
            k = 0
            for p in range(test.shape[0]):
                if ((y_test[p] == 0) and (y_pred_rf_SVDD[p] == 1)):
                    k = k + 1
            pf_RF_SVDD[t] = k / n_test_clean


            def hyperopt_model_score_SVM_combine(params):
                clf = SVC(**params)
                return cross_val_score(clf, x_train_original, y_train_original, scoring="roc_auc", cv=2).mean()


            space_SVM_combine= {
                'C': hp.uniform('C', 0.1, 10),
                'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'rbf', 'sigmoid']),
                'gamma': hp.uniform('gamma', 0.001, 10)
            }


            def fn_SVM_combine(params):
                acc = hyperopt_model_score_SVM_combine(params)
                return -acc


            trials = Trials()

            best = fmin(
                fn=fn_SVM_combine, space=space_SVM_combine, algo=atpe.suggest, max_evals=100, trials=trials)
            print("Best: {}".format(best))
            if (best['kernel'] == 0):
                kernel = 'linear'
            else:
                if (best['kernel'] == 1):
                    kernel = 'sigmoid'
                else:
                    if (best['kernel'] == 2):
                        kernel = 'poly'
                    else:
                        kernel = 'rbf'
            #

            rf_combine = SVC(C=best['C'], kernel=kernel, gamma=best['gamma'], probability=True)

            rf_combine.fit(x_train_combine, y_train_combine)
            rf_combine.predict(x_test_combine)
            y_pred_rf_combine = rf_combine.predict(x_test_combine)
            y_prob_rf_combine = rf_combine.predict_proba(x_test_combine)
            auc_RF_combine[t] = roc_auc_score(y_test, y_prob_rf_combine[:, 1])
            recall_RF_combine[t] = recall_score(y_test, y_pred_rf_combine)
            brier_RF_combine[t] = brier_score_loss(y_test, y_prob_rf_combine[:, 1])
            pop_RF_combine[t] = CE_score(SL, defnum, y_prob_rf_combine[:, 1])
            k = 0
            for p in range(test.shape[0]):
                if ((y_test[p] == 0) and (y_pred_rf_combine[p] == 1)):
                    k = k + 1
            pf_RF_combine[t] = k / n_test_clean

            print(pf_RF_original[t], pf_RF_KNN[t], pf_RF_Kmeans[t],pf_RF_SMR[t],pf_RF_SVDD[t],pf_RF_combine[t])


        data_RF = pd.DataFrame(
            np.column_stack((auc_RF_original, recall_RF_original, brier_RF_original, pop_RF_original, pf_RF_original, auc_RF_KNN, recall_RF_KNN, brier_RF_KNN, pop_RF_KNN, pf_RF_KNN,auc_RF_Kmeans, recall_RF_Kmeans, brier_RF_Kmeans, pop_RF_Kmeans, pf_RF_Kmeans,auc_RF_SMR, recall_RF_SMR, brier_RF_SMR, pop_RF_SMR, pf_RF_SMR,auc_RF_SVDD, recall_RF_SVDD, brier_RF_SVDD, pop_RF_SVDD, pf_RF_SVDD,auc_RF_combine, recall_RF_combine, brier_RF_combine, pop_RF_combine, pf_RF_combine)))#
        data_RF.columns = ['auc_RF_original', 'recall_RF_original', 'brier_RF_original', 'pop_RF_original', 'pf_RF_original', 'auc_RF_KNN', 'recall_RF_KNN', 'brier_RF_KNN', 'pop_RF_KNN', 'pf_RF_KNN','auc_RF_Kmeans', 'recall_RF_Kmeans', 'brier_RF_Kmeans', 'pop_RF_Kmeans', 'pf_RF_Kmeans','auc_RF_SMR', 'recall_RF_SMR', 'brier_RF_SMR', 'pop_RF_SMR', 'pf_RF_SMR','auc_RF_SVDD', 'recall_RF_SVDD', 'brier_RF_SVDD', 'pop_RF_SVDD', 'pf_RF_SVDD','auc_RF_combine', 'recall_RF_combine', 'brier_RF_combine', 'pop_RF_combine', 'pf_RF_combine']#,
        s3 = "E:/gln/C/myclassoverlap/rebuttal/results/compared/SVM/" + file[i]
        data_RF.to_csv(s3, index=False)







