import numpy as np
import pandas as pd
from sklearn import metrics
import os
from sklearn.tree import DecisionTreeClassifier
from hyperopt import fmin, atpe, hp
from hyperopt import Trials
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier

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
    file = ['prop-4-355.csv','prop-5-4.csv','prop-5-40.csv','prop-5-85.csv','prop-5-121.csv','prop-5-157.csv','prop-5-185.csv']
    min_max_scaler_original = MinMaxScaler()
    min_max_scaler_KNN = MinMaxScaler()
    min_max_scaler_lap = MinMaxScaler()

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

        auc_RF_Sep = np.zeros(shape=[100,1])
        recall_RF_Sep =np.zeros(shape=[100,1])
        brier_RF_Sep = np.zeros(shape=[100,1])
        pop_RF_Sep = np.zeros(shape=[100,1])
        pf_RF_Sep=np.zeros(shape=[100,1])


        df = pd.read_csv(s)

        for t in range(100):
            boot = rand(df)
            train = df.iloc[boot]

            x_train_original = train.iloc[:, :train.shape[1] - 7].values
            print(x_train_original.shape[1])
            y_train_original = train.iloc[:, train.shape[1] - 6].values
     #       print(y_train_original)
            x_train_original = min_max_scaler_original.fit_transform(x_train_original)
            y_train_sep = train.iloc[:, train.shape[1] - 5].values


            train_KNN=train[train['KNN_overlap']==0]
            x_train_KNN = train_KNN.iloc[:, :train_KNN.shape[1] - 7].values
      #      print(x_train_KNN.shape[1])
            y_train_KNN = train_KNN.iloc[:, train_KNN.shape[1] - 6].values
      #      print(y_train_KNN)
            x_train_KNN = min_max_scaler_KNN.fit_transform(x_train_KNN)


            train_lap = train[train["KNN_overlap"] == 1]
            x_train_lap = train_lap.iloc[:, :train_lap.shape[1] - 7].values
 #           print(x_train_KNN.shape[1])
            y_train_lap = train_lap.iloc[:, train_lap.shape[1] - 6].values
            x_train_lap = min_max_scaler_lap.fit_transform(x_train_lap)

            oob = [x for x in [i for i in range(0, df.shape[0])] if x not in boot]  # testing data
            test = df.iloc[oob]
            n_test_clean=test[test['label']==0].shape[0]

            SL = test['loc'].values
            defnum = test['bugs'].values

            x_test = test.iloc[:, :test.shape[1] - 7].values
            y_test = test.iloc[:, test.shape[1] - 6].values
            x_test_original = min_max_scaler_original.transform(x_test)
            x_test_KNN = min_max_scaler_KNN.transform(x_test)



            n_train_defect_KNN=train_KNN[train_KNN['label']==1].shape[0]


            if(n_train_defect_KNN==0 or n_train_defect_KNN==1):
                continue


            def hyperopt_model_score_DT(params):
                clf = DecisionTreeClassifier(**params)
                return cross_val_score(clf, x_train_original, y_train_original, scoring="roc_auc",cv=2).mean()

            n_feature=int(math.sqrt(train.shape[1]-7))
            #n=int(train.shape[0]/2)
            space_DT = {
                'max_depth': 1 + hp.randint('max_depth', 15),
                'min_samples_leaf': 1 + hp.randint("min_samples_leaf", 5),
                'criterion': hp.choice('criterion', ["gini", "entropy"])
            }


            def fn_DT(params):
                acc = hyperopt_model_score_DT(params)
                return -acc


            trials = Trials()

            best = fmin(
                fn=fn_DT, space=space_DT, algo=atpe.suggest, max_evals=100, trials=trials)
            print("Best: {}".format(best))
            if (best['criterion'] == 0):
                criterion = "gini"
            else:
                criterion = "entropy"

            best_max_depth = best['max_depth']
            best_min_samples_leaf= best['min_samples_leaf']



            rf_original =DecisionTreeClassifier(criterion=criterion, max_depth=best_max_depth,min_samples_leaf=best_min_samples_leaf)

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



            def hyperopt_model_score_DT_KNN(params):
                clf = DecisionTreeClassifier(**params)
                return cross_val_score(clf, x_train_KNN, y_train_KNN, scoring="roc_auc",cv=2).mean()


            space_DT_KNN = {
                'max_depth': 1 + hp.randint('max_depth', 15),
                'min_samples_leaf': 1 + hp.randint("min_samples_leaf", 5),
                'criterion': hp.choice('criterion', ["gini", "entropy"])
            }


            def fn_DT_KNN(params):
                acc = hyperopt_model_score_DT_KNN(params)
                return -acc


            trials = Trials()

            best = fmin(
                fn=fn_DT_KNN, space=space_DT_KNN, algo=atpe.suggest, max_evals=100, trials=trials)
            print("Best: {}".format(best))
            if (best['criterion'] == 0):
                criterion = "gini"
            else:
                criterion = "entropy"

            best_max_depth = best['max_depth']
            best_min_samples_leaf = best['min_samples_leaf']


            rf_KNN = DecisionTreeClassifier(criterion=criterion, max_depth=best_max_depth,min_samples_leaf=best_min_samples_leaf)

            rf_KNN.fit(x_train_KNN, y_train_KNN)
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


            def hyperopt_model_score_DT_lap(params):
                clf = RandomForestClassifier(**params, n_jobs=5)
                return cross_val_score(clf, x_train_original, y_train_original, scoring="roc_auc", cv=2).mean()



            space_DT_lap = {
                'max_depth': 1 + hp.randint('max_depth', 50),
                'min_samples_leaf': 1 + hp.randint("min_samples_leaf", 5),
                'criterion': hp.choice('criterion', ["gini", "entropy"])
            }


            def fn_DT_lap(params):
                acc = hyperopt_model_score_DT_lap(params)
                return -acc


            trials = Trials()

            best = fmin(
                fn=fn_DT_lap, space=space_DT_lap, algo=atpe.suggest, max_evals=100, trials=trials)
            print("Best: {}".format(best))
            if (best['criterion'] == 0):
                criterion = "gini"
            else:
                criterion = "entropy"

            best_max_depth = best['max_depth']
            best_min_samples_leaf = best['min_samples_leaf']
            best_n_estimators = best['n_estimators']

            rf_overlap = DecisionTreeClassifier(criterion=criterion, max_depth=best_max_depth,min_samples_leaf=best_min_samples_leaf)

            rf_overlap.fit(x_train_original, y_train_sep)
            y_pred_overkap = rf_overlap.predict(x_test)
            test["pre"] = y_pred_overkap

            test_overlap = test[test["pre"] == 1]
            test_nonoverlap = test[test["pre"] == 0]

            SL_overlap = test_overlap['loc'].values
            defnum_overlap = test_overlap['bugs'].values

            SL_nonoverlap = test_nonoverlap['loc'].values
            defnum_nonoverlap = test_nonoverlap['bugs'].values

            x_test_overlap = test_overlap.iloc[:, :test_overlap.shape[1] - 8].values
            y_test_overlap = test_overlap.iloc[:, test_overlap.shape[1] - 7].values
            x_test_nonoverlap = test_nonoverlap.iloc[:, :test_nonoverlap.shape[1] - 8].values
            y_test_nonoverlap = test_nonoverlap.iloc[:, test_nonoverlap.shape[1] - 7].values

            if (x_test_nonoverlap.shape[0] > 0):
                x_test_nonoverlap = min_max_scaler_KNN.transform(x_test_nonoverlap)


                prob_nonoverlap = rf_KNN.predict_proba(x_test_nonoverlap)
                y_pred_nonoverlap = rf_KNN.predict(x_test_nonoverlap)

            if (x_test_overlap.shape[0] > 0):
                x_test_overlap = min_max_scaler_lap.transform(x_test_overlap)


                def hyperopt_model_score_laplap(params):
                    clf = DecisionTreeClassifier(**params)
                    return cross_val_score(clf, x_train_lap, y_train_lap, scoring="roc_auc", cv=2).mean()



                space_RF = {
                    'max_depth': 1 + hp.randint('max_depth', 15),
                    'min_samples_leaf': 1 + hp.randint("min_samples_leaf", 5),
                    'criterion': hp.choice('criterion', ["gini", "entropy"])
                }


                def fn_RF_laplap(params):
                    acc = hyperopt_model_score_laplap(params)
                    return -acc


                trials = Trials()

                best = fmin(
                    fn=fn_RF_laplap, space=space_RF, algo=atpe.suggest, max_evals=100, trials=trials)
                print("Best: {}".format(best))
                if (best['criterion'] == 0):
                    criterion = "gini"
                else:
                    criterion = "entropy"

                best_max_depth = best['max_depth']
                best_min_samples_leaf = best['min_samples_leaf']



                rf_laplap = DecisionTreeClassifier(criterion=criterion, max_depth=best_max_depth,min_samples_leaf=best_min_samples_leaf)
                rf_laplap.fit(x_train_lap, y_train_lap)


                prob_step = rf_laplap.predict_proba(x_test_overlap)
                y_pred_step = rf_laplap.predict(x_test_overlap)
                prob_all = np.concatenate((prob_step, prob_nonoverlap), axis=0)
                y_pred_all = np.concatenate((y_pred_step, y_pred_nonoverlap), axis=0)
                y_test_all = np.concatenate((y_test_overlap, y_test_nonoverlap), axis=0)

                SL_all = np.concatenate((SL_overlap, SL_nonoverlap), axis=0)
                defnum_all = np.concatenate((defnum_overlap, defnum_nonoverlap), axis=0)
            else:
                prob_all = prob_nonoverlap
                y_pred_all = y_pred_nonoverlap
                y_test_all = y_test_nonoverlap
                SL_all = SL_nonoverlap
                defnum_all = defnum_nonoverlap
            print(len(prob_all))
            print(len(SL_all))
            print(len(defnum_all))
            pop_RF_Sep[t] = CE_score(SL_all, defnum_all, prob_all[:, 1])
            auc_RF_Sep[t] = roc_auc_score(y_test_all, prob_all[:, 1])
            recall_RF_Sep[t] = recall_score(y_test_all, y_pred_all)
            brier_RF_Sep[t] = brier_score_loss(y_test_all, prob_all[:, 1])
            k = 0
            for p in range(test.shape[0]):
                if ((y_test_all[p] == 0) and (y_pred_all[p] == 1)):
                    k = k + 1
            pf_RF_Sep[t] = k / n_test_clean



            print(pf_RF_original[t], pf_RF_KNN[t], pf_RF_Sep[t])


        data_RF = pd.DataFrame(
            np.column_stack((auc_RF_original, recall_RF_original, brier_RF_original, pop_RF_original, pf_RF_original, auc_RF_KNN, recall_RF_KNN, brier_RF_KNN, pop_RF_KNN, pf_RF_KNN,auc_RF_Sep, recall_RF_Sep, brier_RF_Sep, pop_RF_Sep, pf_RF_Sep)))#
        data_RF.columns = ['auc_RF_original', 'recall_RF_original', 'brier_RF_original', 'pop_RF_original', 'pf_RF_original', 'auc_RF_KNN', 'recall_RF_KNN', 'brier_RF_KNN', 'pop_RF_KNN', 'pf_RF_KNN','auc_RF_Sep', 'recall_RF_Sep', 'brier_RF_Sep', 'pop_RF_Sep', 'pf_RF_Sep']
        s3 = "E:/gln/C/myclassoverlap/rebuttal/results/performance/DT/" + file[i]
        data_RF.to_csv(s3, index=False)







