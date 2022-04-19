from eli5.sklearn import PermutationImportance
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import multiprocessing as mp
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from hyperopt import fmin, atpe, hp
from hyperopt import Trials
import math
from sklearn.model_selection import cross_val_score

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

def PermutationImportance_(clf, X, y):
    perm = PermutationImportance(clf, n_iter=5, random_state=1024, cv=5)

    perm.fit(X, y)

    # result_ = {'var': var
    #     , 'feature_importances_': perm.feature_importances_
    #     , 'feature_importances_std_': perm.feature_importances_std_}
    # feature_importances_ = pd.DataFrame(result_, columns=['var', 'feature_importances_', 'feature_importances_std_'])
    # feature_importances_ = feature_importances_.sort_values('feature_importances_', ascending=False)
    return perm.feature_importances_


def fun(file):
    df = pd.read_csv('C:/gln/myclassoverlap/Data/new/gooddata/newnew/KNN/'+file)


    feature_original=[]
    feature_remove=[]
    feature_separating = []

    scaler = MinMaxScaler()
    scaler1 = MinMaxScaler()

    for t in range(100):
        boot = rand(df)
        train = df.iloc[boot]
        oob = [x for x in [i for i in range(0, df.shape[0])] if x not in boot]  # testing data
        test = df.iloc[oob]
        train_remove = train[train["laplabel"] == 0]


        x_train = train.iloc[:, :train.shape[1] - 4]
        y_train = train.iloc[:, train.shape[1] - 3].values
        var=x_train.columns.values

        x_train_remove = train_remove.iloc[:, :train_remove.shape[1] - 4]
        y_train_remove = train_remove.iloc[:, train_remove.shape[1] - 3].values


        x_train = scaler.fit_transform(x_train)
        x_train_remove = scaler1.fit_transform(x_train_remove)

        x_test = test.iloc[:, :test.shape[1] - 4].values
        y_test = test.iloc[:, test.shape[1] - 3].values

        def hyperopt_model_score_KNN(params):
            clf = KNeighborsClassifier(**params)
            return cross_val_score(clf, x_train,y_train, scoring="roc_auc", cv=2).mean()

        space_KNN = {
            'n_neighbors': 1 + hp.randint('n_neighbors', 20)
        }

        def fn_KNN(params):
            acc = hyperopt_model_score_KNN(params)
            return -acc

        trials = Trials()

        best = fmin(
            fn=fn_KNN, space=space_KNN, algo=atpe.suggest, max_evals=100, trials=trials)
        print("Best: {}".format(best))

        best_n_neighbors = best['n_neighbors']

        clf = KNeighborsClassifier(n_neighbors=best_n_neighbors)
        clf.fit(x_train,y_train)
        x_test1 = scaler.transform(x_test)
        feature_importances_1= PermutationImportance_(clf, x_test1, y_test)
        feature_original.append(feature_importances_1)
        # print(feature_original.shape)

        print(feature_importances_1)

        def hyperopt_model_score_KNN_remove(params):
            clf = KNeighborsClassifier(**params)
            return cross_val_score(clf, x_train_remove,y_train_remove, scoring="roc_auc", cv=2).mean()

        space_KNN_remove = {
            'n_neighbors': 1 + hp.randint('n_neighbors', 20)
        }

        def fn_KNN_remove(params):
            acc = hyperopt_model_score_KNN_remove(params)
            return -acc

        trials = Trials()

        best = fmin(
            fn=fn_KNN_remove, space=space_KNN_remove, algo=atpe.suggest, max_evals=100, trials=trials)
        print("Best: {}".format(best))

        best_n_neighbors = best['n_neighbors']

        clf1 = KNeighborsClassifier(n_neighbors=best_n_neighbors)
        clf1.fit(x_train_remove,y_train_remove)
        x_test2 = scaler1.transform(x_test)
        feature_importances_2= PermutationImportance_(clf1, x_test2, y_test)
        print(feature_importances_2)
        feature_remove.append(feature_importances_2)

    feature_original=pd.DataFrame(feature_original)
    print(feature_original.shape)
    feature_original.columns=var
    feature_remove=pd.DataFrame(feature_remove)
    feature_remove.columns=var
    feature_separating=pd.DataFrame(feature_separating)
    feature_separating.columns=var
    ss='C:/gln/myclassoverlap/results/featureimportant/new/KNN/original/original_'+file
    feature_original.to_csv(ss,index=False)
    ss1='C:/gln/myclassoverlap/results/featureimportant/new/KNN/remove/remove_'+file
    feature_remove.to_csv(ss1,index=False)

if __name__ == '__main__':
    N =4
    g1 = os.scandir(r"C:/gln/myclassoverlap/Data/new/gooddata/newnew/KNN/")

    with mp.Pool(processes=N) as p:
        results = p.map(fun, [file.name for file in g1])
