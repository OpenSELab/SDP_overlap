import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df = pd.read_csv("E:/gln/C/myclassoverlap/Data/new/all/camel-1.2.csv")
    for i in range(100):
        train, test = train_test_split(df, test_size=0.3)
        print(train.shape)
        s_train="E:/gln/C/myclassoverlap/generateoverlapratio/data/camel/train/camel_"+str(i)+".csv"
        s_test="E:/gln/C/myclassoverlap/generateoverlapratio/data/camel/test/camel_"+str(i)+".csv"
        train.to_csv(s_train, index=False)
        test.to_csv(s_test, index=False)

