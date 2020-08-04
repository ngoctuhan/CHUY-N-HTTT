import pandas as pd
import numpy as np
from ngoctuhan.tree_optimizer_memory import DecisionTreeClassifier
df =  pd.read_csv('datatest.csv')

X_train =  df.iloc[:,1:-1].values
y_train = df.iloc[:,-1].values

model =  DecisionTreeClassifier()

model.fit(X_train, y_train)


# model.find_best_value(X_train[:, 0], y_train)
for i in range(X_train.shape[0]):
    print(model.predict(X_train[i, :]))