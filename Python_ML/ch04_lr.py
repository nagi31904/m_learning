#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 11:35:42 2017

@author: jerry
"""
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

iris = datasets.load_iris()

df_data = pd.DataFrame(iris['data'], columns=iris.feature_names)
df_target = pd.DataFrame(iris['target'], columns=['Target'])
df = pd.concat([df_data,df_target],axis=1)


reg = linear_model.LinearRegression()

iris_pl_re = df['petal length (cm)'].reshape(-1, 1)
iris_pw_re = df['petal width (cm)'].reshape(-1, 1)

reg.fit(iris_pl_re, iris_pw_re)
reg.score(iris_pl_re, iris_pw_re)

prediction_space = np.linspace(min(iris_pl_re),
                               max(iris_pl_re)).reshape(-1, 1)

plt.scatter(iris_pl_re, iris_pw_re, color = 'blue')
plt.plot(prediction_space, reg.predict(prediction_space),
         color='black', linewidth=2)

plt.show()



################All Features
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

iris = datasets.load_iris()

df_data = pd.DataFrame(iris['data'], columns=iris.feature_names)
df_target = pd.DataFrame(iris['target'], columns=['Target'])
df = pd.concat([df_data,df_target],axis=1)

X_train, X_test, y_train, y_test = train_test_split(df_data, df_target,
                                                    test_size = 0.3,
                                                    random_state=42)

reg_all = linear_model.LinearRegression()
reg_all.fit(X_train, y_train)


y_pred = reg_all.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print('R^2', reg_all.score(X_test, y_test))
print('RMSE: \n', rmse)

