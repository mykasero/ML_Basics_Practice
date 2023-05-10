import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS',
        'RAD','TAX','PTRATIO','B','LSTAT','MEDV']

data = pd.read_csv(r'D:\Kursy\Machine Learning\Udemy-Mobilo\housing\housing.data',
                   sep = ' +', engine = 'python', header = None,
                   names = cols)

# data = data[['LSTAT','MEDV']]

# Q1 = data.quantile(0.25)
# Q3 = data.quantile(0.75)

# IQR = Q3 - Q1

# outlier_cond = ((data < Q1 - 1.5*IQR) | (data > Q3 +1.5 *IQR))

# data = data[~outlier_cond.any(axis=1)]

X = data[cols[:-1]]
y = data['MEDV'].values.reshape(-1,1)

# plt.scatter(X,y)
# plt.show

# scaler = StandardScaler()
# scaler.fit(X)
# scaler.transform(X)

# scaler = StandardScaler()
# scaler.fit(y)
# scaler.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

lr = LinearRegression()
lr.fit(X_train,y_train)

MAE_train = mean_absolute_error(y_train, lr.predict(X_train))
MAE_train

MAE_test = mean_absolute_error(y_test, lr.predict(X_test))
MAE_test

MSE_train = mean_squared_error(y_train, lr.predict(X_train))
MSE_train
                               
MSE_test = mean_squared_error(y_test, lr.predict(X_test))
MSE_test

R2_train = r2_score(y_train, lr.predict(X_train))
R2_train

R2_test = r2_score(y_test, lr.predict(X_test))
R2_test
