import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
#from scipy import stats
 
cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS',
        'RAD','TAX','PTRATIO','B','LSTAT','MEDV']
 
data = pd.read_csv(r"D:\Kursy\Machine Learning\Udemy-Mobilo\housing\housing.data",
                   sep=' +', engine='python', header=None, 
                   names=cols)
 
data = data.loc[:,['LSTAT','MEDV']]

# Q1 = data.quantile(0.25)
# Q3 = data.quantile(0.75)

# IQR = Q3-Q1

# outlier_cond = ((data < Q1 - 1.5 *IQR) | (data > Q3 +1.5*IQR))

# data = data[~outlier_cond.any(axis=1)]

lr = LinearRegression()
X = np.array(data['LSTAT'].values.reshape(-1,1))
y = data['MEDV'].values.reshape(-1,1)

scaler = StandardScaler()
scaler.fit(X)
scaler.transform(X)

scaler = StandardScaler()
scaler.fit(y)
scaler.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.3)

lr.fit(X_train,y_train)

y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

fig,ax = plt.subplots(2,2,figsize=(8,8))
ax[0,0].scatter(X_train,y_train,color='gold')
ax[0,0].plot(X_train,y_pred_train, color = 'blue')

ax[0,1].scatter(X_test, y_test, color = 'gold')
ax[0,1].plot(X_test, y_pred_test, color = 'blue')
 
 
ax[1,0].scatter(y_train, y_pred_train - y_train, s=80, 
          facecolors='none', edgecolors='b')
 
ax[1,1].scatter(y_test,  y_pred_test  - y_test,  s=80, 
          facecolors='none', edgecolors='r')
fig.show()