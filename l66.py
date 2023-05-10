import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS',
#         'RAD','TAX','PTRATIO','B','LSTAT','MEDV']

# data = pd.read_csv(r'D:\Kursy\Machine Learning\Udemy-Mobilo\housing\housing.data',
#                    sep = ' +', engine = 'python', header= None,
#                    names = cols)

# data['Ones'] = 1

# X = data[['Ones']+cols[:-1]].values
# X

# y = data['MEDV'].values.reshape(-1,1)
# y

data = pd.read_csv(r"D:\Kursy\Machine Learning\Udemy-Mobilo\iris\iris.data", header = None)
 
data.insert(0,'Ones', value=1)
data[4] = data[4].apply(lambda x: 1 if x == 'Iris-setosa' else 
                        2 if x == 'Iris-versicolor' else 3)
data
 
X = data.iloc[:,:-1].values
X
y = data.iloc[:,-1].values.reshape(-1,1)
y

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

scaler = StandardScaler()
scaler.fit(y)
y = scaler.transform(y)

weights = np.ones((1,X.shape[1])).T/1000
weights

def predict(X, weights):
    predictions = np.dot(X,weights)
    return predictions.reshape(-1,1)

eta = 0.01
lmbda = 0.1
epochs = 100
N = X.shape[0]

for e in range(epochs):
    y_pred = predict(X,weights)
    error_pred = np.sum(np.square(y - y_pred)) + lmbda*np.sum(np.square(weights))
    delta_weight = np.zeros(weights.shape[0]).reshape(-1,1)
    
    for j in range(weights.shape[0]):
        
        lin_delta_weights = -2* np.sum(np.dot(X[:,j],(y-y_pred)))/N
        
        if j==0:
            weights[j] = weights[j] -eta*lin_delta_weights
        else:
            weights[j] = (1-2*eta*lmbda) * weights[j] - eta *lin_delta_weights
    # print('Epoka: ',e,'\nBlad: ', error_pred,'\n Suma kwadratow wag: ',np.square(weights).sum())

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)

from sklearn.linear_model import Ridge
ridge = Ridge(alpha = lmbda)
ridge.fit(X,y)

from sklearn.metrics import r2_score
r2_my_ridge = r2_score(y, predict(X,weights))
r2_lr = r2_score(y, lr.predict(X))
r2_ridge = r2_score(y, ridge.predict(X))

print('R2\t','My Ridge\t', r2_my_ridge, 
      '\tLINREG\t', r2_lr, '\tRIDGE\t', r2_ridge)
print('W \t','My Ridge\t', np.square(weights).sum(), 
      '\tLINREG\t', np.square(lr.coef_).sum(), 
      '\tRIDGE\t', np.square(ridge.coef_).sum())