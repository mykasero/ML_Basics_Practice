import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
        
cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS',
        'RAD','TAX','PTRATIO','B','LSTAT','MEDV']

data = pd.read_csv(r'D:\Kursy\Machine Learning\Udemy-Mobilo\housing\housing.data',
                   sep= ' +', engine='python', header = None,
                   names = cols)

X = data.drop('MEDV', axis = 1)
y = data['MEDV'].values.reshape(-1,1)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

scaler = StandardScaler()
scaler.fit(y)
y = scaler.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

lasso_df = pd.DataFrame({'param_value': np.arange(start=0, stop = 10.1, step=0.1),
                         'r2_result': 0.,
                         'number_of_features':0})

for i in range(len(lasso_df)):
    alpha = lasso_df.at[i,'param_value']
    lasso = Lasso(alpha= alpha)
    lasso.fit(X_train,y_train)
    lasso_df.at[i,'r2_result'] = (r2_score(y_test,lasso.predict(X_test)))
    lasso_df.at[i,'number_of_features']= len(lasso.coef_[lasso.coef_>0])
    
ridge_df = pd.DataFrame({'param_value': np.arange(start=0,stop=10.1,step=0.1),
                         'r2_result':0.,
                         'number_of_features':0})
for i in range(ridge_df.shape[0]):
    alpha = ridge_df.at[i,'param_value']
    ridge = Ridge(alpha = alpha)
    ridge.fit(X_train,y_train)
    ridge_df.at[i,'r2_result'] = (r2_score(y_test,ridge.predict(X_test)))
    ridge_df.at[i,'number_of_features'] = len(ridge.coef_[ridge.coef_ > 0])
    
elastic_df = pd.DataFrame({'param_value': np.arange(start=0,stop=10.1,step=0.1),
                         'r2_result':0.,
                         'number_of_features':0})

for i in range(len(elastic_df)):
    alpha = elastic_df.at[i,'param_value']
    elastic = ElasticNet(alpha=alpha, l1_ratio = 0.5)
    elastic.fit(X_train,y_train)
    elastic_df.at[i,'r2_result'] = (r2_score(y_test,elastic.predict(X_test)))
    elastic_df.at[i,'number_of_features'] = len(elastic.coef_[elastic.coef_ > 0])
    
fig, axs = plt.subplots(3, figsize=(10,10))
axs[0].title.set_text('Lasso')
axs[0].scatter(x = lasso_df['param_value'], y=lasso_df['r2_result']*10,color='green')
axs[0].scatter(x = lasso_df['param_value'], y=lasso_df['number_of_features'])
axs[0].legend(loc='center right')
axs[1].title.set_text('Ridge')
axs[1].scatter(x = ridge_df['param_value'], y=ridge_df['r2_result']*10,color='green')
axs[1].scatter(x = ridge_df['param_value'], y=ridge_df['number_of_features'])
axs[1].legend(loc='center right')
axs[2].title.set_text('Elastic Net')
axs[2].scatter(x = elastic_df['param_value'], y=elastic_df['r2_result']*10,color='green')
axs[2].scatter(x = elastic_df['param_value'], y=elastic_df['number_of_features'])
axs[2].legend(loc='center right')
fig.show()