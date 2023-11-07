# importing libraries
import os
import datetime
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

#importing dataset
train = pd.read_csv(r'D:\bee\diabetes ml\train.csv')
test = pd.read_csv(r'D:\bee\diabetes ml\test.csv')
train.head()

train.info()
test.info()

train.describe()

sns.boxplot(x='count',data=train,color='mediumpurple')
plt.show()

# histogram of count
sns.set_style('darkgrid')
sns.distplot(train['train'],bins=100,color='green')
plt.show()

#scatter plot between count and numeric features
fields = [f for f in train]
fields=fields[5:-3]
print(fields)

fig=plt.figure(figsize=(17,3))

for i,f in enumerate(fields):
   ax=fig.sub_plot(1,4,i+1)
   ax=scatter(train[f],train[count])
   ax.set_ylabel('count')
   ax.set_xlabel('f')
   
plt.show()

#boxplot between count and each categorical feature
fig.axes=plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(20,10)

sns.boxplot(data='train',y='count',x='season',ax=axes[0][0])
sns.boxplot(data='train',y='count',x='holiday',ax=axes[0][1])
sns.boxplot(data='train',y='count',x='working day',ax=axes[1][0])
sns.boxplot(data='train',y='count',x='weather',ax=axes[1][1])

axes[0][0].set(xlabel='season',ylabel='count')
axes[0][1].set(xlabel='holiday',ylabel='count')
axes[1][0].set(xlabel='working day',ylabel='count')
axes[1][1].set(xlabel='weather',ylabel='count')


#correlation between each features
plt.figure(figsize=(10,10))
sns.heatmap(train.corr('pearson'),vmin=-1,vmax=1,cmap='coolwarm',annot=True,square=True)

#convert datetime column to each elements
train['datetime']=pd.to_datetime(train['datetime'])
test['datetime']=pd.to_datetime(test['datetime'])
train.head()

def split_datetime(df):
    df['year']=df['datetime'].apply(lambda t:t.year)
    df['month']=df['datetime'].apply(lambda t:t.month)
    df['day']=df['datetime'].apply(lambda t:t.day)
    df['dayofweek']=df['datetime'].apply(lambda t:t.dayofweek)
    df['hour']=df['datetime'].apply(lambda t:t.hour)
    df=df.drop(['datetime',axis=1])
    return df

train=split_datetime(train)
test=split_datetime(test)
train=train.drop(['casual','registered'])
train.head()


#boxplot between count and each categorical feature
fig,axes=plt.subplots(nrows=1,ncols=3)
fig.set_size_inches(25,5)
sns.barplot(data='train',x='year',y='count',ax=axes[0])
sns.barplot(data='train',x='month',y='count',ax=axes[1])
sns.pointplot(data='train',x='hour',y='count',ax=axes[2],hue='dayofweek')

#count column looks skew
sns.displot(train['count'])

#take a log for count column
train['count']=np.log1p(train['count'])
sns.displot(train['count'])

#eliminate outliers
train=train[np.abs(train['count'])-train['count'].mean() <=(3*train['count'].std())]

#boxplot of count
sns.boxplot(x='count',data='train',color='mediumpurple')
plt.show()

#eliminate outliers between correlation
fig=plt.figure(figsize=(15,15))
for i,f1 in enumerate(fields):
    for j,f2 in enumerate(fields):
        idx=i*len(fields)+i+i
        ax=fig.add_subplot(len(fields),len(fields),idx)
        ax.scatter(train[f1],train[f2])
        ax.set_ylabel(f1)
        ax.set_xlabel(f2)
        
plt.show()

drop_idx=train[(train['atemp']>20) & (train['atemp']<40) & (train['atemp']>10) & (train['atemp']<20)].index
train=train.drop(drop_idx)

#standard scaling nunmeric columns
from sklearn.preprocessing import MinMaxScaler
def scaling(df):
    scaler=MinMaxScaler()
    num_cols=['temp','atemp','humidity','windspeed']
    df[num_cols]=scaler.fit_transform(df[num_cols])
    return df

train=scaling(train)
test=scaling(test)

train.head()

#split train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(train.drop['count'],axis=1),train['count'],test_size=0.3

def rmsle(y,pred):
    log_y=np.log1p(y)
    log_pred=np.log1p(pred)
    squared_error=(log_y-log_pred)**2
    rmsle=np.sqrt(np.mean(squared_error))
    return rmsle

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV


def evaluate(reg_cls,params=None):
    reg=reg_cls()
    if params:
        reg=GridSearchCV(reg,param_grid=params,refit=True)
    reg.fit(X_train,y_train)
    pred=reg.predict(X_test)
    
    y_test_exp=np.expm1(y_test)
    pred_exp=np.expm1(pred)
    print('\n',reg_cls)
    
    if params:
        print(reg.best_params_)
        reg=reg.best_estimator_
    print(rmsle(y_test_exp,pred_exp))
    return reg,pred_exp


lr_reg,pred_lr=evaluate(LinearRegression)
rg_reg,pred_rg=evaluate(Ridge)
ls_reg,pred_ls=evaluate(Lasso)
rf_reg,pred_rf=evaluate(RandomForestRegressor)
gb_reg,pred_gb=evaluate(GradientBoostingRegressor)
xg_reg,pred_xg=evaluate(XGBRegressor)
lg_reg,pred_lg=evaluate(LGBMRegressor)

params={'n_estimators':[100*i for i in range(1,6)]}
xg_reg,pred_xg=evaluate(XGBRegressor,params)
lg_reg,pred_lg=evaluate(LGBMRegressor,params)

def feature_importance(reg):
    plt.figure(figsize=(20,10))
    print(type(reg))
    df=pd.DataFrame(sorted(zip(X_train.columns,reg.feature_importance_)),columns=['features','values'])
    sns.barplot(x='values',y='features',data=df.sort_values(by='values',ascending=True))
    plt.show()
    
feature_importance(xg_reg)
feature_importance(lg_reg)

submission=pd.read_csv("/content/sampleSubmission.csv")
submission.head()

test.shape
submission.shape

#pred=xg_reg.predict(test)
pred=lg_reg.predict(test)
pred_exp=np.expm1(pred)
print(pred_exp)

submission.loc[:,'count']=pred_exp


    
    
