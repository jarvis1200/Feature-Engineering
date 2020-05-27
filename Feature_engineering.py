import numpy as np
import pandas as pd

train= pd.read_csv('F:\\titanic\\train.csv')
test= pd.read_csv('F:\\titanic\\test.csv')

print(train.head())
print(train.describe())


a=['Cabin','Ticket']

train.drop(a,1, inplace= True)
test.drop(a, 1, inplace= True)
print(train.columns)

print(train.isnull().sum().sort_values(ascending= False))
train['Age'].fillna(train['Age'].median(), inplace= True)
train['Embarked'].fillna('s', inplace= True)
print(train['Embarked'].head(20))

for name in train['Name']:
    train['Title']= train['Name'].str.extract('([A-za-z]+)\.', expand= True)
titel_replace={"Mlle": "Other", "Major": "Other", "Col": "Other", "Sir": "Other", "Don": "Other", "Mme": "Other",
          "Jonkheer": "Other", "Lady": "Other", "Capt": "Other", "Countess": "Other", "Ms": "Other", "Dona": "Other"}
train.replace({'Title': title_replace}, inplace= True)
train.replace({'Title': title_replace}, inplace= True)
print(train['Title'].head())

#one hot encoding
b= train[['Embarked', 'Pclass', 'Title']]
b= pd.get_dummies(b, columns=['Embarked', 'Pclass', 'Title'], drop_first= False)
print(b.head())

#label encoding
from sklearn.preprocessing import LabelEncoder
c= train[['Embarked', 'Pclass', 'Title']]
c= c.apply(LabelEncoder().fit_transform)
print(c.head())

#frequency encoder
ap= train[['Embarked', 'Pclass', 'Title']]
q= ap.groupby(['Title']).size().reset_index()
q.columns = ['Title', 'Freq_Encode']
print(q.head())
ap= pd.merge(ap,q,on='Title', how= 'left')
print(ap.head(10))

#mean encoding
sample_train= train[['Title', 'Survived']]
w= sample_train.groupby(['Title'])['Survived'].sum().reset_index()
w= w.rename(columns={'Survived': 'Tit_sur_sum'})

e= sample_train.groupby(['Title'])['Survived'].count().reset_index()
e= e.rename(columns={'Survived': 'Tit_sur_count'})

r= pd.merge(w, e, on= 'Title', how= 'inner')
r['target_encoded']= r['Tit_sur_sum']/r['Tit_sur_count']
print(r.head())

r= r[['Title', 'target_encoded']]

sample_train= pd.merge(sample_train, r, on= 'Title', how= 'left')
print(sample_train.head(10))

#direct method mean encode

sample_train= train[['Title', 'Fare']]

t= sample_train.groupby(['Title'])['Fare'].mean().reset_index()
t= t.rename(columns= {'Fare': 'Title'+'_mean_enc'})
print(t.head())

sample_train= pd.merge(sample_train, t, on= 'Title', how= 'left')
print(sample_train.head())

# k-fold encode

import  sklearn
from sklearn.model_selection import StratifiedKFold
X= train[['Embarked','Title','Pclass', 'Fare' ]]
cols= ['Embarked', 'Pclass', 'Title']
kf= sklearn.model_selection.KFold(n_splits= 10, shuffle= False)

for i in cols:
    X['Mean_Encode_on']= np.nan

    for tr_ind, val_ind in kf.split(X):
        x_tr,x_val= X.iloc[tr_ind], X.iloc[val_ind]
        X.loc[X.index[val_ind], 'Mean_Encode_on']= x_val[i].map(x_tr.groupby(i).Fare.mean())
        X= X.rename(index= str, columns={'Mean_Encode_on': i + '_k_enc'})
print(X['Pclass_k_enc'].head(15))

