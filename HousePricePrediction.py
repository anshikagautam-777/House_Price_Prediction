#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
dataset = pd.read_excel("HousePricePrediction.xlsx")
 
# Printing first 5 records of the dataset
print(dataset.head(5))


# In[36]:


dataset.shape


# In[37]:


obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))

int_ = (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))


# In[38]:


plt.figure(figsize=(12, 6))
sns.heatmap(dataset.corr(),
			cmap = 'BrBG',
			fmt = '.2f',
			linewidths = 2,
			annot = True)


# In[39]:


unique_values = []
for col in object_cols:
  unique_values.append(dataset[col].unique().size)
plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols,y=unique_values)


# In[40]:


#Data Cleaning

dataset.drop(['Id'],
			axis=1,
			inplace=True)

dataset['SalePrice'] = dataset['SalePrice'].fillna(
dataset['SalePrice'].mean())

new_dataset = dataset.dropna()

new_dataset.isnull().sum()

from sklearn.preprocessing import OneHotEncoder

s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('No. of. categorical features: ',
	len(object_cols))

OH_encoder = OneHotEncoder(sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names()
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)


# In[41]:


#Splitting Data into test and training

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

# Split the training set into
# training and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(
	X, Y, train_size=0.8, test_size=0.2, random_state=0)


# In[42]:


#Random Forest Method for Predicting Result

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)

Res=X_valid
Res['Predicted Sales Price']=Y_pred
Res['Actual Sales Price']=Y_valid
Res


# In[34]:


mean_absolute_percentage_error(Y_valid, Y_pred)


# In[ ]:




