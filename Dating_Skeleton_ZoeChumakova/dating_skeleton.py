
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
from matplotlib import style
style.use("seaborn-white")


# In[74]:


df = pd.read_csv("profiles.csv")


# In[4]:


#print(df['education'].value_counts())
#print(df['location'].value_counts())
#print(df['status'].value_counts())
#print(df['sex'].value_counts())
#print(df['pets'].value_counts())
#print(df['sign'].value_counts())
#print(df['diet'].value_counts())
#print(df['body_type'].value_counts())
#print(df['offspring'].value_counts())
#print(df['income'].value_counts())
#print(df['age'].value_counts())


# In[75]:


print(len(df.income))


# In[76]:


df['age'] = df['age'].replace(np.nan, '', regex=True)
#df['income'] = df['income'].replace(-1, '', regex=True)
#print(df['income'])
#income = df['income']
#age = df['age']


# In[77]:


df['age_group'] = pd.cut(df.age,[0,20,30,40,50,60, 110], labels = ['0-20', '21-30', '31-40', '41-50', '51-60','60+'])
print(df.age_group.value_counts())
age_group = df['age_group']


# In[78]:


#df['income_group'] = pd.cut(df.income,[0,20000,30000, 40000,50000, 100000, 500000], labels = ['0-20K','20-30K', '30-40K', '40-50K', '50-100K','100-500K'])
#print(df.income_group.value_counts())
#income_group = df['income_group']
#[0,20000','50000', '80000', '120000','200000', '500000']


# In[79]:


#print(len(df.age_group))
#print(len(df.income_group))


# In[80]:


plt.hist(age_group, bins=10)
plt.xlabel("Age")
plt.ylabel("Number of responses")


# In[81]:


print(df.drinks.value_counts())
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often" : 3, "very often":4, "desperately": 5}
df["drinks_code"] = df.drinks.map(drink_mapping)


# In[82]:


print(df.drinks_code.value_counts())


# In[83]:


#same for smokes
print(df.smokes.value_counts())
smokes_mapping = {"no": 0, "sometimes":1, "when drinking": 2, "yes": 3, "trying to quit": 4}
df["smokes_code"] = df.smokes.map(smokes_mapping)


# In[84]:


print(df.smokes_code.value_counts())


# In[85]:


#same for drugs
print(df.drugs.value_counts())
drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}
df["drugs_code"] = df.drugs.map(drugs_mapping)


# In[86]:


print(df.drugs_code.value_counts())


# In[87]:


print(df.sex.value_counts())


# In[88]:


sex_mapping = {"m": 1, "f":2}
df["sex_code"] = df.sex.map(sex_mapping)
print(df.sex_code.value_counts())


# In[117]:


import statsmodels as sm
import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics
import sklearn.tree as tree
import seaborn as sns

import math

future_data=df[["age", "sex_code", "income", "education", "status", "drinks_code","drugs_code", "smokes_code","job"]]

print(future_data)

fig = plt.figure(figsize=(30,30))
cols = 3
rows = math.ceil(float(future_data.shape[1]) / cols)
for i, column in enumerate(future_data.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if future_data.dtypes[column] == np.object:
        future_data[column].value_counts().plot(kind="bar", axes=ax)
    else:
        future_data[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.5, wspace=0.2)


# In[105]:


(df["income"].value_counts() / df.shape[0])


# In[106]:


(df["drinks_code"].value_counts() / df.shape[0])


# In[107]:


(df["sex"].value_counts() / df.shape[0])


# In[108]:


(df["drugs_code"].value_counts() / df.shape[0])


# In[109]:


(df["education"].value_counts() / df.shape[0])


# In[110]:


(df["status"].value_counts() / df.shape[0])


# In[111]:


(df["job"].value_counts() / df.shape[0])


# In[139]:


#same for smokes
print(df.status.value_counts())
status_mapping = {"single": 0, "seeing someone":1, "available": 2, "married": 3, "unknown": 4}
df["status_code"] = df.status.map(status_mapping)
print(df.status_code.value_counts())


# In[118]:


future_data.plot(kind = 'scatter', x ='drinks_code', y = 'drugs_code', c = 'smokes_code', colormap = 'ocean_r')


# In[119]:


pd.plotting.scatter_matrix(future_data[['drinks_code', 'drugs_code', 'smokes_code',  'sex_code']],
                          figsize=(10,8))


# In[140]:


df.corr()


# In[147]:


df.corr()


# In[148]:


sns.heatmap(df.corr())


# In[143]:


sns.heatmap(df.corr(),
                cbar=True,
                annot=True,
                square=True,
                fmt='.2f',
                annot_kws={'size':10},)
plt.show()


# In[149]:


df.plot(kind='scatter',x='income',y= 'sex_code',alpha=0.2)


# In[151]:


sns.lmplot(x='drugs_code',
           y='drinks_code',
           data=df,
           aspect=1.5,
           scatter_kws={'alpha':0.2})


# In[153]:


df[["sex_code", "status_code"]].head(15)


# In[158]:


df.plot(kind='scatter',x='income',y= 'status_code',alpha=0.2)


# In[161]:


sns.lmplot(x='income',
           y='status_code',
           data=df,
           aspect=1.5,
           scatter_kws={'alpha':0.2})


# In[169]:


cls = linear_model.LogisticRegression()

cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["Target"].classes_, yticklabels=encoders["Target"].classes_)
plt.ylabel("Income")
plt.xlabel("Predicted status")
print ("F1 score:" % skl.metrics.f1_score(y_test, y_pred))
coefs = pd.Series(cls.coef_[0], index=X_train.columns)
coefs.sort()
plt.subplot(2,1,2)
coefs.plot(kind="bar")
plt.show()

