#!/usr/bin/env python
# coding: utf-8

# ## LIVER PATIENT 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv('Indian Liver Patient Dataset (ILPD).csv')


# In[3]:


data.head()


# In[4]:


data.describe()


# In[5]:


data.info()


# In[6]:


data.describe(include=['O'])


# In[7]:


data.dropna(inplace=True)


# In[8]:


data.info()


# ## DOMAIN ANALYSIS

# In[9]:


data['Target'] = data['Target'].replace({2:0})


# In[10]:


import sweetviz as sv
my_report = sv.analyze(data)
my_report.show_html()


# ## INSIGHTS

# In[11]:


sns.countplot(x='Gender',data=data,hue='Target')
plt.show()


# In[12]:


data.columns


# In[13]:


sns.histplot(x='Age',data=data,hue='Target',palette=['pink','blue'])
plt.show()


# In[14]:


sns.histplot(x='Direct_Bilirubin',data=data,hue='Target',palette=['pink','blue'])
plt.show()


# In[15]:


sns.histplot(x='Alkaline_Phosphotase',data=data,hue='Target',palette=['pink','blue'])
plt.show()


# In[16]:


sns.relplot(y='Age',data=data,x='Alkaline_Phosphotase',hue='Target',palette=['red','blue'])
plt.show()


# In[17]:


sns.relplot(y='Total_Bilirubin',data=data,x='Direct_Bilirubin',hue='Target',palette=['red','blue'])
plt.show()


# In[18]:


sns.histplot(x='Direct_Bilirubin',data=data,hue='Target',palette=['pink','blue'])
plt.show()


# In[22]:


plt.figure(figsize=(30,15))
sns.boxplot(x='Age',data=data)
plt.tight_layout()


# In[23]:


plt.figure(figsize=(30,15))
sns.boxplot(x='Gender',data=data)
plt.tight_layout()


# In[24]:


plt.figure(figsize=(30,15))
sns.boxplot(x='Total_Bilirubin',data=data)
plt.tight_layout()


# In[25]:


plt.figure(figsize=(30,15))
sns.boxplot(x='Direct_Bilirubin',data=data)
plt.tight_layout()


# In[26]:


plt.figure(figsize=(30,15))
sns.boxplot(x='Alkaline_Phosphotase',data=data)
plt.tight_layout()


# In[27]:


plt.figure(figsize=(30,15))
sns.boxplot(x='Alamine_Aminotransferase',data=data)
plt.tight_layout()


# In[28]:


plt.figure(figsize=(30,15))
sns.boxplot(x='Alkaline_Phosphotase',data=data)
plt.tight_layout()


# In[29]:


plt.figure(figsize=(30,15))
sns.boxplot(x='Aspartate_Aminotransferase',data=data)
plt.tight_layout()


# In[30]:


plt.figure(figsize=(30,15))
sns.boxplot(x='Total_Protiens',data=data)
plt.tight_layout()


# In[31]:


plt.figure(figsize=(30,15))
sns.boxplot(x='Albumin',data=data)
plt.tight_layout()


# In[32]:


plt.figure(figsize=(30,15))
sns.boxplot(x='Albumin_and_Globulin_Ratio',data=data)
plt.tight_layout()


# In[ ]:





# In[ ]:





# In[19]:


data['Gender'] = data['Gender'].replace({'Male':0,'Female':1 })


# In[20]:


data.Gender.value_counts()


# In[21]:


data['Gender'] = data['Gender'].replace({'Male':1,'Female':0 })


# In[ ]:





# ## FEATURE ENGENERING

# In[33]:


plt.figure(figsize=(15,15))
sns.heatmap(data.corr(),annot=True)


# ## MODEL CREATION

# In[34]:


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler() # creating the object
dl=['Target']
data1=sc.fit_transform(data.drop(dl,axis=1))


# In[35]:


con_data=data[['Target']]


# In[36]:


data.columns


# In[37]:


data2=pd.DataFrame(data1,columns=['Age','Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
       'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens','Albumin','Albumin_and_Globulin_Ratio'])
data2


# In[38]:


final_df=pd.concat([data2,con_data],axis=1)


# In[39]:


final_df


# In[40]:


final_df.isnull().sum()


# In[41]:


final_df.dropna(inplace=True)


# In[42]:


final_df


# In[43]:


final_df.isnull().sum()


# In[ ]:





# In[44]:


X = data.drop('Target', axis=1) 
y = data.Target


# In[45]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=45)


# In[46]:


from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(X_train,y_train)


# In[47]:


y_pred=clf.predict(X_test)


# In[48]:


from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,classification_report,f1_score


# In[49]:


print(classification_report(y_test,y_pred))


# In[50]:


precision=precision_score(y_test,y_pred)
precision


# In[51]:


recall=recall_score(y_test,y_pred)
recall


# In[52]:


f1score=f1_score(y_test,y_pred)
f1score


# In[53]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()  
dt.fit(X_train,y_train)
y_hat=dt.predict(X_test)
y_hat 


# In[54]:


print(classification_report(y_test,y_hat))


# In[ ]:





# In[55]:


from imblearn.over_sampling import SMOTE
smote = SMOTE()


# In[56]:


X_smote, y_smote = smote.fit_resample(X_train,y_train)


# In[57]:


from sklearn.svm import SVC
svclassifier = SVC() 
svclassifier.fit(X_smote, y_smote) 


# In[58]:


y_hat=svclassifier.predict(X_test)


# In[59]:


print(classification_report(y_test,y_hat))


# In[60]:


recall=recall_score(y_test,y_hat)
recall


# In[61]:


f1score=f1_score(y_test,y_hat)
f1score


# In[ ]:




