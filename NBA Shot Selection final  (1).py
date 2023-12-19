#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv('data.csv')


# In[3]:


data


# In[4]:


data.shot_zone_basic.value_counts()


# In[5]:


data.columns


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.describe(include=['O'])


# In[9]:


data.columns


# In[10]:


data.drop(['team_name','team_id','game_event_id', 'game_id','lat','shot_id','season','game_date', 'matchup' ,'lon'],axis='columns',inplace=True)


# In[11]:


data


# In[12]:


data.describe()


# In[13]:


data.describe(include=['O'])


# In[14]:


data.info()


# In[15]:


data.dropna(inplace=True)


# In[16]:


data.info()


# # DOMAIN ANALYSIS:

# Action_type - The action type here is the type of the shot a player(kobe bryant) has played. The action type includes jump shot,running jump shot,layup shots.
# Combined_shot_type- The combined shots are the same as the shots such as Dunk shots ,jump shots performed the team.
# Minutes_remaining - As the basketball game is of 4 quarters,each of 12 minutes, so this column represent the minutes remaining                       at the quarter end.
# Period - The period is commonly known as quarter which consists of 12 minutes each.
# Shot_zone - It represents the zone from where the shot was made. The zones are basically the centre, right,left,left side                   centre.
#  Shot_zone_range - This column shows the particular range of the zone from where the shot is made.
#  Team_name - The name of the team playing the game/match.
#  Game_date- On which date the game was played.
#  Opponent - Againts whom the game was being played, the name of the oppposite team.
#  Match_up - The column mentions the name of teams which played against each other.

# In[17]:


import sweetviz as sv
my_report = sv.analyze(data)
my_report.show_html()


# # UNIVARIATE INSIGHTS:

# # Jump shot has the highest percentage,around 65%.
# #  Most of shot made at the beginning and the last minute of the game (22%).
# #  3rd quarter has slightly higher percentage than other (27%).
# #  Playoff has very low shots accurracy percentage.
# #  seasons during 2005-10 are the most scored seasons (7%).
# #  Higher number of shots made when the shot distace is below 30.
# #  Missing shots has higher percentage than accurate shots (60%).
# #  2PT shots have higher percentage than 3PT shots (80%).
# #  Center court is the most favourite area of the players.40% shots are from this     region.
# #  Most of the shots are made from mid range when shot range is less than 8ft.

# In[18]:


categorical_col = []
for column in data.columns:
    if data[column].dtype == object and len(data[column].unique()) <= 50: 
        categorical_col.append(column)
        print(f"{column} : {data[column].unique()}")
        print("====================================")


# In[19]:


categorical_col


# In[20]:


data1=data[['combined_shot_type'
 ,
 'shot_type',
 'shot_zone_area',
 'shot_zone_basic',
 'shot_zone_range',
 'opponent']]


# In[21]:


data1


# In[22]:


plt.figure(figsize=(70,60))#canvas size
plotnumber = 1

for column in data1:
    ax = plt.subplot(3,3,plotnumber)
    sns.countplot(x=data1[column].dropna(axis=0)
                    ,hue=data.shot_made_flag)
    plt.xlabel(column,fontsize=20)
    plt.ylabel('shot_made_flag',fontsize=20)
    plotnumber+=1
plt.tight_layout()


# In[23]:


plt.figure(figsize=(20,20))
sns.countplot(x=data['action_type'],hue=data.shot_made_flag)
plt.xticks(fontsize=10,rotation=45)
plt.show()


# # INSIGHTS OF ACTION TYPE:
# 
# 1) Jump Shot-There are only 50% chances of the shot being made.
# 2) Driving Layup shot – There are more chances of this type of shot being made.
# 
# 

# In[24]:


sns.countplot(x=data1['combined_shot_type'],hue=data.shot_made_flag)
plt.show()


# # INSIGHTS OF COMBINED SHOT TYPE:
# 
# 1)Here also the chances of shots in jump shot is more same as action type.

# In[25]:


sns.scatterplot(x='loc_x',y='loc_y',data=data, hue=data.shot_made_flag)
plt.show()


# # INSIGHTS OF LOC_X AND LOC_Y:
# 
# 1)In this the probability of shots made were very less after 230 loc_y.

# In[26]:


sns.countplot(x='period',data=data,hue='shot_made_flag')
plt.show()


# # INSIGHTS OF PERIOD:
# 
# 1)There are 50-50 chances of shot being made in each period(quarter). No specific insight found.
# 
# 

# In[27]:


sns.countplot(x='playoffs',data=data,hue='shot_made_flag')
plt.show()


# # INSIGHTS OF PLAYOFF:
# 
# 1)No specific insight.
# 
# 

# In[28]:


sns.histplot(x='shot_distance',data=data,hue='shot_made_flag')
plt.show()


# # INSIGHTS OF SHOT_DISTANCE:
# 
# 1)There are more chances of shot being made  when the shot is made from a nearer distance from the basket.
# 
# 

# In[29]:


sns.countplot(x=data1['shot_type'],hue=data.shot_made_flag)
plt.show()


# # INSIGHTS OF SHOT_TYPE:
# 1)50-50 chances of shot being made in 2PT field goal.
# 
# 

# In[30]:


sns.countplot(x=data1['shot_zone_area'],hue=data.shot_made_flag)
plt.show()


# # INSIGHTS OF SHOT_ZONE_AREA:
# 
# 1)There are more chances of shot being made from the centre zone area.

# In[31]:


sns.countplot(x=data['shot_zone_basic'],hue=data.shot_made_flag)
plt.show()


# # INSIGHTS OF SHOT_ZONE_BASIC:
# 
# 1)The shots made are more in restricted area and 50-50 in In the Paint area and chances of shots made are less in mid range and above the break 3.
# 
# 

# In[32]:


sns.countplot(x=data1['shot_zone_range'],hue=data.shot_made_flag)
plt.show()


# # INSIGHTS OF SHOT_ZONE_RANGE:
# 
# 1)– The chances of shot being made are more in less than 8ft. And in the other ranges there are less chances.
# 
# 

# In[33]:


plt.figure(figsize=(30,40))
sns.countplot(x=data['opponent']
                    ,hue=data.shot_made_flag)
plt.xticks(fontsize=15)
plt.show()


# # INSIGHTS OF OPPONENT:
# 
# 1)No specific insight found.

# In[34]:


data['total_time_remaining']=data['minutes_remaining']*60 + data['seconds_remaining']


# In[35]:


data.drop(['minutes_remaining', 'seconds_remaining'],axis='columns',inplace=True)


# In[36]:


sns.histplot(x='total_time_remaining',data=data,hue='shot_made_flag')
plt.show()


# # INSIGHTS OF TOTAL_TIME_REMAINING:
# 
# 1)No specific insight found.
# 

# In[ ]:





# # Encoding

# In[37]:


data.action_type.value_counts()


# In[38]:


test = pd.DataFrame(data.action_type.values.tolist()).stack().value_counts()


# In[39]:


test.reset_index()
test.index


# In[40]:


One=['Jump Shot', 'Layup Shot', 'Driving Layup Shot', 'Turnaround Jump Shot',
       'Fadeaway Jump Shot', 'Running Jump Shot', 'Pullup Jump shot',
       'Turnaround Fadeaway shot', 'Slam Dunk Shot', 'Reverse Layup Shot',]
Two=['Turnaround Fadeaway shot', 'Slam Dunk Shot', 'Reverse Layup Shot',
       'Jump Bank Shot', 'Driving Dunk Shot', 'Dunk Shot', 'Tip Shot',
       'Step Back Jump shot', 'Alley Oop Dunk Shot', 'Floating Jump shot',]
Thr=['Step Back Jump shot', 'Alley Oop Dunk Shot', 'Floating Jump shot',
       'Driving Reverse Layup Shot', 'Hook Shot', 'Driving Finger Roll Shot',
       'Alley Oop Layup shot', 'Reverse Dunk Shot',
       'Driving Finger Roll Layup Shot', 'Turnaround Bank shot',]
fou=['Running Layup Shot', 'Driving Slam Dunk Shot', 'Running Bank shot',
       'Running Hook Shot', 'Finger Roll Layup Shot', 'Fadeaway Bank shot',
       'Finger Roll Shot', 'Driving Jump shot', 'Jump Hook Shot',
       'Running Dunk Shot']
fiv=['Reverse Slam Dunk Shot', 'Driving Hook Shot',
       'Pullup Bank shot', 'Follow Up Dunk Shot', 'Putback Layup Shot',
       'Turnaround Hook Shot', 'Running Reverse Layup Shot',
       'Cutting Layup Shot', 'Running Finger Roll Layup Shot',
       'Hook Bank Shot']
six=['Running Finger Roll Shot', 'Driving Bank shot',
       'Putback Dunk Shot', 'Driving Floating Jump Shot',
       'Running Pull-Up Jump Shot', 'Putback Slam Dunk Shot',
       'Turnaround Finger Roll Shot', 'Tip Layup Shot', 'Running Tip Shot',
       'Running Slam Dunk Shot', 'Driving Floating Bank Jump Shot']


# In[41]:


for i in data.action_type:
    if i in One:
        data.loc[data["action_type"]==i,"action_type"]=5


# In[42]:


for i in data.action_type:
    if i in Two:
        data.loc[data["action_type"]==i,"action_type"]=4


# In[43]:


for i in data.action_type:
    if i in Thr:
        data.loc[data["action_type"]==i,"action_type"]=3


# In[44]:


for i in data.action_type:
    if i in fou:
        data.loc[data["action_type"]==i,"action_type"]=2


# In[45]:


for i in data.action_type:
    if i in fiv:
        data.loc[data["action_type"]==i,"action_type"]=1


# In[46]:


for i in data.action_type:
    if i in six:
        data.loc[data["action_type"]==i,"action_type"]=0


# In[47]:


data.action_type.value_counts()


# In[48]:


data.combined_shot_type.value_counts()


# In[49]:


data.combined_shot_type=data.combined_shot_type.map({'Jump Shot':5,'Layup':4, 'Dunk':3, 'Tip Shot':2, 'Hook Shot':1, 'Bank Shot':0})


# In[50]:


data.combined_shot_type.value_counts()


# In[51]:


data['period'] = data['period'].replace({6: 3,5: 3,7: 3})


# In[52]:


data.shot_type.value_counts()


# In[53]:


data.shot_type=data.shot_type.map({'2PT Field Goal':1,'3PT Field Goal':0})


# In[54]:


data.shot_zone_area.value_counts()


# In[55]:


data.shot_zone_area=data.shot_zone_area.map({'Center(C)':5, 'Right Side Center(RC)':4, 'Right Side(R)':3, 'Left Side Center(LC)':2, 'Left Side(L)':1, 'Back Court(BC)':0 })


# In[56]:


data.shot_zone_basic.value_counts()


# In[57]:


data.shot_zone_basic=data.shot_zone_basic.map({'Mid-Range':6, 'Restricted Area':5, 'Above the Break 3':4, 'In The Paint (Non-RA)':3, 'Right Corner 3':2, 'Left Corner 3':1, 'Backcourt':0 })


# In[58]:


data.shot_zone_range.value_counts()


# In[59]:


data.shot_zone_range=data.shot_zone_range.map({'Less Than 8 ft.':4,'16-24 ft.':3,'8-16 ft.':2,'24+ ft.':1,'Back Court Shot':0})


# In[60]:


test = pd.DataFrame(data.opponent.values.tolist()).stack().value_counts()


# In[61]:


test.reset_index()
test.index


# In[62]:


One=['SAS', 'PHX', 'HOU', 'SAC', 'DEN', 'POR', 'UTA', 'MIN', 'GSW', 'LAC']
Two=['DAL', 'MEM', 'BOS', 'SEA', 'IND', 'ORL', 'PHI', 'DET', 'NYK', 'OKC']
Thr=['TOR', 'MIA', 'CHI', 'CLE', 'MIL', 'WAS', 'CHA', 'NOH', 'ATL', 'NJN']
fou=['NOP', 'VAN', 'BKN']


# In[63]:


for i in data.opponent:
    if i in One:
        data.loc[data["opponent"]==i,"opponent"]=3


# In[64]:


for i in data.opponent:
    if i in Two:
        data.loc[data["opponent"]==i,"opponent"]=2


# In[65]:


for i in data.opponent:
    if i in Thr:
        data.loc[data["opponent"]==i,"opponent"]=1


# In[66]:


for i in data.opponent:
    if i in fou:
        data.loc[data["opponent"]==i,"opponent"]=0


# In[ ]:





# In[ ]:





# In[67]:


sns.heatmap(data.corr(),annot=True)


# # Model Creation

# In[68]:


X = data.drop('shot_made_flag', axis=1) 
y = data.shot_made_flag


# In[69]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=45)


# In[70]:


from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(X_train,y_train)


# In[71]:


y_pred=clf.predict(X_test)


# In[72]:


y_pred_prob=clf.predict_proba(X_test)


# In[73]:


from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,classification_report,f1_score


# In[74]:


print(classification_report(y_test,y_pred))


# In[75]:


from sklearn.tree import DecisionTreeClassifier#importing decision tree from sklearn.tree
dt=DecisionTreeClassifier()#object creation for decision tree  
dt.fit(X_train,y_train)#training the model
y_hat=dt.predict(X_test)#prediction
y_hat#predicted values 


# In[77]:


print(classification_report(y_test,y_hat))


# In[78]:


from imblearn.over_sampling import SMOTE
smote = SMOTE()


# In[79]:


X_smote, y_smote = smote.fit_resample(X_train,y_train)


# In[80]:


from collections import Counter
print("Actual Classes",Counter(y_train))
print("SMOTE Classes",Counter(y_smote))


# In[81]:


from sklearn.svm import SVC
svclassifier = SVC() ## base model with default parameters
svclassifier.fit(X_smote, y_smote) #Training 


# In[82]:


y_hat=svclassifier.predict(X_test)


# In[83]:


print(classification_report(y_test,y_hat))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




