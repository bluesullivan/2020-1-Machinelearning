#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler  
StdScaler = StandardScaler()


# In[25]:


train = pd.read_csv('train.csv')
test= pd.read_csv('test.csv')


# In[26]:


train1 = train.drop(['id'],axis=1)
test1 = test.drop(['id'],axis=1)


# In[27]:


#(행의개수, 열의개수)
print(train1.shape) 
print(test1.shape)


# In[28]:


#클래스 유형, 행 인덱스 구성, 열이름의 종류와 개수, 각 열의 자료형과 개수, 메모리햘당량정보 
print(train1.info())
print(test1.info())


# In[29]:


#기술통계정보(평균, 표준편차, 최대값, 최소값, 중간값)
print(train1.describe())
print(test1.describe())


# In[30]:


#결측확인 
train.isnull().sum()


# In[31]:


#중복확인 
train.duplicated().sum()


# In[32]:


temperature = ["X00","X07","X28","X31","X32"] #기온
localpress = ["X01","X06","X22","X27","X29"] #현지기압
speed= ["X02","X03","X18","X24","X26"] #풍속
water = ["X04","X10","X21","X36","X39"] #일일 누적강수량
press= ["X05","X08","X09","X23","X33"] #해면기압
sun = ["X11","X14","X16","X19","X34"] #일일 누적일사량
humidity= ["X12","X20","X30","X37","X38"] #습도
direction= ["X13","X15","X17","X25","X35"] #풍향


# In[33]:


ylocal = ['Y00', 'Y01', 'Y02', 'Y03', 'Y04', 'Y05', 'Y06',
          'Y07', 'Y08','Y09', 'Y10', 'Y11', 'Y12',
          'Y13', 'Y14', 'Y15', 'Y16', 'Y17', 'Y18']
train[ylocal].head()


# In[34]:


#plot 그리기 plot(xvalue , yvalue, figsize=(가로,세로), 제목)
train.plot(x='id', y=temperature, figsize=(8,3), title="temperature")
train.plot(x='id', y=localpress, figsize=(8,3), title="localpress")
train.plot(x='id', y=speed, figsize=(8,3), title="speed")
train.plot(x='id', y=water, figsize=(8,3), title="water")
train.plot(x='id', y=press, figsize=(8,3), title="press")
train.plot(x='id', y=sun, figsize=(8,3), title="sun")
train.plot(x='id', y=humidity, figsize=(8,3), title="humidity")
train.plot(x='id', y=direction, figsize=(8,3), title="direction")


# In[35]:


#예를들어 풍속과 습도 평균 표준편차값 
print(train[speed])
print(train[speed].mean())
print(train[speed].std())
print('\n')
print(train[humidity])
print(train[humidity].std())


# In[36]:


#y00 ~ y18 temperature
train.loc[:,ylocal].plot(figsize=(20,10))


# In[37]:


#X00 ~ X39 변수와 Y00 ~ Y17의 상관관계 및 히트맵
heatmap = train.iloc[: , 1:-1]
corr=heatmap.corr()
plt.figure(figsize=(60,30))
sns.heatmap(corr, cmap = "RdYlGn", annot = True, vmin=0, vmax=1) 


# In[38]:


#X00~ X39 변수와 Y18끼리의 상관관계 및 히트맵 
heatmap2 = train.iloc[4320: ,1:40]
heatmap3 = train.iloc[:, -1:]
heatmap = pd.concat([heatmap2, heatmap3], axis=1)
corr = heatmap.corr()
plt.figure(figsize=(60,30))
sns.heatmap(corr, cmap =plt.cm.RdYlBu_r, annot = True, vmin=0, vmax=1) 


# In[39]:


print(heatmap)


# In[40]:


#heatmap = heatmap.drop()
#plt.figure(figsize=(20,10))
#sns.barplot(x='X00', y='Y18', data=heatmap[temperature].T.reset_index()).set(ylim=(0, 1))
#plt.grid()
#plt.show()


# In[41]:


#Y00 ~ Y17의 상관관계 및 히트맵
y17 = ylocal[:-1]
corr = train[y17].corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr, cmap = "RdYlGn", annot = True, vmin=0, vmax=1) 


# In[42]:


#표준화 
def standard(df):
    mean = np.mean(df)
    std = np.std(df)
    norm = (df - mean) / (std - 1e-07)
    return norm, mean, std

xname = train.loc[:,"X00":"X39"].columns
tnorm,tmean,tstd = standard(train[xname])
print(tnorm)
print(tmean)
print(tstd)


# In[43]:


#표준화한 새로운 데이터셋
train2 = pd.concat([train["id"], tnorm], axis=1) 

print(train)
print(train2)


# In[44]:


#범주형(카데고리) 데이터 처리   온도
#"X00","X07","X28","X31","X32"

count, bin_dividers = np.histogram(train[temperature], bins=3)
print(bin_dividers)

bin_names = ['저온', '보통', '고온']

train['hp_bin00'] = pd.cut(x=train['X00'], #데이터 배열
bins=bin_dividers, # 경계 값 리스트
labels=bin_names, # bin 이름
include_lowest=True) # 첫 경계값 포함

train['hp_bin07'] = pd.cut(x=train['X07'], #데이터 배열
bins=bin_dividers, # 경계 값 리스트
labels=bin_names, # bin 이름
include_lowest=True) # 첫 경계값 포함

print(train[['X00','hp_bin00','X07', 'hp_bin07']].sample(3))
print("\n")
temperature_dummies = pd.get_dummies(train['hp_bin00'])
print(temperature_dummies)


# In[45]:


#히트맵으로 파악한 고장난 센서 버리기
sensor = ['X14','X16','X19']
train.drop(sensor, axis=1, inplace=True)
test.drop(sensor, axis=1, inplace=True)


# In[46]:


#풍향 버리기 
drop_wind = ['X13','X15','X17','X25','X35']
train.drop(drop_wind, axis=1, inplace=True)
test.drop(drop_wind, axis=1, inplace=True)


# In[17]:


print(train.info())


# In[46]:


#시간...?


# In[47]:


print(train['X11'].head(100))
print(train['X34'])


# In[50]:


#누적일조량->단순일조량
def my_sun(train):
    temp = train.iloc[:,np.where(train.columns.str.find('X11') == 0)[0]]
    temp2 = temp.copy()
    for i in range(1, len(temp)):
        temp2.iloc[i,:] = temp.iloc[i,:] - temp.iloc[i-1,:]
    temp2.iloc[np.where(temp2.sum(axis=1) < 0)[0],:] = 0
    train['X11'] = temp2['X11']
    
    temp = train.iloc[:,np.where(train.columns.str.find('X34') == 0)[0]]
    temp2 = temp.copy()
    for i in range(1, len(temp)):
        temp2.iloc[i,:] = temp.iloc[i,:] - temp.iloc[i-1,:]
    temp2.iloc[np.where(temp2.sum(axis=1) < 0)[0],:] = 0    
    train['X34'] = temp2['X34']
    
    return train

train = my_sun(train)
test = my_sun(test)


# In[49]:


print(train['X11'].head(100))
print(train['X34'])


# In[47]:


interval = 12
value = int(144/interval)
a = np.array([[i]*value for i in range(0, interval)]).reshape(-1)
print(a)


# In[48]:


#interval 12 value 12 
interval = 12
value = int(144/interval)
a = np.array([[i]*value for i in range(0, interval)]).reshape(-1)
test['date'] = np.array([[i]*144 for i in range(0,80)]).reshape(-1)
test['day'] = np.array([a for i in np.arange(0,80)]).reshape(-1)

train['date'] = np.array([[i]*144 for i in range(0,33)]).reshape(-1)
train['day'] = np.array([a for i in np.arange(0,33)]).reshape(-1)


# In[64]:


print(train['day'].head(15))
print(train['day'].head(100))
print(train['day'].head(144))
print(train['day'].head(145))
print(train['day'].head(150))
print(train['day'].head(200))


# In[59]:


print(train['date'].head(144))
print(train['date'].head(145))
print(train['date'].head(200))


# In[61]:


a = np.array([[i]*24 for i in range(0, 6)]).reshape(-1)
test['day2'] = np.array([a for i in np.arange(0,80)]).reshape(-1)
train['day2'] = np.array([a for i in np.arange(0,33)]).reshape(-1)


# In[63]:


print(train['day2'].head(30))
print(train['day2'].head(145))
print(train['day2'].head(200))


# In[54]:


print(train['day'].value_counts())
print(train['day2'])


# In[ ]:




