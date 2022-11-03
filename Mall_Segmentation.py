#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('E:\Data Analyst\Python Projects\Mall\Mall_Customer.csv')


# In[3]:


df.head(5)


# # Univariant Analysis

# In[4]:


df.describe()


# In[5]:


sns.distplot(df['Annual Income (k$)']);


# In[6]:


df.columns


# In[7]:


cols = [ 'Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in cols:
    plt.figure()
    sns.distplot(df[i])


# In[8]:


sns.kdeplot(df['Annual Income (k$)'],shade=True,hue= df['Gender']);


# In[9]:


cols = [ 'Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in cols:
    plt.figure()
    sns.kdeplot(df[i],shade=True,hue= df['Gender']);


# In[10]:


cols = [ 'Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in cols:
    plt.figure()
    sns.boxplot(data = df, x = 'Gender', y = df[i]);


# In[11]:


df['Gender'].value_counts(normalize = True)


# # Bivariant Analysis

# In[12]:


sns.scatterplot(data = df, x = 'Annual Income (k$)', y = 'Spending Score (1-100)');


# In[13]:


#df = df.drop('CustomerID',axis = 1)
sns.pairplot(df,hue = 'Gender')


# In[14]:


df.groupby(['Gender'])[ 'Age', 'Annual Income (k$)','Spending Score (1-100)'].mean()


# In[15]:


df.corr()


# In[16]:


sns.heatmap(df.corr(),annot = True,cmap = 'coolwarm');


# # Clustering - Univariant, Bivariant, Multivariant

# In[17]:


clustering1 = KMeans(n_clusters = 3)


# In[18]:


clustering1.fit(df[['Annual Income (k$)']])


# In[19]:


clustering1.labels_


# In[20]:


df['Income Cluster'] = clustering1.labels_
df.head()


# In[21]:


df['Income Cluster'].value_counts()


# In[22]:


clustering1.inertia_


# In[23]:


inertia_scores = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df[['Annual Income (k$)']])
    inertia_scores.append(kmeans.inertia_)


# In[24]:


inertia_scores


# In[25]:


plt.plot(range(1,11),inertia_scores);


# In[26]:


df.columns


# In[27]:


df.groupby('Income Cluster')['Age', 'Annual Income (k$)', 'Spending Score (1-100)'].mean()


# # Bivariant Clustering

# In[28]:


clustering2 = KMeans(n_clusters = 5)
clustering2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
clustering2.labels_
df['Spending and Income Cluster'] = clustering2.labels_
df.head()


# 

# In[29]:


inertia_scores2 = []
for i in range(1,11):
    kmeans2 = KMeans(n_clusters = i)
    kmeans2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
    inertia_scores2.append(kmeans2.inertia_)

plt.plot(range(1,11),inertia_scores2);


# In[30]:


centers = pd.DataFrame(clustering2.cluster_centers_)
centers.colums=['x','y']


# In[31]:


plt.figure(figsize=(10,8))
#plt.scatter(x=centers['x'], y=centers['y']),s=100,c='black',marker='*')
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Spending and Income Cluster', palette='tab10')
plt.savefig('E:\Data Analyst\Python Projects\Mall\Clustering_BIvariant.png')


# In[32]:


pd.crosstab(df['Spending and Income Cluster'],df['Gender'],normalize ='index')


# In[33]:


df.groupby('Spending and Income Cluster')['Age', 'Annual Income (k$)', 'Spending Score (1-100)'].mean()


# # Multivariant Cluster

# In[34]:


from sklearn.preprocessing import StandardScaler


# In[35]:


scale = StandardScaler()


# In[36]:


df.head()


# In[37]:


dff = pd.get_dummies(df,drop_first = True)
dff.head()


# In[38]:


dff.columns


# In[39]:


dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Male']]
dff.head()


# In[40]:


dff = pd.DataFrame(scale.fit_transform(dff))


# In[41]:


inertia_scores3 = []
for i in range(1,11):
    kmeans3 = KMeans(n_clusters = i)
    kmeans3.fit(dff)
    inertia_scores3.append(kmeans3.inertia_)

plt.plot(range(1,11),inertia_scores3);


# In[42]:


df.to_csv('E:\Data Analyst\Python Projects\Mall\Clustering.csv')


# In[ ]:




