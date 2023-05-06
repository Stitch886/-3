# coding=gbk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import plotly as py
# import plotly.graph_objs as go
from sklearn.cluster import KMeans
import warnings
import os

warnings.filterwarnings("ignore")
# py.offline.init_notebook_mode(connected = True)
df = pd.read_csv(r'Mall_Customers.csv')
df.head()
# 将Gender的Male改为1,Female改为0
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# 新增一列Gender_int
df['Gender_int'] = df['Gender']
plt.figure(1, figsize=(15, 6))
n = 0
for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.distplot(df[x], bins=20)
    plt.title('Distplot of {}'.format(x))
plt.show()
# The Elbow Method手肘法则，计算平方误差和
'''Age and spending Score'''
X1 = df[['Age', 'Spending Score (1-100)', 'Gender_int', 'Annual Income (k$)']].iloc[:, :].values
inertia = []
algorithm = (KMeans(n_clusters=5, init='random', n_init=10, max_iter=300,
                    tol=0.0001, random_state=111, algorithm='elkan'))
algorithm.fit(X1)
inertia.append(algorithm.inertia_)
print(algorithm.inertia_)
# The Elbow Method手肘法则
'''Age and spending Score'''
X1 = df[['Age', 'Spending Score (1-100)', 'Gender_int', 'Annual Income (k$)']].iloc[:, :].values
inertia = []
for n in range(1, 11):
    algorithm = (KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300,
                        tol=0.0001, random_state=111, algorithm='elkan'))
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)
plt.figure(1, figsize=(15, 6))
plt.plot(np.arange(1, 11), inertia, 'o')
plt.plot(np.arange(1, 11), inertia, '-', alpha=0.5)
plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
plt.show()
algorithm = (KMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=300,
                    tol=0.0001, random_state=111, algorithm='elkan'))
algorithm.fit(X1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_
print(labels1)
plt.figure(1, figsize=(15, 7))
plt.clf()
plt.scatter(x='Age', y='Spending Score (1-100)', data=df, c=labels1,
            s=200)
plt.scatter(x=centroids1[:, 0], y=centroids1[:, 1], s=900, c='red', alpha=0.5)
plt.ylabel('Spending Score (1-100)'), plt.xlabel('Age')
plt.show()
