# -*- coding: utf-8 -*-
#########################################################
#   John McDonnell
#########################################################

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read in datasets
dataset = pd.read_csv('cardio_train.csv', sep=';')
dataset.describe()

######################
# 1. Research
######################
# Recreate the Neural Network classification model using MLPClassifier instead of TensorFlow

# Pre-processing including duplicates, feature engineering, and dropping unnecessary features
# Thanks to https://www.kaggle.com/vbmokin/20-models-for-cardiovascular-disease-prediction
dataset.drop('id',axis=1,inplace=True)
dataset.drop_duplicates(inplace=True)
dataset['bmi'] = dataset['weight'] / (dataset['height']/100)**2
out_filter = ((dataset['ap_hi']>250) | (dataset['ap_lo']>200))
dataset = dataset[~out_filter]
len(dataset)

X = dataset.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,12]]
y = dataset.iloc[:,11]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fit Neural Network to training set
from sklearn.neural_network import MLPClassifier
nn_model = MLPClassifier(solver = 'sgd', activation='relu', hidden_layer_sizes = (12,), learning_rate = 'constant')
nn_model = MLPClassifier()
nn_model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = nn_model.predict(X_test)

# Evaluate model against test set
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_true = y_test, y_pred = y_pred)
(confusion_matrix[0,0] + confusion_matrix[1,1]) / (confusion_matrix[0,0] + confusion_matrix[0,1] + confusion_matrix[1,0] + confusion_matrix[1,1])

# Using GridSearch to optimise model
# This can take over 15 mins so in the interest of time, perhaps skip this section.
#from sklearn.model_selection import GridSearchCV
#parameters = [{
#                'activation':['logistic', 'tanh', 'relu'],
#                'solver':['sgd', 'adam', 'lbfgs'], 
#                'learning_rate':['constant', 'invscaling', 'adaptive'],
#                'hidden_layer_sizes': [(100,), (12,), (12, 12)]
#                }]
#grid_search = GridSearchCV(estimator = nn_model, param_grid = parameters, scoring = 'r2', cv = 10, n_jobs = -1)
#grid_search = grid_search.fit(X_train, y_train)
#best_accuracy = grid_search.best_score_
#best_params = grid_search.best_params_


######################
# 3. Explortatory Data Analysis
######################
# 
# Read in dataset
dataset = pd.read_csv('cardio_train.csv', sep=';')
dataset.describe()
dataset.head()
dataset.info()

# Pre-processing including duplicates, feature engineering, and outliers
# Thanks to https://www.kaggle.com/vbmokin/20-models-for-cardiovascular-disease-prediction
dataset.drop('id',axis=1,inplace=True)
dataset.drop_duplicates(inplace=True)
out_filter = ((dataset['ap_hi']>250) | (dataset['ap_lo']>200))
dataset = dataset[~out_filter]
# Remove only impossible outliers such as negative values
dataset.drop(dataset[dataset['ap_lo'] <= 0].index, inplace = True)
dataset.drop(dataset[dataset['ap_hi'] <= 0].index, inplace = True)
len(dataset)
# age is in days, let's convert to years and add bmi feature
# thanks to https://www.kaggle.com/sulianova/eda-cardiovascular-data
dataset['years'] = (dataset['age'] / 365).round().astype('int')
sns.countplot(x='years', hue='cardio', data=dataset, palette='Set2')
dataset['bmi'] = dataset['weight'] / (dataset['height']/100)**2

# Check for target class imbalance
# Thanks to https://www.kaggle.com/vaibhavs2/cardiovascular-disease-complete-eda-modelling
fig, ax = plt.subplots(1,1)
sns.countplot(dataset['cardio'], ax = ax)
for i in ax.patches:
    height = i.get_height()
    ax.text(i.get_x()+i.get_width()/2,height,'{:.2f}'.format((i.get_height()/len(dataset['cardio']))*100,'%'))
plt.show()

# Generate a correlation matrix to find individual risk factors
# thanks to https://www.kaggle.com/sulianova/eda-cardiovascular-data
corr = dataset.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,annot = True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});


######################
# 4. Implementation
# We will implement this analysis in 4 stages
######################
# Part (i) Load data and perform pre-processing

X_cardio1 = dataset[dataset['cardio'] == 1]
X_cardio0 = dataset[dataset['cardio'] == 0]

# Apply PCA to reduce number of dimensions
# Thanks to Udemy - https://www.udemy.com/course/machinelearning/
# Machine Learning A-Z™: Hands-On Python & R In Data Science
X1 = X_cardio1.values[:,[0,1,2,3,4,5,6,7,8,9,10,12,13]]
X0 = X_cardio0.values[:,[0,1,2,3,4,5,6,7,8,9,10,12,13]]
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X1 = sc_X.fit_transform(X1)
X0 = sc_X.fit_transform(X0)
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 10)
X1 = pca.fit_transform(X1)
explained_variance1 = pd.DataFrame(data = pca.explained_variance_ratio_)
explained_variance1['cumulative'] = np.cumsum(explained_variance1[0])
X0 = pca.fit_transform(X0)
explained_variance0 = pd.DataFrame(pca.explained_variance_ratio_)
explained_variance0['cumulative'] = np.cumsum(explained_variance0[0])
# Plot of first 3 Principle Components of each dataset
from mpl_toolkits import mplot3d
ax = plt.axes(projection = '3d')
ax.scatter3D(X1[:,0], X1[:,1], X1[:,2], c='Red')
ax.scatter3D(X0[:,0], X0[:,1], X0[:,2], c='Green')

# Part (ii) Find the optimum number of clusters

# Find Optimal number of k-means clusters using elbow method
from sklearn.cluster import KMeans
wcss0 = []
wcss1 = []
for i in range(1, 11):
   kmeans0 = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10)
   kmeans1 = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10)
   kmeans0.fit(X0)
   kmeans1.fit(X1)
   wcss0.append(kmeans0.inertia_)
   wcss1.append(kmeans1.inertia_)
plt.plot(range(1, 11), wcss1)
plt.title('Elbow Method for patients with Cardiovascular Disease')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

plt.plot(range(1, 11), wcss0)
plt.title('Elbow Method for patients without Cardiovascular Disease')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Find Optimal number of clusters using Dendrogram
# This may take up to 55 mins to run so commented out - Fast cluster is used below for performance reason!
from scipy.cluster import hierarchy as sch
#dendrogram = sch.dendrogram(sch.linkage(X1, method='ward'))
#plt.title('Dendrogram of patients with Cardiovascular Disease')
#plt.xlabel('Patients')
#plt.ylabel('Euclidean Distance')
#plt.show()
# This may also take up to 55 mins to run!
#dendrogram = sch.dendrogram(sch.linkage(X0, method='ward'))
#plt.title('Dendrogram of patients without Cardiovascular Disease')
#plt.xlabel('Patients')
#plt.ylabel('Euclidean Distance')
#plt.show()

# To improve performance, I used a random sample of 20%
X0Samp = pd.DataFrame(X0).sample(6000)
dendrogram = sch.dendrogram(sch.linkage(X0Samp, method='ward'))
plt.title('Dendrogram of patients without Cardiovascular Disease (sampled)')
plt.xlabel('Patients')
plt.ylabel('Euclidean Distance')
plt.show()

# Using FastCluster to generate dendrograms (still takes 45 mins for each dendrogram)
# Patients with Cardiovascular
import fastcluster
dendrogram = sch.dendrogram(fastcluster.linkage(X1, method = 'ward'))
plt.title('Dendrogram of patients with Cardiovascular Disease')
plt.xlabel('Patients')
plt.ylabel('Euclidean Distance')
plt.show()
# Patients without Cardiovascular
dendrogram = sch.dendrogram(fastcluster.linkage(X0, method = 'ward'))
plt.title('Dendrogram of patients without Cardiovascular Disease')
plt.xlabel('Patients')
plt.ylabel('Euclidean Distance')
plt.show()

# Part (iii) Cluster data using 7 clusters, output initial plots and evaluate

# Apply k-means Algorithm to both datasets using 7 clusters
kmeans = KMeans(n_clusters = 7, init = 'k-means++', max_iter = 300, n_init = 10)
y_kmeansX1 = kmeans.fit_predict(X1)
kmeans1 = kmeans.fit(X1)
y_kmeansX0 = kmeans.fit_predict(X0)
kmeans0 = kmeans.fit(X0)

# Plot size of each cluster from each dataset
sns.countplot(x = y_kmeansX1, palette='Set2')
plt.title('Number of patients with Cardiovascular Disease per cluster')
plt.xlabel('Cluster Index')
plt.ylabel('Number of patients')
plt.show()
sns.countplot(x = y_kmeansX0, palette='Set2')
plt.title('Number of patients without Cardiovascular Disease per cluster')
plt.xlabel('Cluster Index')
plt.ylabel('Number of patients')
plt.show()

# Evaluate Clustering
from sklearn import metrics
from sklearn.metrics import pairwise_distances
# Evaluate clusters according to Silhouette Score
metrics.silhouette_score(X0, kmeans0.labels_, metric='euclidean')
metrics.silhouette_score(X1, kmeans1.labels_, metric='euclidean')
# Evaluate clusters according to Calinski-Harabasz Index
metrics.calinski_harabasz_score(X0, kmeans0.labels_)
metrics.calinski_harabasz_score(X1, kmeans1.labels_)
# Evaluate clusters according to Calinski-Harabasz Index
metrics.davies_bouldin_score(X0, kmeans0.labels_)
metrics.davies_bouldin_score(X1, kmeans1.labels_)


# Part (iv) Generate clustermap to interpret clusters

# Plot clustermap for patients with Cardiovascular disease
X1_clustered = pd.DataFrame(sc_X.fit_transform(X_cardio1), columns = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'years', 'bmi'])
X1_clustered['cluster'] = y_kmeansX1
X1_means_by_cluster = X1_clustered.groupby('cluster').mean()
sns.clustermap(X1_means_by_cluster.transpose()).fig.suptitle('Clustermap of patients with Cardiovascular Disease') # to swap rows and columns in clustermap
# Nice plot, would medians give a clearer separation between clusters?
X1_medians_by_cluster = X1_clustered.groupby('cluster').median()
sns.clustermap(X1_medians_by_cluster.transpose()).fig.suptitle('Clustermap of patients with Cardiovascular Disease') # to swap rows and columns in clustermap

# Plot clustermap for patients without Cardiovascular disease
X0_clustered = pd.DataFrame(sc_X.fit_transform(X_cardio0), columns = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'years', 'bmi'])
X0_clustered['cluster'] = y_kmeansX0
X0_means_by_cluster = X0_clustered.groupby('cluster').mean()
sns.clustermap(means_by_cluster.transpose()).fig.suptitle('Clustermap of patients without Cardiovascular Disease')
# Nice plot, would medians give a clearer separation between clusters?
X0_medians_by_cluster = X0_clustered.groupby('cluster').median()
sns.clustermap(X0_medians_by_cluster.transpose()).fig.suptitle('Clustermap of patients without Cardiovascular Disease')
