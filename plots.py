# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 15:50:43 2019

@author: abhinav.jhanwar
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# set url
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
# set feature names
names = ['class','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols',
         'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
# read data
data = pd.read_csv(url,names=names)


##############################################################################
############################## PAIR PLOTS (SCATTER, BAR) #####################
##############################################################################
###### USING SNS PAIR PLOT
sns.pairplot(data[names], size=2)
plt.show()

###### USING PANDAS PAIR PLOT
scatter_matrix(data)
plt.show()


##############################################################################
############################## SCATTER PLOTS #################################
##############################################################################
####### USING SNS #########
## 1 ##
sns.lmplot(x='Alcohol', y='class', data=data)
plt.show()

## 2 ##
sns.pairplot(data, x_vars=['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols',
         'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline'], 
                           y_vars='class', size=7, aspect=0.7, kind='scatter')
plt.show()

####### USING MATPLOTLIB #######
# define figure
fig = plt.figure(figsize=(16,16))
# define subplots
for i in range(6):
    ax = fig.add_subplot(3,2,i+1)
    ax.set_xlabel(names[i])
    ax.set_ylabel(names[-1])
    ax.scatter(data[names[i]], data[names[-1]])
plt.show()


##############################################################################
############################## BAR PLOTS #####################################
##############################################################################
sns.factorplot(x='class', y='Alcohol', ci=None, kind='bar', data=data)
plt.xlabel("mean alcohol")
plt.show()


##############################################################################
######################## BOX AND WHISKER PLOTS ###############################
##############################################################################
data[names[2:4]].plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


##############################################################################
############################## CORRELATION MATRIX PLOTS ######################
##############################################################################
corr = data.corr()

####### USING SNS ############
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()

########## USING MATPLOTLIB ###########
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(names),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


##############################################################################
############################## DENSITY PLOTS #################################
##############################################################################
data[names[:2]].plot(kind='density', subplots=True, layout=(2,2), sharex=False)
plt.show()


##############################################################################
################################ COUNT PLOTS #################################
##############################################################################
# this will plot the count of a particular feature in the dataset
sns.countplot(x="class", data=data, palette="bright")
plt.show()


##############################################################################
################################# HISTOGRAM PLOTS ############################
##############################################################################
# Univariate Histograms
data[names[:2]].hist()
plt.show()

