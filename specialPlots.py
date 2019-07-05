# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 15:50:43 2019

@author: abhinav.jhanwar
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# set url
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
# set feature names
names = ['class','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols',
         'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
# read data
data = pd.read_csv(url,names=names)


##################################### plot missing values
sns.set_style("whitegrid")
missing = data.isnull().sum()
missing = missing[missing>0]
missing.sort_values(inplace=True)
missing.plot.bar()


############################### plot counts of a particular feature
pd.value_counts(data['class']).plot.bar()


#################################### target variable distribution for a particular feature
##### both target and feature should be categorical/discrete
pd.crosstab(data.Alcohol, data['class']).plot(kind="bar",figsize=(20,6))
plt.title('Class(Target) frequency for alcohol')
plt.xlabel('Target- Class')
plt.ylabel('Frequency')
plt.savefig('targetFrequency.png')
plt.show()


