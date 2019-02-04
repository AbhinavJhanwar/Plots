
import matplotlib.pyplot as plt
import pandas
import numpy
from pandas.plotting import scatter_matrix
import seaborn as sns

# sns plots
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
names = ['class','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols',
         'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
data = pandas.read_csv(url,names=names)
sns.pairplot(data[names], size=2)
plt.show()

# scatter plot
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
names = ['class','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols',
         'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
data = pandas.read_csv(url,names=names)
sns.pairplot(data, x_vars=['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols',
         'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline'], 
                           y_vars='class', size=7, aspect=0.7, kind='scatter')
plt.show()

# bar plot
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
names = ['class','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols',
         'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
data = pandas.read_csv(url,names=names)
sns.factorplot(x='class', y='Alcohol', ci=None, kind='bar', data=data)
plt.xlabel("mean alcohol")
plt.show()


## sns heat plot
corr = data.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()

# Univariate Histograms
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'UI4AI', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
data.hist()
plt.show()



# Univariate Density Plots
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'UI4AI', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
plt.show()


# Box and Whisker Plots
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'UI4AI', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()


# Correction Matrix Plot
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'UI4AI', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
correlations = data.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# Scatterplot Matrix    
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'UI4AI', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
scatter_matrix(data)
plt.show()

# define figure
fig = plt.figure(figsize=(16,16))
# define subplots
for i in range(6):
    ax = fig.add_subplot(3,2,i+1)
    ax.set_xlabel(names[i])
    ax.set_ylabel(names[-1])
    ax.scatter(data[names[i]], data[names[-1]])
# show plots
plt.show()