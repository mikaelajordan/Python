import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
import os

# Change working directory to be able to import FTIC easily
os.chdir('/Users/mikaelajordan/Documents/PythonPractice/Python/assignment1')


### Import the FTIC.csv file as a pandas dataframe ###
ftic = pd.read_csv('./FTIC.csv', header=0)

### Find dimensions and print first few rows of DataFrame
print(ftic.shape)
# 9218 rows and 6 columns

print(ftic.columns)
# TERM,             YearMo
# QUARTER,          Quarter of Year (1st, 2nd, 3rd, 4th)
# PERCENTILE,       Percentile Rank of Student
# SAT,              Student SAT Score
# 1st_Spring,       Retained until 1st Spring?
# 2nd_Fall          Retained until 2nd Fall?

print(ftic.head())
# prints out first 5 columns of data so it is easy to inspect


### Create a retention vector for second Fall
retention = ftic['2nd_Fall']
# print(retention.shape)
# 9218,
# print(retention.head())
# Prints Y/N values
# print(retention.value_counts())
# 6162 were retained, 3056 were not

### Create vector for SAT and PERCENTILE, create histogram, summary, mean
sat = ftic['SAT']
rank = ftic['PERCENTILE']

# Histogram for Rank
# plt.hist(rank)
# plt.title("Histogram of Rank Vector with Automatic Bins")
# plt.xlabel("Rank Bins")
# plt.ylabel("Frequency")
# plt.show()

# Summary for Rank: Min, 1st Quartile, Median, Mean, 3rd Quartile, Max
# print('Minimum Rank:', min(rank))
# print('1st Quartile:', np.percentile(rank, 25))
# print('Median:', stat.median(rank))
# print('Mean:', rank.mean())
# print('3rd Quartile:', np.percentile(rank, 75))
# print('Maximum Rank:', max(rank))

# Histogram for SAT
# plt.hist(sat)
# plt.title("Histogram for SAT Vector with Automatic Bins")
# plt.xlabel("SAT Bins")
# plt.ylabel("Frequency")
# plt.show()

# Summary for SAT: Min, 1st Quartile, Median, Mean, 3rd Quartile, Max
# print('Minimum SAT Score:', min(sat))
# print('1st Quartile:', np.percentile(sat, 25))
# print('Median SAT Score:', stat.median(sat))
# print('Mean SAT Score:', sat.mean())
# print('3rd Quartile:', np.percentile(sat, 75))
# print('Maximum SAT Score:', max(sat))

### Make Retention a 1/0 vector, calculate total students retained and retention rate
retention = (retention=='Y')*1
# print('Total Number of Students Retained:', sum(retention))
# print('Retention Rate:', retention.mean())

### Basic statistics
# plt.scatter(rank, sat)
# plt.title("Rank vs SAT Score")
# plt.xlabel("RANK")
# plt.ylabel("SAT")
# plt.show()

# Creating a linear regression model
rank = rank[:, np.newaxis]
sat = sat[:, np.newaxis]
# print(rank.shape)
# print(sat.shape)
est = sm.OLS(sat, rank).fit()
# print(est.summary())

# print("Type for rank", type(rank))
# print("Type for SAT", type(sat))

# plt.scatter(rank, retention)
# plt.show()


#### Indexing for Admissions Requirements
newData = ftic[( (ftic['PERCENTILE'] < 25) & (ftic['SAT'] >= 1030) ) | ( (ftic['PERCENTILE'] >= 25) & (ftic['PERCENTILE'] < 50) & (ftic['SAT'] >= 950) ) |
( (ftic['PERCENTILE'] >= 50) & (ftic['PERCENTILE'] <90) & (ftic['SAT'] >= 400) ) | (ftic['PERCENTILE'] >= 90)]

# print(newData.head())
print(newData.shape)

newRetention = newData['2nd_Fall']
newRetention = (newRetention == 'Y')*1
print(newRetention.value_counts())
print(newRetention.mean())


### Create function to evaluate admissions standards
def evalThresholds(rankName, satName, retentionName, dataFrame, rankThresholds, satThresholds, topRank):
    newData = dataFrame[( (dataFrame[rankName] < rankThresholds[0]) & (dataFrame[satName] >= satThresholds[0]) ) |
    ((dataFrame[rankName] >= rankThresholds[0]) & (dataFrame[rankName] < rankThresholds[1]) & (dataFrame[satName] >= satThresholds[1]) ) |
    ((dataFrame[rankName] >= rankThresholds[1]) & (dataFrame[rankName] < topRank) & (dataFrame[satName] >= satThresholds[2]) ) |
    (dataFrame[rankName] >= topRank)]

    newRetention = (newData[retentionName] == 'Y')*1
    enrollmentLoss = (dataFrame.shape[0] - newData.shape[0])/dataFrame.shape[0]
    newRetentionRate = newRetention.mean()

    return([enrollmentLoss, newRetentionRate])

print(evalThresholds('PERCENTILE', 'SAT', '2nd_Fall', ftic, [25, 50], [1030, 950, 400], 90))
print(evalThresholds('PERCENTILE', 'SAT', '2nd_Fall', ftic, [33, 50], [1610, 950, 400], 90))

# for(i in range(90))
