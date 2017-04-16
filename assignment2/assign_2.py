import pandas as pd
import numpy as np
import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


### Write function that accepts class distribution vector and returns Gini, Entropy, and Classification Error Impurity Measures
# Gini
def impurityMeasures(classdist):
    gini = 1 - sum([x**2 for x in classdist])
    ent = sum([x*math.log(x, 2) for x in classdist if x > 0.0])
    classification = 1 - max(classdist)
    impurityDict ={'Gini Impurity Measure': gini, 'Entropy Impurity Measure': ent, 'Classification Impurity Measure': classification}
    return(impurityDict)

print(impurityMeasures([0.2, 0.3, 0.25, 0.15, 0.1]))

# Plot y = x2, -5 \leq x \leq 5
# x = np.linspace(-5, 5, 1000)
# y = x**2
# plt.plot(x, y)
# plt.title("y = x^2")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# Plot Gini, Entropy, Classification on same plot
def twoClassGini(p0):
    p1 = 1 - p0
    # cd = [p0, p1]
    # tot = sum([x*x for x in cd])
    # print(tot)
    # gini = 1 - tot
    return(1 - sum([x**2 for x in [p0, p1]]))

def twoClassEntropy(p0):
    p1 = 1 - p0
    return(- sum([x*math.log(x, 2) for x in [p0, p1] if x > 0.0]))

def twoClassClassification(p0):
    p1 = 1 - p0
    return(1 - max([p0, p1]))

x = np.linspace(0, 1, 100)
ygini = [twoClassGini(xval) for xval in x]
yentropy = [twoClassEntropy(xval) for xval in x]
yclass = [twoClassClassification(xval) for xval in x]

# gini = plt.plot(x, ygini, label='Gini')
# entropy = plt.plot(x, yentropy, 'r', label='Entropy')
# misclassification = plt.plot(x, yclass, 'c', label='Misclassification')
# plt.legend(handles = [gini, entropy, misclassification], labels = ['Gini', 'Entropy', 'Misclassification'])
# plt.title("Impurity Measures")
# plt.xlabel("p0")
# plt.ylabel("Impurity")
# plt.show()


### Use Hw2.csv
data = pd.read_csv('/Users/mikaelajordan/Documents/PythonPractice/Python/assignment2/Hw2.csv')
print(data.shape)
print(data.head())
print(data['Class'].value_counts()[:])
p0 = data['Class'].value_counts()[1]
p1 = data['Class'].value_counts()[0]
# print('Entropy:', twoClassEntropy(p0))
# print('Entropy:', impurityMeasures([p0, p1])['Entropy Impurity Measure'])



print(data[('Gender' == 'M')]['Class'].value_counts())
print(data[('Gender' == 'F')]['Class'].value_counts())
