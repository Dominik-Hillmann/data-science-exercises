"""
This data contains 961 instances of masses detected in mammograms, and contains the following attributes:

	BI-RADS assessment: 1 to 5 (ordinal) (discarded)
	Age: patient's age in years (integer)
	Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
	Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
	Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
	Severity: benign=0 or malignant=1 (binominal)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

# ***** Part 1: Data Preparation *****

colNames = ['BI_RADS', 'age', 'shape', 'margin', 'density', 'severity']
data = pd.read_csv(
	'https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data', 
	na_values = ['?'],
	names = colNames
)
print(data.head())

print(data.describe())
print()
print(data.shape)
print()
# print(data[data[['age']] > 50.0].dropna().shape)
data = data.dropna()
print(data.head())
print(data.shape)

# create NumPy arrays for the features
Y = (data[['severity']].values)#.flatten() cannot be flattened here because standard sclaer expects 2D array
X = data.drop(['BI_RADS', 'severity'], axis = 1).values
print('X shape: ' + str(X.shape))
print('Y shape: ' + str(Y.shape))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X) #
X = scaler.transform(X)
scaler.fit(Y)
Y = scaler.transform(Y)
print(X)
print(Y)

def randSplit(data, percent):
	from random import randint

	for i in range(len(data) - 1):
		if random() < percent:
			print(),