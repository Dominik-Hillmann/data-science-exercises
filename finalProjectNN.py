"""
This data contains 961 instances of masses detected in mammograms, and contains the following attributes:

	BI-RADS assessment: 1 to 5 (ordinal) (discarded)
	Age: patient's age in years (integer)
	Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
	Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
	Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
	Severity: benign=0 or malignant=1 (binominal)

"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop