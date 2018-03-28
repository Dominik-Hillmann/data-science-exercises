# for loop mit verschiedenen Polynomen und Graphen

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(4) # same random data every time

syntaxTest = np.random.normal(50.0, 10.0, 1000)
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = syntaxTest / pageSpeeds

for i in range(3):
    print(pageSpeeds[i])
    print((syntaxTest / pageSpeeds)[i] == purchaseAmount[i])

# Try different polynomial orders. Can you get a better fit with higher orders?
# Do you start to see overfitting, even though the r-squared score looks good for
# this particular data set?

plt.scatter(pageSpeeds, purchaseAmount)
plt.show()

polyn2 = np.polyfit(pageSpeeds, purchaseAmount, 2) # returns coefficients, highest power first, so 2, 1, intercept
print(polyn2) # np.poly

# def polyn(x, coeff):
#     n = len(coeff)
#     re = 0
#     for i in range(n):
#         re += coeff[n - i] * (x ** (n - i))
#         # coeff[0] * (x ** 2) + coeff[1] * x + coeff[2]
#     return re
#
# print(polyn(5, [2, 4])) # testing: (2 * 2 ** 2) + (4 * 2 ** 1) + (6 * 2 ** 0)
