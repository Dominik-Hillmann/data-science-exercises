import numpy as np
import matplotlib.pyplot as plt

purchaseAmount = np.random.normal(50.0, 10.0, 1000)
pageSpeeds = np.random.normal(15.0, 5.0, 1000)
print(np.cov(purchaseAmount, pageSpeeds))
print(np.corrcoef(purchaseAmount, pageSpeeds))
plt.scatter(purchaseAmount, pageSpeeds)
plt.show()


purchaseAmount = purchaseAmount / pageSpeeds # now purchaseAmount is a function of pageSpeeds and therefore definitly correlated
plt.scatter(purchaseAmount, pageSpeeds)
plt.xlabel("purchaseAmount")
plt.ylabel("pageSpeeds")
print(np.cov(purchaseAmount, pageSpeeds))
print(np.corrcoef(pageSpeeds, purchaseAmount))
plt.show()
