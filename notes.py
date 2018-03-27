import numpy as np # numpy ist lib füe mathematische Funktionen in Python
import pandas as pd # mit pandas lassen sich Datenstrukturen für die Analyse einfach erschaffen ähnlich derer aus R
import matplotlib.pyplot as plt # zeichnet Grafiken


# ***** PANDAS DATENSTRUKTUREN *****
df = pd.read_csv("PastHires.csv") # .csv mithilfe des paths einlesen
print(df[["Previous employers", "Hired"]]) # auswählen aller Elemente, bestimmte Eigenschaften mit Array passender Strings
smallDf = (df[["Previous employers", "Hired"]][5:])[:6] # zweites Auswahlelement sind die Zahlen der Beobachtungen
numEmploy = smallDf["Previous employers"].value_counts() # zählt wie oft jeder Wert in Previous employers vorkommt -> histogram
numEmploy.sort_values()
numEmploy.plot(kind = "bar")
plt.show() # histogram


# ***** NUMPY Funktionen
# from scipy import stats ==
import scipy.stats as st
incomes = np.random.normal(100.0, 20.0, 10000) # mit Numpy lassen sich sich Zufallsfunktionen aufrufen und Elemente daraus sampeln
# hier: Normalverteilung mit Stdabweichung von 20, 10 000 Elemente
print(np.mean(incomes)) # mittel aus np
print(np.median(incomes)) # Median aus np
print(stats.mode(incomes)) # Modalwert aber aus scipy
incomes = np.append(incomes, [1000000]) # so werden Werte an Numpy array gehängt, hier Outlier von 100000
x = np.arange(-3, 3, 0.01) # np array von -3 bis 3 in 0.01-Schritten


# ***** STANDARDABWEICHUNG *****
import matplotlib.pyplot as plt
# gegeben eines np Arrays kann man gleich Stdabweichung und Varianz raus ziehen
print(incomes.std())
print(incomes.var())


# ***** PERZENTILE *****
# ein Perzentil ist der Wert, der den eindimensionalen Datensetz in alpha, 1- alpha, alpha element [0, 1] Teil
np.percentile(incomes, 50) # ist Median
np.percentile(incomes, 90) # Wert der Daten in 0.9 und 0.1 teilt


# ***** MATPLOTLIB *****
from scipy.stats import norm
x = np.arange(-3, 3, 0.01)
plt.show(x, norm.pdf(x)) # plt.show() hat Argumente (x, f(x)), plottet jedes Element in np arr x und die Funktion dessen
def f(x):
    return x
def g(x):
    return x * x
plt.show(x, f(x))
# mehrere plots in einem Graph
plt.plot(x, f(x))
plt.plot(x, g(x))
plt.show() # alle Graphen sollten gleiche Input haben
plt.savefig('C:\\Users\\Frank\\MyPlot.png', format='png') # save img
# Änderung der Achsen
axes = plt.axes()
axes.set_xlim([-5, 5]) # Gezeigter Teil x
axes.set_ylim([0, 1.0]) # Gezeigter Teil y
axes.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]) # angezeigte Zahlen als Array
axes.grid() # ganz einfach ein Gitter an Ticks
# Änderung der Linien
plt.plot(x, f(x), "b--") # blaue Linie dashed
plt.plot(x, g(x), "r:") # gepunktete rote Linie
# Labels der Achsen, Legenden
plt.title('Student Locations')
plt.xlabel("Achse 1")
plt.ylabel("Achse 2")
plt.legend(['Sneetches', 'Gacks'], loc = 2) # loc 1 oben rechts, 2 oben links, ... bis 4
# kind of graphs
plt.bar(range(0,5), values, color= colors)
plt.scatter(X,Y)
plt.hist(incomes, 50)

values = [12, 55, 4, 32, 14]
colors = ['r', 'g', 'b', 'c', 'm']
explode = [0, 0, 0.2, 0, 0]
labels = ['India', 'United States', 'Russia', 'China', 'Europe']
plt.pie(values, colors = colors, labels = labels, explode = explode)


# ***** COVARIANCE, CORRELATION *****
# first, if you divide arrays of the same length, each element will be divided
np.cov(x, y) # returns cov Matriox
np.corrcoef # returns corr matrix
