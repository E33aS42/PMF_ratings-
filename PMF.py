from __future__ import division
import numpy as np
import sys
import pandas as pd
from numpy.random import multivariate_normal as MvN
from matplotlib import pyplot as plt

"""
This file has to be run directly from the terminal using the following command line:
    > python PMF.py "ratings.csv"
where 'hw4_PMF.py' is the name of python file and "ratings.csv" contains the ratings database to analyze.
"""

try:
    assert len(sys.argv) > 1, "missing input argument"
    train_data = np.genfromtxt(sys.argv[1], delimiter=",")
except Exception as e:
    print(e)
    exit()

print('data_loaded')

# Parameters of the PMF objective function to maximize
lam = 2         # lambda shape parameter
sigma2 = 0.1    # variance
d = 5           # rank
iterations = 50 # number of iterations


LD = len(train_data)    # nb of data points

# Load data as dataframe
data = pd.DataFrame(train_data)

# Get index set of objects omega_ui rated by user i
omega_ui = {}
set_ui = set()
for user, obj in zip(data[0],data[1]):
    if user not in set_ui:
        set_ui.add(int(user))
        omega_ui[int(user)] = [int(obj)]
    else:
        omega_ui[int(user)].append(int(obj))
N1 = len(omega_ui.keys())

# Get index set of users omega_vj who rated object j
omega_vj = {}
set_vj = set()
for user, obj in zip(data[0],data[1]):
    if obj not in set_vj:
        set_vj.add(int(obj))
        omega_vj[int(obj)] = [int(user)]
    else:
        omega_vj[int(obj)].append(int(user))
N2 = len(omega_vj.keys())

# create dictionary with known data
Mij = {}
set_Mij = set()
for user, obj, mij in zip(data[0], data[1], data[2]):
    Mij[(int(user), int(obj))] = mij

# Initialisation of vj
def init_V():
    V = MvN(mean=np.zeros((N2)), cov=np.identity(N2) * 1 / lam, size = d)
    return V

# update ui
def update_ui(V,i):
    i += 1
    ui = np.zeros((5,1))
    VT = V.T

    # first sum
    sum1 = 0
    for j in omega_ui[i]:
        vjT = VT[j-1]
        vj = vjT.T
        sum1 += np.outer(vj,vjT)
    sum1 += lam*sigma2*np.identity(5)

    # second sum
    sum2 = 0
    for j in omega_ui[i]:
        vjT = VT[j-1]
        vj = vjT.T
        sum2 += Mij[(i, j)]*vj
    a = np.zeros((d,d))
    for k in range(d):
        a[k][0] = sum2[k]
    sum2 = a

    # product
    prod = np.dot(np.linalg.inv(sum1), sum2)
    for k in range(d):
        ui[k] = prod[k][0]

    return ui.reshape(-1)

# update vj
def update_vj(U,j):
    j += 1
    vj = np.zeros((5,1))
    # first sum
    sum1 = 0
    for i in omega_vj[j]:
        ui = U[i-1]
        uiT = ui.T
        sum1 += np.outer(ui,uiT)
    sum1 += lam*sigma2*np.identity(5)

    # second sum
    sum2 = 0
    for i in omega_vj[j]:
        ui = U[i-1]
        sum2 += Mij[(i, j)]*ui
    a = np.zeros((d,d))
    for k in range(d):
        a[k][0] = sum2[k]
    sum2 = a

    # product
    prod = np.dot(np.linalg.inv(sum1), sum2)
    for k in range(d):
        vj[k] = prod[k][0]

    return vj.reshape(-1)

# Calculate loss function
def loss(U,V):
    # first sum
    sum1 = 0
    VT = V.T
    for ij in Mij.keys():
        i = ij[0]
        j = ij[1]
        vjT = VT[j-1]
        sum0 = Mij[ij]- np.dot(U[i-1],vjT)
        sum1 += (sum0)**2
    sum1 *= 1/(2*sigma2)

    # second sum
    sum2 = 0
    for i in range(N1):
        sum2 += np.linalg.norm(U[i], ord=2)**2
    sum2*= lam / 2

    # third sum
    sum3 = 0
    for j in range(N2):
        sum3 += np.linalg.norm(VT[j], ord=2)**2
    sum3 *= lam / 2

    # global sum
    L = - sum1 - sum2 - sum3

    return L

def PMF(V):
    #Initialisation
    VT = np.zeros((N2,d))
    U = np.zeros((N1,d))

    # update ui
    for i in range(N1):
        U[i] = update_ui(V, i)


    # update vj
    for j in range(N2):
        VT[j] = update_vj(U, j)
    V = VT.T

    # Calculate loss function
    L = loss(U, V)

    return L, U, V


V = init_V()
L_save = []
for iteri in range(iterations):
    # print(iteri)
    L, U, V = PMF(V)
    VT=V.T
    L_save.append(L)
    if iteri == 9:
        np.savetxt("U-10.csv", U, delimiter=",")
        np.savetxt("V-10.csv", VT, delimiter=",")
    if iteri == 24:
        np.savetxt("U-25.csv", U, delimiter=",")
        np.savetxt("V-25.csv", VT, delimiter=",")
    if iteri == 49:
        np.savetxt("U-50.csv", U, delimiter=",")
        np.savetxt("V-50.csv", VT, delimiter=",")

np.savetxt("objective.csv", L_save, delimiter=",")


# Plot objective function
plt.plot(L_save)
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('Loss function')
plt.show()
