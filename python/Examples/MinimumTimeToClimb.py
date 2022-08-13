import numpy as np
import asset as ast
import matplotlib.pyplot as plt

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments


################### Non Dimensionalize ##################################
g0 = 32.2 
W  = 203000

Lstar = 100000.0     ## feet
Tstar = 60.0         ## sec
Mstar = W/g0         ## slugs

Vstar   = Lstar/Tstar
Fstar   = Mstar*Lstar/(Tstar**2)
Astar   = Lstar/(Tstar**2)
Rhostar = Mstar/(Lstar**3)
BTUstar = 778.0*Lstar*Fstar
Mustar  = (Lstar**3)/(Tstar**2)


Machs = np.array([0, 0.4, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8])

CLas  = np.array([3.44, 3.44 ,3.44 ,3.58, 4.44 ,3.44 ,3.01, 2.86 ,2.44])

CD0s  = np.array([0.013, 0.013, 0.013, 0.014, 0.031, 0.041, 0.039, 0.036, 0.035])

MachsE = np.array([0, 0.4, .799, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8])
Etas  =  np.array([0.54, 0.54,.54, 0.54, 0.75, 0.79, 0.78, 0.89, 0.93, 0.93])

tab = ast.VectorFunctions.InterpTable1D(MachsE,Etas,True)

Ms = np.linspace(0,1.8,550000)
CLs1 = [tab.interp_deriv2(M)[0][0] for M in Ms]
CLs2 = [tab.interp_deriv2(M)[1][0] for M in Ms]
CLs3 = [tab.interp_deriv2(M)[2][0] for M in Ms]

plt.plot(Ms,CLs1)
plt.plot(Ms,CLs2)

plt.scatter(MachsE,Etas)

plt.show()


########################################################################
