import numpy as np
import math
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import Legendre

import asset_asrl as ast

vf = ast.VectorFunctions

Args = vf.Arguments


order =2
m = 1
k = order+1

coeefs = np.zeros(order)
coeefs[-1]=1

lpol = Legendre(coeefs).deriv()
tsn = [-1.0]+list(lpol.roots())+[1.0]
cc = 1/((2**k)*math.factorial(k-m)*math.factorial(m-1))

print(tsn)
xs = np.linspace(-1,1,100000)
ms = []
for x in xs:
    roots = [x for i in range(0,m-1)]
    roots+=list(tsn)
    p = Polynomial.fromroots(roots)
    q = p.integ()
    ms.append(abs(q(x)-q(-1)))
   


print(cc*max(ms))