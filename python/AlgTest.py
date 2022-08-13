import numba
import numpy as np
import asset as ast
import time as time
import random



@numba.njit
def fun(x):
    return [xi*xi + 39 for xi in x] 


AF = ast.VectorFunctions.PyVectorFunction(3,3,fun)

AF.compute([1,1,1])

AF.rpt([1,1,1],100000)



