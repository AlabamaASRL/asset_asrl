import asset as ast
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import time
from scipy.spatial.transform import Rotation as R

from DerivChecker import FDDerivChecker

vf    = ast.VectorFunctions
Args = vf.Arguments





if __name__ == "__main__":
    n=18
    XX = Args(n)
    f1 =  XX.head3().normalized()
    f2 =  XX.segment3(3).normalized()
    f3 =  XX.tail3().normalized()
    
    fs = [f1,f2,f3,f1.dot(f2),f3+f2, f1.cross(f2),f1-f2,XX[1]/XX[2]]
    
    #fs = [f1,f2,f3,f3,f1,f2,f3]
    F1 = vf.stack(fs)
    F2 =vf.DynamicStackTest(fs)
    
    F1.SpeedTest(np.ones((n)),1000000)
    F2.SpeedTest(np.ones((n)),1000000)
    
    X = Args(4)
    x = [1,0,0,0]
    
    M = R.from_euler('ZXZ', [45,0,0],True).as_matrix()
   
    V = X.head(3).normalized()
    
    f  = vf.matmul(M,V)
    f2 = vf.cross(M[0],V)
    f3 = X[0]*np.array([1,1,1])
    
    '''
    print(f3)
    FDDerivChecker(f3,x)
    
    f.SuperTest(x,1000000)
    f2.SuperTest(x,1000000)
    f3.SuperTest(x,1000000)

    
    V.SuperTest(x,1000000)
    '''
    
    
    
   
    

