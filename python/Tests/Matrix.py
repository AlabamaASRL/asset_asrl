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
    
    
    Cs = Args(18)
    
    C1 = vf.ColMatrix(Cs.head(9),3,3)
    C2 = vf.ColMatrix(Cs.tail(9),3,3)
    
    
    MM = C1*C2
    
    
    X = range(1,19)
    
    
    MM.rpt(X,1000000)
    
    #FDDerivChecker(MM,X)
    
    
    
    
    
    
    
    
    
    
    
    
    

