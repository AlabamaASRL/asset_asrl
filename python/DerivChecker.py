import numpy as np
import asset as ast

def FDDerivChecker(Fun,X):
    
    IRows = Fun.IRows()
    ORows = Fun.ORows()
    L = np.ones((ORows))*2
    L = np.array(range(2,ORows+2))
    
    def Value(X):
        return Fun.compute(X)
    
    def Adjoint(X):
        return Fun.adjointgradient(X,L)
    
    
    szs =np.array([1.0e-4,1.0e-5,1.0e-6,1.0e-7,1.0e-8,1.0e-9])
    print("---------------------------------------------")
    for s in szs:
        np.set_printoptions(precision  = 3, linewidth = 200)
        print("Step Size: ", s)
        F = ast.VectorFunctions.PyVectorFunction(IRows,ORows,Value,s,s)
        FD = ast.VectorFunctions.PyVectorFunction(IRows,IRows,Adjoint,s,s)
        
        JF = Fun.jacobian(X) 
        JFT = F.jacobian(X) 
        Jerr = JF-JFT
        
        print("  Abs Max Jacobian Error: ", (abs(Jerr)).max(),", Rel Max Jacobian Error:",np.nanmax(abs(Jerr/(JF+1.0e-12))))
        HF  = Fun.adjointhessian(X,L)
        HFT = FD.jacobian(X)
        HFT = 0.5*HFT + 0.5*HFT.T
        Herr = HF-HFT
        print("  Abs Max Hessian Error: ", (abs(Herr)).max(),", Rel Max Hessian Error:",np.nanmax(abs(Herr/(HF+1.0e-12))))
        #print(HFT)
        print("--")
        print(Herr)
        #print(JF,HF)
    