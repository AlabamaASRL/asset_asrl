import numpy as np
import asset as ast


def FiniteDiffCheck(Fun,X,L, jsize=1.0e-6,hsize=1.0e-6):
    IRows = Fun.IRows()
    ORows = Fun.ORows()

def FDDerivChecker(Fun,X):
    IRows = Fun.IRows()
    ORows = Fun.ORows()
    L = np.random.rand((ORows))
    L = np.ones((ORows))
    def Value(X):
        return Fun.compute(X)
    
    def Adjoint(X):
        return Fun.adjointgradient(X,L)
    
    
    szs =np.array([1.0e-4,1.0e-5,1.0e-6,1.0e-7,1.0e-8,1.0e-9])
    print("---------------------------------------------")
    for s in szs:
        np.set_printoptions(precision  = 3, linewidth = 200)
        print("#############################################################")
        print("Step Size: ", s)
        F  = ast.VectorFunctions.PyVectorFunction(IRows,ORows,Value,s,s)
        FD = ast.VectorFunctions.PyVectorFunction(IRows,IRows,Adjoint,s,s)
        
        JF   = Fun.jacobian(X) 
        JFT  = F.jacobian(X) 
        Jerr = JF-JFT
        
        print("  Abs Max Jacobian Error: ", (abs(Jerr)).max(),", Rel Max Jacobian Error:",np.nanmax(abs(Jerr/(JFT+1.0e-18))))
        HF  = Fun.adjointhessian(X,L)
        HFT = FD.jacobian(X)
        HFT = 0.5*HFT + 0.5*HFT.T
        Herr = HF-HFT
        print("  Abs Max Hessian Error: ", (abs(Herr)).max(),", Rel Max Hessian Error:",np.nanmax(abs(Herr/(HFT+1.0e-18))))
        
        print("Raw Jacobian Error:")
        print(Jerr)
        print("Jacobian Value:")
        print(JFT)
        
        print("Raw Hessian Error:")
        print(Herr)
        print("Hessian Value:")
        print(HFT)
    
        #print(HF)
        #print(JF,HF)
    
