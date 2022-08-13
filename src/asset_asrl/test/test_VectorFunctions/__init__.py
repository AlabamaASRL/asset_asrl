import numpy as np
import asset as ast
import unittest

vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments

class test_VectorFunctions(unittest.TestCase):
    
    def function_test_impl(self,Fun,X,L,
                           jsize=1.0e-6,hsize=1.0e-6,
                           maxjerror=.0001,maxherror=.0001,
                           PyImpl=None,Verbose=False):
        
        np.set_printoptions(precision  = 3, linewidth = 200)

        
        IRows = Fun.IRows()
        ORows = Fun.ORows()
        
        def Value(X):
            return Fun.compute(X)
        def Adjoint(X):
            return Fun.adjointgradient(X,L)
        
        JacFun  = ast.VectorFunctions.PyVectorFunction(IRows,ORows,Value,jsize,jsize)
        HessFun = ast.VectorFunctions.PyVectorFunction(IRows,IRows,Adjoint,hsize,hsize)
        
        
        
        fx,jx,gx,hx = Fun.computeall(X,L)
        
        jx2 = JacFun.jacobian(X)
        hx2 = HessFun.jacobian(X)
        
        hx2 = (hx2 + hx2.T)/2.0
        
        
        selfadjointgrad_error =np.dot(jx.T,L) - gx
        max_selfadjointgrad_error = (abs(selfadjointgrad_error)).max()
        
        jacobian_error = jx-jx2
        max_jacobian_error = (abs(jacobian_error)).max()

        hessian_error  = hx-hx2
        max_hessian_error = (abs(hessian_error)).max()
        
        if(Verbose or max_selfadjointgrad_error>1.0e-14):
            print("Self adjoint_gradient Error:")
            print(selfadjointgrad_error)
         
        if(Verbose or max_hessian_error>maxherror):
            print("Hessian Error:")
            print(hessian_error)
            
        self.assertLess(max_selfadjointgrad_error,1.0e-14,"Adjoint gradients do not match")
        
        with self.subTest("Jacobian"):
            if(Verbose or max_jacobian_error>maxjerror):
                print("Jacobian Error:")
                print(jacobian_error) 
            
            self.assertLess(max_jacobian_error,maxjerror,"Jacobians do not Match")
            
        
        self.assertLess(max_hessian_error,maxherror,"Hessians do not match")
        
        
        
        
        
        
    def test_Args(self):
        
        for n in range(1,20):
            Fun = Args(n).normalized()
            X   = range(1,n+1)
            L   = range(2,n+2)
            with self.subTest(n=n):
                self.function_test_impl(Fun, X, L,Verbose=False)

    def test_dotproduct(self):
        s=3
        
        





if __name__ == "__main__":
    unittest.main(exit=False)
