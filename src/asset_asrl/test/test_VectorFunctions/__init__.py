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
        
        if(Verbose or max_selfadjointgrad_error>1.0e-12):
            print("Self adjoint_gradient Error:")
            print(selfadjointgrad_error)
         
        if(Verbose or max_hessian_error>maxherror):
            print("Hessian Error:")
            print(hessian_error)
            
        self.assertLess(max_selfadjointgrad_error,1.0e-12,"Adjoint gradients do not match")
        
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

    

    ######################################################
    
    def MatrixOps_impl(self, Ltype,lrows,lcols,Rtype,rrows,rcols):
        
        M1val = np.random.rand(lrows,lcols)
        M2val = np.random.rand(rrows,rcols)
        
        M1shift = np.random.rand(lrows,lcols)
        M2shift = np.random.rand(rrows,rcols)
        
        M1scale = np.random.uniform(0,1)
        M2scale = np.random.uniform(0,1)

        
        X = Args(lrows*lcols+rrows*rcols)
        
        M1 = Ltype(X.head(lrows*lcols),lrows,lcols)
        M2 = Rtype(X.tail(rrows*rcols),rrows,rcols)
        
        MProd = (M1*M1scale + M1shift)*(M2*M2scale +M2shift)
        
        MProdval_truth = np.dot(M1val*M1scale+M1shift,M2val*M2scale +M2shift).flatten("F")
        
        Xin = np.zeros((lrows*lcols+rrows*rcols))
        
        if(Ltype==vf.ColMatrix):
            Xin[0:lrows*lcols] = M1val.flatten("F")
        else:
            Xin[0:lrows*lcols] = M1val.flatten("C")
            
        if(Rtype==vf.ColMatrix):
            Xin[lrows*lcols:len(Xin)] = M2val.flatten("F")
        else:
            Xin[lrows*lcols:len(Xin)] = M2val.flatten("C")
        
        
        MProdval = MProd.vf()(Xin)
        
        ProdErr = MProdval-MProdval_truth
        
        with self.subTest("Output"):
            self.assertLess(abs(ProdErr).max(), 1.0e-12)
        
        Fun = MProd.vf()
        
        L = range(2,Fun.ORows()+2)
        
        with self.subTest("Derivatives"):
            self.function_test_impl(Fun, Xin, L,Verbose=False,maxjerror=1.0e-5,maxherror=1.0e-5)
    
    def test_MatrixOperations(self):
        
        for m1rows in range(1,10):
            for m1cols_m2rows in range(1,10):
                for m2cols in range(1,10):
                    with self.subTest(f" Col * Row {m1rows}, ({m1cols_m2rows},{m2cols})"):
                        self.MatrixOps_impl(vf.ColMatrix, m1rows, m1cols_m2rows, vf.RowMatrix, m1cols_m2rows, m2cols)
                    with self.subTest(f" Col * Col {m1rows}, ({m1cols_m2rows},{m2cols})"):
                        self.MatrixOps_impl(vf.ColMatrix, m1rows, m1cols_m2rows, vf.ColMatrix, m1cols_m2rows, m2cols)
                    with self.subTest(f" Row * Col {m1rows}, ({m1cols_m2rows},{m2cols})"):
                        self.MatrixOps_impl(vf.RowMatrix, m1rows, m1cols_m2rows, vf.ColMatrix, m1cols_m2rows, m2cols)
                    with self.subTest(f" Row * Row {m1rows}, ({m1cols_m2rows},{m2cols})"):
                        self.MatrixOps_impl(vf.RowMatrix, m1rows, m1cols_m2rows, vf.RowMatrix, m1cols_m2rows, m2cols)


                    
        
    ##########################################################
    
        





if __name__ == "__main__":
    unittest.main(exit=False)
