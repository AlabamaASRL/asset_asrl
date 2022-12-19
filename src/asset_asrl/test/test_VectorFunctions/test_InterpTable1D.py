import numpy as np
import asset_asrl as ast

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from scipy import interpolate


import unittest


vf        = ast.VectorFunctions
oc        = ast.OptimalControl
Args      = vf.Arguments

class test_InterpTable1D(unittest.TestCase):
    
    def Interp_test(self,n,Func,dFunc,d2Func,tstab,tscheck):
        
        
        Vts0 =[]
        Vts2 =[]
        
        for t in tstab:
            Vt0=np.zeros((n+1))
            Vt0[0]=t
            Vt0[1:n+1]=Func(t)
            Vts0.append(Vt0)
            
            Vt2=np.zeros((n+1))
            Vt2[n]=t
            Vt2[0:n]=Func(t)
            Vts2.append(Vt2)
            

                
        kind   = 'cubic'
        #print(Func(tstab))
        TabAx0 = vf.InterpTable1D(tstab,Func(tstab).T,axis=0,kind=kind)
        TabAx1 = vf.InterpTable1D(tstab,Func(tstab),axis=1,kind=kind)
        
        TabVt0 = vf.InterpTable1D(Vts0,tvar=0,kind=kind)
        TabVt2 = vf.InterpTable1D(Vts2,tvar=n,kind=kind)
        
        
        RetAx0 = TabAx0(tscheck)
        RetAx1 = TabAx1(tscheck)
        RetVt0 = TabVt0(tscheck)
        RetVt2 = TabVt2(tscheck)
        
        
        
        ErrAx1Ax0 = abs(RetAx1-RetAx0).max()
        ErrVt0Ax0 = abs(RetVt0-RetAx0).max()
        ErrVt2Ax0 = abs(RetVt2-RetAx0).max()
        
        
        self.assertLess(ErrAx1Ax0, 1.0e-13)
        self.assertLess(ErrVt0Ax0, 1.0e-13)
        self.assertLess(ErrVt2Ax0, 1.0e-13)
        
        
        Valtol   = 1.0e-6
        dValtol  = 1.0e-2
        d2Valtol = 5.0e-2
        
        
        
        
        Tabs=[TabAx0,TabAx1,TabVt0,TabVt2]
        
        for Tab in Tabs:
            TabFunc = Tab.vf()
            L = range(1,n+1)
            
            for t in tscheck:
                
                Val   = Func(t)
                dVal  = dFunc(t)
                d2Val = d2Func(t)
                
                
                Vali,dVali,d2Vali = Tab.interp_deriv2(t)
                
                #print(d2Vali,d2Val,t)

                
                Valerr = abs(Val-Vali).max()
                dValerr = abs(dVal-dVali).max()
                d2Valerr = abs(d2Val-d2Vali).max()
                
                self.assertLess(Valerr, Valtol)
                self.assertLess(dValerr, dValtol)
                self.assertLess(d2Valerr, d2Valtol)
                
                fx,jx,gx,hx = TabFunc.computeall([t],L)
                
                fxerr = abs(Vali-fx).max()
                jxerr= abs(dVali-jx.T).max()
                gxerr =abs(np.dot(L,dVali)-gx[0]).max()
                hxerr =abs(np.dot(L,d2Vali)-hx[0,0]).max()
                
                self.assertLess(fxerr, 1.0e-13)
                self.assertLess(jxerr, 1.0e-13)
                self.assertLess(gxerr, 1.0e-13)
                self.assertLess(hxerr, 1.0e-13)
        
        
        
    def test_EvenInterp(self):
        
        
        def  Func(t) :return np.array([ np.cos(t), np.sin(t)])
        def dFunc(t) :return np.array([-np.sin(t), np.cos(t)])
        def d2Func(t):return np.array([-np.cos(t),-np.sin(t)])
        
        
        tstab = list(np.linspace(0,2*np.pi,500))
        
        tscheck = np.linspace(0.01,2*np.pi,37)
        
        self.Interp_test(2,Func,dFunc,d2Func,tstab,tscheck)
        
        def  Func(t) :return np.array([ np.cos(t)])
        def dFunc(t) :return np.array([-np.sin(t)])
        def d2Func(t):return np.array([-np.cos(t)])

        self.Interp_test(1,Func,dFunc,d2Func,tstab,tscheck)

        
        
        
if __name__ == "__main__":
    unittest.main(exit=False)