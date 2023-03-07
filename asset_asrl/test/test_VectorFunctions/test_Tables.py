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

class test_InterpTables(unittest.TestCase):
    
    def Interp1D_test(self,n,Func,dFunc,d2Func,tstab,tscheck):
        
        
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
        dValtol  = 1.0e-4
        d2Valtol = 1.0e-2
        
        
        
        
        Tabs=[TabAx0,TabAx1,TabVt0,TabVt2]
        
        for Tab in Tabs:
            TabFunc = Tab.vf()
            L = range(1,n+1)
            
            for t in tscheck:
                
                Val   = Func(t)
                dVal  = dFunc(t)
                d2Val = d2Func(t)
                
                
                Vali,dVali,d2Vali = Tab.interp_deriv2(t)
                

                
                Valerr = abs(Val-Vali).max()
                dValerr = abs(dVal-dVali).max()
                d2Valerr = abs(d2Val-d2Vali).max()
                
                #print(Valerr,dValerr,d2Valerr)
                
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
        
    def Interp2D_test(self,Func,dFunc,d2Func,xstab,ystab,xscheck,yscheck):
        
        X, Y = np.meshgrid(xstab, ystab)
        
        F = Func(X,Y)
                
        Tab = vf.InterpTable2D(xstab,ystab,F,kind='cubic')
        
        args = Args(2)
        
        sf1 = Tab(args)
        sf2 = Tab(args[0],args[1])
        sf3 = Tab(args[0]*1,args[1]*1)
        sf4 = Tab.sf()
        
        sfs = [sf1,sf2,sf3,sf4]
        lm = 2.0
        
        Valtol = 1.0e-6
        dValtol = 1.0e-4
        d2Valtol = 1.0e-2
        
        for i,x in enumerate(xscheck):
            for j,y in enumerate(yscheck):
                Val = Func(x,y)
                dVal = dFunc(x,y)
                d2Val = d2Func(x,y)
                
                for func in sfs:
                    xin = [x,y]
                    
                    fx,jx,gx,hx = func.computeall(xin,[lm])
                    fx1 = func.compute(xin)
                    jx1 = func.jacobian(xin)
                    
                    fxerr = abs(Val-fx).max()
                    jxerr= abs(dVal-jx.T).max()
                    gxerr =abs(dVal*lm-gx).max()
                    hxerr =abs(d2Val*lm-hx).max()
                    selffxerr = abs(fx1-fx).max()
                    selfjxerr= abs(jx1-jx).max()
                    
                    self.assertLess(fxerr, Valtol)
                    self.assertLess(jxerr, dValtol)
                    self.assertLess(gxerr, dValtol)
                    self.assertLess(hxerr, d2Valtol)
                    
                    self.assertLess(selffxerr, 1.0e-12)
                    self.assertLess(selfjxerr, 1.0e-12)
                    
    def Interp3D_test(self,Func,dFunc,d2Func,xstab,ystab,zstab,xscheck,yscheck,zscheck,cache = False,indexing = 'ij'):
        
        X,Y,Z = np.meshgrid(xstab, ystab,zstab,indexing='ij')
        
        F = Func(X,Y,Z)
                
        Tab = vf.InterpTable3D(xstab,ystab,zstab,F,kind='cubic',cache = cache)
        args = Args(3)
        
        sf1 = Tab(args)
        sf2 = Tab(args[0],args[1],args[2])
        sf3 = Tab(args[0]*1,args[1]*1,args[2]*1)
        sf4 = Tab.sf()
        
        sfs = [sf1,sf2,sf3,sf4]
        sfs = [sf4]
        lm = 2.0
        
        Valtol = 1.0e-6
        dValtol = 1.0e-4
        d2Valtol = 1.0e-2
        
        for i,x in enumerate(xscheck):
            for j,y in enumerate(yscheck):
                for k,z in enumerate(zscheck):
                    Val = Func(x,y,z)
                    dVal = dFunc(x,y,z)
                    d2Val = d2Func(x,y,z)
                    
                    for func in sfs:
                        xin = [x,y,z]
                        
                        fx1 = func.compute(xin)
                        jx1 = func.jacobian(xin)
                        
                        fx,jx,gx,hx = func.computeall(xin,[lm])
                       
                        fxerr = abs(Val-fx).max()
                        jxerr= abs(dVal-jx.T).max()
                        gxerr =abs(dVal*lm-gx).max()
                        hxerr =abs(d2Val*lm-hx).max()
                        
                        selffxerr = abs(fx1-fx).max()
                        selfjxerr= abs(jx1-jx).max()
                        
                        self.assertLess(fxerr, Valtol)
                        self.assertLess(jxerr, dValtol)
                        self.assertLess(gxerr, dValtol)
                        self.assertLess(hxerr, d2Valtol)  
                        
                        self.assertLess(selffxerr, 1.0e-12)
                        self.assertLess(selfjxerr, 1.0e-12)  
    
    def test_Interp1D(self):
    
        def  Func(t) :return np.array([ np.cos(t), np.sin(t)])
        def dFunc(t) :return np.array([-np.sin(t), np.cos(t)])
        def d2Func(t):return np.array([-np.cos(t),-np.sin(t)])
        
        n = 100
        
        tstab  = list(np.linspace(0,2*np.pi,n))
        tstabu = list(np.linspace(0,2*np.pi,n))
        tstabu.pop(int(n/4))
        tstabu.pop(int(n/3))
        tstabu.pop(int(n/2))
        tstabu.pop(int(2*n/3))
        tstabu.pop(int(3*n/4))

        tscheck = np.linspace(0.00,2*np.pi,37)
        self.Interp1D_test(2,Func,dFunc,d2Func,tstab,tscheck)
        self.Interp1D_test(2,Func,dFunc,d2Func,tstabu,tscheck)

        
        def  Func(t) :return np.array([ np.cos(t)])
        def dFunc(t) :return np.array([-np.sin(t)])
        def d2Func(t):return np.array([-np.cos(t)])

        self.Interp1D_test(1,Func,dFunc,d2Func,tstab,tscheck)
        self.Interp1D_test(1,Func,dFunc,d2Func,tstabu,tscheck)
        
    def test_Interp2D(self):
        
        def Func(x,y):
            return np.cos(x)*np.cos(y)
        def dFunc(x,y):
            return - np.array([np.sin(x)*np.cos(y),
                               np.cos(x)*np.sin(y)])
        def d2Func(x,y):
            row1 = - np.array([ np.cos(x)*np.cos(y),-np.sin(x)*np.sin(y)])
            row2 = - np.array([-np.sin(x)*np.sin(y), np.cos(x)*np.cos(y)])
            return  np.array([row1,row2])

        nx = 103
        ny = 111
        
        xs = np.linspace(-np.pi, np.pi*1.1,nx)
        ys = np.linspace(-1.33*np.pi, np.pi*.9,ny)
        
        xsu = list(np.copy(xs))
        xsu.pop(int(nx/4))
        xsu.pop(int(nx/3))
        xsu.pop(int(nx/2))
        xsu.pop(int(2*nx/3))
        xsu.pop(int(3*nx/4))
        
        
        ysu = list(np.copy(ys))
        ysu.pop(int(ny/5))
        ysu.pop(int(4*ny/5))
        
        xscheck = np.linspace(xs[0],xs[-1],50)
        yscheck = np.linspace(ys[0],ys[-1],60)
        
        
        with self.subTest("Even on Even"):
            self.Interp2D_test(Func,dFunc,d2Func,xs,ys,xscheck,yscheck)
            
        with self.subTest("Even on Uneven"):
            self.Interp2D_test(Func,dFunc,d2Func,xs,ysu,xscheck,yscheck)
            
        with self.subTest("Uneven on Even"):
            self.Interp2D_test(Func,dFunc,d2Func,xsu,ys,xscheck,yscheck)
            
        with self.subTest("Uneven on Uneven"):
            self.Interp2D_test(Func,dFunc,d2Func,xsu,ysu,xscheck,yscheck)
    
    def test_Interp3D(self):
        
        def Func(x,y,z):
            return np.cos(x)*np.cos(y)*np.cos(z)

        def dFunc(x,y,z):
            return - np.array([np.sin(x)*np.cos(y)*np.cos(z),
                               np.cos(x)*np.sin(y)*np.cos(z),
                               np.cos(x)*np.cos(y)*np.sin(z)])

        def d2Func(x,y,z):
            row1 = - np.array([ np.cos(x)*np.cos(y)*np.cos(z),-np.sin(x)*np.sin(y)*np.cos(z),-np.sin(x)*np.cos(y)*np.sin(z)])
            row2 = - np.array([-np.sin(x)*np.sin(y)*np.cos(z), np.cos(x)*np.cos(y)*np.cos(z),-np.cos(x)*np.sin(y)*np.sin(z)])
            row3 = - np.array([-np.sin(x)*np.cos(y)*np.sin(z),-np.cos(x)*np.sin(y)*np.sin(z), np.cos(x)*np.cos(y)*np.cos(z)])

            return  np.array([row1,row2,row3])
        
        
        nx = 103
        ny = 111
        nz = 101
        
        
        xs = np.linspace(-np.pi, np.pi*1.1,nx)
        ys = np.linspace(-1.33*np.pi, np.pi*.9,ny)
        zs = np.linspace(-1.1*np.pi, np.pi*1.06,nz)


        xsu = list(np.copy(xs))
        xsu.pop(int(nx/4))
        xsu.pop(int(nx/3))
        xsu.pop(int(nx/2))
        xsu.pop(int(2*nx/3))
        xsu.pop(int(3*nx/4))
        
        
        ysu = list(np.copy(ys))
        ysu.pop(int(ny/5))
        ysu.pop(int(4*ny/5))
        
        zsu = list(np.copy(zs))
        zsu.pop(int(nz/7)+1)
        zsu.pop(int(3*nz/5))
        
        
        xscheck = np.linspace(xs[0],xs[-1],20)
        yscheck = np.linspace(ys[0],ys[-1],23)
        zscheck = np.linspace(zs[0],zs[-1],19)
        
       

        with self.subTest("Even, Even, Even"):
            self.Interp3D_test(Func,dFunc,d2Func,xs,ys,zs,xscheck,yscheck,zscheck)
        
        with self.subTest("Even, Uneven, Even"):
            self.Interp3D_test(Func,dFunc,d2Func,xs,ysu,zs,xscheck,yscheck,zscheck)  
            
        with self.subTest("Uneven, Even, Uneven"):
            self.Interp3D_test(Func,dFunc,d2Func,xsu,ys,zsu,xscheck,yscheck,zscheck)
            
        with self.subTest("Even, Even, Even: Cached"):
            self.Interp3D_test(Func,dFunc,d2Func,xs,ys,zs,xscheck,yscheck,zscheck,cache=True)
        
        with self.subTest("Even, Uneven, Even: Cached"):
            self.Interp3D_test(Func,dFunc,d2Func,xs,ysu,zs,xscheck,yscheck,zscheck,cache=True)  
        
        

    
        
        


        


        
        
if __name__ == "__main__":
    unittest.main(exit=False)