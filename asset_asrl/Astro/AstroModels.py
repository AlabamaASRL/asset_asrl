import asset as ast
import numpy as np
import asset_asrl.Astro.Constants as c
from   asset_asrl.Astro.Extensions.TwoBodyFrame import TwoBodyFrame
from   asset_asrl.Astro.Extensions.NBodyFrame   import NBodyFrame
from   asset_asrl.Astro.Extensions.CR3BPFrame   import CR3BPFrame
from   asset_asrl.Astro.Extensions.EPPRFrame    import EPPRFrame
from   asset_asrl.Astro.Extensions.MEETwoBodyFrame   import MEETwoBodyFrame
from   asset_asrl.Astro.Extensions.ThrusterModels   import SolarSail,LowThrustAcc,CSIThruster
from   asset_asrl.OptimalControl import ODEBase


vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments


###############################################################################
'''
Ballistic Models
'''

class TwoBody(oc.ode_6.ode,TwoBodyFrame):
    def __init__(self,P1mu,lstar):
        TwoBodyFrame.__init__(self,P1mu,lstar)
        args = oc.ODEArguments(6,0)
        r =args.head3()
        v =args.segment3(3)
        odeeq = self.TwoBodyEOMs(r,v,otherAccs=[],otherEOMs=[])
        oc.ode_6.ode.__init__(self,odeeq,6)
        
class NBody(ODEBase,TwoBodyFrame):
    def __init__(self,Frame, ActiveAltBodies = 'All', Enable_J2=False,Enable_P1_Acc=True):
        
        self.Frame = Frame
        TwoBodyFrame.__init__(self,Frame.P1mu,Frame.lstar)
        
        args = oc.ODEArguments(6,0)
        r =args.head3()
        v =args.segment3(3)
        t= args[6]
        odeeq = self.Frame.NBodyEOMs(r,v,t,
                                     otherAccs=[],
                                     otherEOMs=[],
                                     ActiveAltBodies = ActiveAltBodies, 
                                     Enable_J2=Enable_J2,
                                     Enable_P1_Acc=Enable_P1_Acc)
        ODEBase.__init__(self,odeeq,6)
        
class CR3BP(oc.ode_6.ode,CR3BPFrame):
    def __init__(self,P1mu,P2mu,lstar):
        CR3BPFrame.__init__(self,P1mu,P2mu,lstar)
        ###################################
        args = oc.ODEArguments(6,0)
        r = args.XVec().head3()
        v = args.XVec().tail3()
        ode = self.CR3BPEOMs(r,v,
                             otherAccs=[],
                             otherEOMs=[])
        oc.ode_6.ode.__init__(self,ode,6)
        ###################################
        
class EPPR(ODEBase,CR3BPFrame):
    def __init__(self,Frame, ActiveAltBodies = 'All', Enable_J2=False):
        
        self.Frame = Frame
        CR3BPFrame.__init__(self,self.Frame.P1mu,self.Frame.P2mu,self.Frame.lstar)
        
        args = oc.ODEArguments(6,0)
        r =args.head3()
        v =args.segment3(3)
        t =args[6]
        odeeq = self.Frame.EPPREOMs(r,v,t,
                                     otherGaccs=[],
                                     otherAccs=[],
                                     otherEOMs=[],
                                     ActiveAltBodies = ActiveAltBodies, 
                                     Enable_J2=Enable_J2)
        ODEBase.__init__(self,odeeq,6)   
    
###############################################################################
###############################################################################
'''
Simple Low Thrust Models
'''        
class TwoBody_LT(ODEBase,TwoBodyFrame):
    def __init__(self,P1mu,lstar,thruster=LowThrustAcc()):
        self.thruster=thruster
        TwoBodyFrame.__init__(self,P1mu,lstar)
        ###################################
        args = oc.ODEArguments(6,3)
        r = args.head3()
        v = args.segment3(3)
        u = args.tail3()
        otherAccs = [self.thruster.ThrustExpr(u,self.astar)]

        odeeq = self.TwoBodyEOMs(r,v,
                                    otherAccs=otherAccs,
                                    otherEOMs=[])
        ODEBase.__init__(self,odeeq,6,3) 
        ###################################

class MEETwoBody_LT(ODEBase,MEETwoBodyFrame):
    def __init__(self,P1mu,lstar,thruster=LowThrustAcc()):
        self.thruster=thruster
        MEETwoBodyFrame.__init__(self,P1mu,lstar,HasControl=True,HasMass=False)
        ###################################
        args = oc.ODEArguments(6,3)
        X = args.head(6)
        u = args.tail3()
        otherAccs = [self.thruster.ThrustExpr(u,self.astar)]

        odeeq = self.MEETwoBodyFrame(X,
                                    otherAccs=otherAccs,
                                    otherEOMs=[])
        ODEBase.__init__(self,odeeq,6,3) 
        ###################################
        
class NBody_LT(ODEBase,TwoBodyFrame):
    def __init__(self,Frame,thruster=LowThrustAcc(), ActiveAltBodies = 'All', Enable_J2=False):
        self.thruster=thruster
        self.Frame = Frame
        TwoBodyFrame.__init__(self,Frame.P1mu,Frame.lstar)
        ###################################
        args = oc.ODEArguments(6,3)
        r =args.head3()
        v =args.segment3(3)
        t= args[6]
        u = args.tail3()
        otherAccs = [self.thruster.ThrustExpr(u,self.astar)]

        odeeq = self.Frame.NBodyEOMs(r,v,t,
                                     otherAccs=otherAccs,
                                     otherEOMs=[],
                                     ActiveAltBodies = ActiveAltBodies, 
                                     Enable_J2=Enable_J2)
        ODEBase.__init__(self,odeeq,6,3) 
        ###################################
        
class CR3BP_LT(ODEBase,CR3BPFrame):
    def __init__(self,P1mu,P2mu,lstar,thruster=LowThrustAcc()):
        self.thruster=thruster
        CR3BPFrame.__init__(self,P1mu,P2mu,lstar)
        ###################################
        args = oc.ODEArguments(6,3)
        r = args.XVec().head3()
        v = args.XVec().tail3()
        u = args.tail3()
        otherAccs = [self.thruster.ThrustExpr(u,self.astar)]

        odeeq = self.CR3BPEOMs(r,v,
                                    otherAccs=otherAccs,
                                    otherEOMs=[])
        ODEBase.__init__(self,odeeq,6,3) 
        ###################################
        
class EPPR_LT(ODEBase,CR3BPFrame):
    def __init__(self,Frame,thruster=LowThrustAcc(), ActiveAltBodies = 'All', Enable_J2=False):
        self.thruster=thruster
        self.Frame = Frame
        CR3BPFrame.__init__(self,self.Frame.P1mu,self.Frame.P2mu,self.Frame.lstar)
        ###################################
        args = oc.ODEArguments(6,3)
        r =args.head3()
        v =args.segment3(3)
        t =args[6]
        
        u = args.tail3()
        otherAccs = [self.thruster.ThrustExpr(u,self.astar)*self.Frame.AccscaleFunc.eval(t)]
        
        odeeq = self.Frame.EPPREOMs(r,v,t,
                                     otherGaccs=[],
                                     otherAccs=otherAccs,
                                     otherEOMs=[],
                                     ActiveAltBodies = ActiveAltBodies, 
                                     Enable_J2=Enable_J2)
        ODEBase.__init__(self,odeeq,6,3)   
        ###################################

###############################################################################
###############################################################################
class MEETwoBody_CSI(ODEBase,MEETwoBodyFrame):
    def __init__(self,P1mu,lstar,CSIthrust):
        self.thruster=CSIthrust
        MEETwoBodyFrame.__init__(self,P1mu,lstar,HasControl=True,HasMass=True)
        ###################################
        args = oc.ODEArguments(7,3)
        X = args.head(6)
        m = args[6]
        
        u = args.tail3()
        otherAccs = [self.thruster.ThrustExpr(u,m,self.astar)]
        otherEOMs = [self.thruster.MdotExpr(u,self.tstar)]

        odeeq = self.MEETwoBodyEOMs(X,
                                    otherAccs=otherAccs,
                                    otherEOMs=otherEOMs)
        ODEBase.__init__(self,odeeq,7,3) 
        ###################################
        
###############################################################################        
###############################################################################
'''
Solar Sail Models, Currently assuming sun is P1
'''   

class TwoBody_SolarSail(ODEBase,TwoBodyFrame):
    def __init__(self,P1mu=c.MuSun,lstar=c.AU,SailModel = SolarSail(.02,False)):
        self.SailModel = SailModel
        TwoBodyFrame.__init__(self,P1mu,lstar)
        ###################################
        args = oc.ODEArguments(6,3)
        r = args.head3()
        v = args.segment3(3)
        u = args.tail3()
        otherAccs = [ self.SailModel.ThrustExpr(r,u,self.mu)]

        odeeq = self.TwoBodyEOMs(r,v,otherAccs=otherAccs,otherEOMs=[])
        ODEBase.__init__(self,odeeq,6,3) 
        ###################################
        
class NBody_SolarSail(ODEBase,TwoBodyFrame):
    def __init__(self,Frame,SailModel = SolarSail(.02,False), ActiveAltBodies = 'All', Enable_J2=False):
        self.SailModel = SailModel
        self.Frame = Frame
        TwoBodyFrame.__init__(self,Frame.P1mu,Frame.lstar)
        ###################################
        args = oc.ODEArguments(6,3)
        r =args.head3()
        v =args.segment3(3)
        t= args[6]
        u = args.tail3()
        otherAccs = [ self.SailModel.ThrustExpr(r,u,self.mu)]
        odeeq = self.Frame.NBodyEOMs(r,v,t,
                                     otherAccs=otherAccs,
                                     otherEOMs=[],
                                     ActiveAltBodies = ActiveAltBodies, 
                                     Enable_J2=Enable_J2)
        ODEBase.__init__(self,odeeq,6,3) 
        ###################################
        
class CR3BP_SolarSail(ODEBase,CR3BPFrame):
    def __init__(self,mu1=c.MuSun,mu2=c.MuEarth, lstar=c.AU, SailModel = SolarSail(.02,False)):
        
        CR3BPFrame.__init__(self,mu1,mu2,lstar)
        self.SailModel = SailModel
        
        ####################################
        args = oc.ODEArguments(6,3)
        r = args.XVec().head3()
        v = args.XVec().tail3()
        u = args.tail3()
        thrust = self.SailModel.ThrustExpr(r-self.P1,u,1.0-self.mu)
        ode = self.CR3BPEOMs(r,v,otherAccs=[thrust],otherEOMs=[])
        ODEBase.__init__(self,ode,6,3)
        #####################################
        func = self.vf().eval(vf.stack([Args(7),(Args(7).head(3)-self.P1).normalized()]))
        self.CalcSubPoints(func)
        
        
class CR3BP_SolarSail_ZeroAlpha(ODEBase,CR3BPFrame):
    def __init__(self,mu1=c.MuSun,mu2=c.MuEarth, lstar=c.AU, SailModel = SolarSail(.02,False)):
        
        CR3BPFrame.__init__(self,mu1,mu2,lstar)
        self.SailModel = SailModel
        
        ####################################
        args = oc.ODEArguments(6,0)
        r = args.XVec().head3()
        v = args.XVec().tail3()
        
        thrust = (r-self.P1).normalized_power3()*self.SailModel.Normalbeta*(1.0-self.mu)
        ode = self.CR3BPEOMs(r,v,otherAccs=[thrust],otherEOMs=[])
        ODEBase.__init__(self,ode,6)
        #####################################
        self.CalcSubPoints(self.vf())
        
    def GenSubL1Lissajous(self,xnd,znd,phideg,psideg,nplanrev,npo,t0 = 0):
        func = self.vf()
        X = np.zeros((7))
        X[0:3] = self.SubL1
        return self.GenLissajousImpl(func,X,xnd,znd,phideg,psideg,nplanrev,npo,t0)
    def GenSubL2Lissajous(self,xnd,znd,phideg,psideg,nplanrev,npo,t0 = 0):
        func = self.vf()
        X = np.zeros((7))
        X[0:3] = self.SubL2
        return self.GenLissajousImpl(func,X,xnd,znd,phideg,psideg,nplanrev,npo,t0)
    
    
class EPPR_SolarSail(ODEBase,CR3BPFrame):
    def __init__(self,Frame,SailModel = SolarSail(.02,False), ActiveAltBodies = 'All', Enable_J2=False):
        self.SailModel = SailModel
        self.CR3BP_ZeroAlpha=CR3BP_SolarSail_ZeroAlpha(SailModel = self.SailModel)
        self.Frame = Frame
        CR3BPFrame.__init__(self,self.Frame.P1mu,self.Frame.P2mu,self.Frame.lstar)
        ###################################
        nargs = oc.ODEArguments(6,3)
        r =nargs.head3()
        v =nargs.segment3(3)
        t =nargs[6]
        u = nargs.tail3()
        
        f1 = self.SailModel.ThrustExpr(r-self.P1,u,1.0-self.mu)
        f2 = self.SailModel.ThrustExpr(Args(6).head3(),Args(6).tail3(),1.0-self.mu).eval(vf.Stack([r-self.P1,u]))
        #f2 = self.SailModel.ThrustExpr(Args(6).head3(),Args(6).tail3(),1.0-self.mu).eval(r-self.P1,u)

        otherGaccs = [ f2 ]
        
        odeeq = self.Frame.EPPREOMs(r,v,t,
                                     otherGaccs=otherGaccs,
                                     otherAccs=[],
                                     otherEOMs=[],
                                     ActiveAltBodies = ActiveAltBodies, 
                                     Enable_J2=Enable_J2)
        ODEBase.ode.__init__(self,odeeq,6,3)   
        self.CalcSubPoints(func = self.CR3BP_ZeroAlpha.vf())
        ###################################
    
