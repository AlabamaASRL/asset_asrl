import numpy as np
import asset as ast

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments

class MEETwoBodyFrame():
    """
    Two body dynamics

    Attributes
    ----------
    mu : float
        system non-dimensional gravity parameter
    P1mu : float
        system dimensional gravity parameter
    lstar : float
        system characteristic length
    tstar : float
        system characteristic time
    vstar : float
        system characteristic velocity
    astar : float
        system characteristic acceleration
    mustar : float
        system characteristic gravity parameter
    """
    def __init__(self,P1mu,lstar,HasControl=False,HasMass=False):
        """
        TwoBodyFrame init function. Be sure that P1mu and lstar are of the same units.

        Parameters
        ----------
        P1mu : float
            gravitational constant for primary
        lstar : float
            characteristic length of system
        """
        self.mu = 1
        self.P1mu   = P1mu
        self.lstar  = lstar
        self.tstar  = np.sqrt((lstar**3)/(P1mu))
        self.vstar  = lstar/self.tstar
        self.astar  = (P1mu)/(lstar**2)
        self.mustar = (P1mu)
        self.HasControl = HasControl
        self.HasMass = HasMass
    def MEETwoBodyEOMs(self,X,otherAccs=[],otherEOMs=[]):
        """
        Two body equations of motion

        Parameters
        ----------
        X : ASSET VectorFunction
            6x1 Particle modified equinictual elemets [p,f,g,h,k,L]
        otherAccs : list
            list of other accelerations in RTN Frame to add to model
        otherEOMS : list
            list of other equations of motion driving model

        Returns
        -------
        Full equations of motion : ASSET VectorFunction
            All equations of motion for two body model

        """
        if(len(otherAccs)>1):    
            accs = vf.sum(otherAccs)
        else:
            accs = otherAccs[0]
        MEEdot   = ast.Astro.ModifiedDynamics(1.0).eval(vf.stack([X,accs]))

      
        return vf.stack([MEEdot]+otherEOMs)
    
    def RTNtoCartesianFunc(self):
        R,V,U = Args(9).tolist([(0,3),(3,3),(6,3)])

        ## Three orthonormal basis vectors
        Rhat = R.normalized()
        Nhat = R.cross(V).normalized()
        That = Nhat.cross(Rhat).normalized()
        
        RTNcoeffs = vf.stack([Rhat,That,Nhat]) 
        
        return vf.ColMatrix(RTNcoeffs,3,3)*U
    
    def MEEtoCartesianFunc(self):
        args = Args(6)
        
        p,f,g,h,k,L = Args(6).tolist()
        
        sinL = vf.sin(L)
        cosL = vf.cos(L)
        sqp = vf.sqrt(1.0/p)
        
        w = 1+f*cosL +g*sinL
        s2 = 1+h**2 +k**2
        a2 = h**2 - k**2
        r = p/w
        r_s2 = r/s2
        subs2 = 1.0/s2
        
        rvec = r_s2*vf.stack([cosL + a2*cosL + 2.*h*k*sinL, sinL - a2*sinL + 2.*h*k*cosL,
                         2.0*(h*sinL - k*cosL)])
        
        vvec = -subs2*sqp*vf.stack([sinL + a2*sinL - 2.*h*k*cosL + g - 2.*f*h*k + a2*g,
                         -cosL + a2*cosL + 2.*h*k*sinL - f + 2.*g*h*k + a2*f,
                         -2.0*(h*cosL + k*sinL + f*h + g*k)])
        
        return vf.stack([rvec, vvec])
    
    def MEEToCartesian(self,Traj,TransformControl=True):
        NewTraj = []
        meecart = self.MEEtoCartesianFunc()
        rtncart = self.RTNtoCartesianFunc()
        
        for T in Traj:
            Tmp = np.zeros(len(T))
            
            RVcart = meecart.compute(np.copy(T[0:6]))
            
            Tmp[0:6]=RVcart
            
            if(self.HasMass):
                Tmp[6] = T[6]
                Tmp[7] = T[7]
            else:
                Tmp[6] = T[6]
            
            if(self.HasControl):
                if(self.HasMass):
                    U = T[8:11]
                else:
                    U = T[7:10]
                
                if(TransformControl):
                    Xtmp = np.zeros((9))
                    Xtmp[0:6]=RVcart
                    Xtmp[6:9]=U
                    U = rtncart.compute(Xtmp)
                
                if(self.HasMass):
                    Tmp[8:11]=U
                else:
                    Tmp[7:10]=U
            NewTraj.append(Tmp)
            
        return NewTraj

            
            
        
        
        
        

