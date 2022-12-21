import numpy as np
import asset as ast

vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments

class TwoBodyFrame():
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
    def __init__(self,P1mu,lstar):
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
    def TwoBodyEOMs(self,r,v,otherAccs=[],otherEOMs=[]):
        """
        Two body equations of motion

        Parameters
        ----------
        r : ASSET VectorFunction
            3x1 Particle position from primary
        v : ASSET VectorFunction
            3x1 particle velocity
        otherAccs : list
            list of other accelerations to add to model
        otherEOMS : list
            list of other equations of motion driving model

        Returns
        -------
        Full equations of motion : ASSET VectorFunction
            All equations of motion for two body model

        """
        accG = (-self.mu )* r.normalized_power3() 
        accT = vf.sum([accG ]+ otherAccs)
        return vf.stack([v, accT]+otherEOMs)
