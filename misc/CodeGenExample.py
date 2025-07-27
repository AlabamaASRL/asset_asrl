import sympy as sp
import numpy as np
from CodeGen import AssetHeaderGen


def MEE():
     
     Xs = sp.symbols('x:9')
     
     p,f,g,h,k,L,ur,ut,un = Xs
     
     mu = sp.symbols("mu")
     
     sinL = sp.sin(L)
     cosL = sp.cos(L)
     sqp  = sp.sqrt(p)/sp.sqrt(mu)
     
     w  = 1.+f*cosL +g*sinL
     s2 = 1.+h**2 +k**2
     
     pdot  = 2.*(p/w)*ut
     fdot  = sum([ ur*sinL  ,((w+1)*cosL +f)*(ut/w), -(h*sinL-k*cosL)*(g*un/w)])
     gdot  = sum([-ur*cosL , ((w+1)*sinL +g)*(ut/w) , (h*sinL-k*cosL)*(f*un/w)])
     hdot  = cosL*((s2*un/w)/2.0) 
     kdot  = sinL*((s2*un/w)/2.0)
     Ldot  = mu*(w/p)*(w/p) + (1.0/w)*(h*sinL -k*cosL)*un
     
     Eq = sp.Matrix([pdot,fdot,gdot,hdot,kdot,Ldot])*sqp
     
     header= AssetHeaderGen('MEETest',Eq,sp.Matrix(Xs),
                       [(mu,"Gravitational Parameter")],
                       docstr = "Modified Equinoctial Element Dynamics")
     
     header.make_header()


def CR3BP():
    
    Xs = sp.symbols('x:6')
    
    x,y,z,x_dot,y_dot,z_dot = Xs
    
    mu = sp.symbols("mu")
    
    x_ddot = x+2*y_dot-((1-mu)*(x+mu))/((x+mu)**2+y**2+z**2)**(3/2) -(mu*(x-(1-mu)))/((x-(1-mu))**2+y**2+z**2)**(3/2)
             
    y_ddot = y-2*x_dot-((1-mu)*y)/((x+mu)**2+y**2+z**2)**(3/2)-(mu*y)/((x-(1-mu))**2+y**2+z**2)**(3/2)
    z_ddot = -((1-mu)*z)/((x+mu)**2+y**2+z**2)**(3/2)-(mu*z)/((x-(1-mu))**2+y**2+z**2)**(3/2)
    

    Eq = sp.Matrix([x_dot,y_dot,z_dot,x_ddot,y_ddot,z_ddot])
    
    header = AssetHeaderGen("CR3BPTest",Eq,sp.Matrix(Xs),
                       [(mu,"Gravitational Parameter")],
                       docstr = "CR3BP Equations of Motion VectorFunction")
    
    header.make_header()


if __name__ == "__main__":
     
     MEE()
     CR3BP()
