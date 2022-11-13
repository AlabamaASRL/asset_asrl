import asset as _asset
import inspect

class ODEBase:
 
    def __init__(self,odefunc,Xvars,Uvars = None,Pvars = None):

        mlist = inspect.getmembers(_asset.OptimalControl)
        
        
        hasUvars = (Uvars != None and Uvars != 0)
        hasPvars = (Pvars != None and Pvars != 0)
        name = 'ode_'+str(Xvars)

        if(hasUvars):
            name +='_'+str(Uvars)
        if(hasPvars):
            if(not hasUvars):
                name+='_0'
            name +='_'+str(Pvars)
            
        
        constsize =False
        for m in mlist:
            if(name == m[0]):
                
                self.ode = m[1].ode(odefunc)
                constsize =True
                break
        if(constsize==False):
            if(hasUvars and hasPvars):
                self.ode = _asset.OptimalControl.ode_x_u_p.ode(odefunc,Xvars,Uvars,Pvars)
            elif(hasUvars):
                Pvars = 0
                self.ode = _asset.OptimalControl.ode_x_u.ode(odefunc,Xvars,Uvars)
            elif(hasPvars):
                Uvars = 0
                self.ode = _asset.OptimalControl.ode_x_u_p.ode(odefunc,Xvars,0,Pvars)
            else:
                Uvars = 0
                Pvars = 0
                self.ode = _asset.OptimalControl.ode_x.ode(odefunc,Xvars)
                
        
        
    def phase(self,*args):
        return self.ode.phase(*args)
    def integrator(self,*args):
        return self.ode.integrator(*args)
    def vf(self):
        return self.ode.vf()