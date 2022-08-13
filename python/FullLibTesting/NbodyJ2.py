import numpy as np
import asset_asrl as ast
import asset_asrl.Astro as Astro
import asset_asrl.Astro.Constants as c
import asset_asrl.Astro.Date as date
from asset_asrl.Astro.FramePlot import plt,CRPlot,TBPlot
from asset_asrl.Astro.AstroModels import TwoBody,TwoBodyFrame,NBody,NBodyFrame
from asset_asrl.Astro.AstroModels import EPPR,EPPRFrame,CR3BP,CR3BPFrame
import spiceypy as sp
import time


sp.furnsh('BasicKernel.txt')


JD0 = date.date_to_jd(2022, 6, 14)
JDF = date.date_to_jd(2042, 6, 14)

SpiceFrame = "J2000"
N = 12000

Lstar = c.RadiusEarth

eframe = NBodyFrame("EARTH",c.MuEarth,Lstar,JD0,JDF,N,SpiceFrame)
eframe.AddSpiceBodies(["MOON","SUN","VENUS","JUPITER BARYCENTER","SATURN BARYCENTER","MARS BARYCENTER","URANUS BARYCENTER"])
eframe.Add_P1_J2Effect(c.J2Earth,c.RadiusEarth)



ode = NBody(eframe,Enable_J2=True)
integ = ode.integrator(.001)
integ.Adaptive = True
integ.setAbsTol(1.0e-10)


coptraj = eframe.Copernicus_to_Frame("ASSETEarthJ2Test.CSV")

t0 = time.perf_counter()
Traj = integ.integrate_dense(coptraj[0],coptraj[-1][6])
tf = time.perf_counter()


print(tf-t0)

print(len(Traj))

error = np.linalg.norm(Traj[-1][0:3]-coptraj[-1][0:3])
print("Position Error: ",error*eframe.lstar," m")


plot1 = TBPlot(None,"Earth",'g')
plot1.addTraj(coptraj,"J2 Trajectory",'r')

 
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
r = 1
u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
ax.plot_surface(x, y, z, color='g',alpha=.4)
plot1.Plot3dAx(ax)
plt.show()

