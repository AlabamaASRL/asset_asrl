

import asset_asrl as ast

import asset_asrl.Astro as Astro
import asset_asrl.Astro.Constants as c
import asset_asrl.Astro.Date as date
from   asset_asrl.Astro.FramePlot import plt,CRPlot,TBPlot
from   asset_asrl.Astro.AstroModels import TwoBody,TwoBodyFrame,NBody,NBodyFrame
from   asset_asrl.Astro.AstroModels import EPPR,EPPRFrame,CR3BP,CR3BPFrame

import spiceypy as sp


sp.furnsh('BasicKernel.txt')


JD0 = date.date_to_jd(2022, 6, 14)
JDF = date.date_to_jd(2042, 6, 14)

SpiceFrame = "ECLIPJ2000"

N = 6000

sframe  = NBodyFrame("SUN",c.MuSun,c.AU,JD0,JDF,N,SpiceFrame)
sframe.AddSpiceBodies(["JUPITER BARYCENTER"])

seepprframe = EPPRFrame("SUN", c.MuSun, "EARTH", c.MuEarth, c.AU, JD0, JDF,N,SpiceFrame)
epprode = EPPR(seepprframe)



plot1 = TBPlot(None)
plot2 = CRPlot(epprode)

 
earth_i = sframe.GetSpiceBodyTraj("EARTH", N)
venus_i = sframe.GetSpiceBodyTraj("VENUS", N)

venus_eppr = seepprframe.GetSpiceBodyEPPRTraj("VENUS", N)

plot1.addTraj(earth_i,    'Earth','g')
plot1.addTraj(venus_i,    'Venus','orange')
plot2.addTraj(venus_eppr, 'Venus','orange')



fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.grid(True)
plot1.Plot2dAx(ax1,legend = True)

ax2 = fig.add_subplot(122)
ax2.grid(True)
plot2.Plot2dAx(ax2,legend = True)


plt.show()

