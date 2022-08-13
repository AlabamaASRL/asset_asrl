import spiceypy as spice
import matplotlib.pyplot as plt
from pykep.orbit_plots import plot_planet, plot_lambert
from pykep import AU, DAY2SEC
import numpy as np
import pykep as pk
import scipy.io
mu=132712440041.940

spice.furnsh("standard.html")
steps = 200

utc_start_1 = ['Apr 30, 2024']
utc_start_2 = ['Jan 1, 2025']
utc_end_1 = ['Mar 1, 2025'] #Old ['Oct 1, 2027']
utc_end_2 = ['Oct 1, 2026']
secondsInAYear=  365.*24.*60.*60.
# get et values one and two, we could vectorize str2et
etOne_1 = spice.str2et(utc_start_1[0])
etOne_2 = spice.str2et(utc_start_2[0])
etTwo_1 = spice.str2et(utc_end_1[0])
etTwo_2 = spice.str2et(utc_end_2[0])

start_times = np.linspace(etOne_1,etOne_2,steps)
end_times = np.linspace(etTwo_1,etTwo_2,steps+1)
dvHolder = np.zeros((steps+1,steps))

stateMars, lightTimes = spice.spkezr('Mars', end_times, 'J2000', 'NONE', 'SUN')
stateEarth, lightTimes = spice.spkezr('EARTH', start_times, 'J2000', 'NONE', 'SUN')
stateMars = np.asarray(stateMars)
stateEarth = np.asarray(stateEarth)

for c in range(steps+1):
    for d in range(steps):
        l_arc = pk.lambert_problem(r1 = stateEarth[d,0:3], r2 = stateMars[c,0:3], tof = end_times[c]-start_times[d] ,mu=mu,max_revs=1)
        dv = np.linalg.norm(l_arc.get_v1()[0]-stateEarth[d,3:6])+np.linalg.norm(l_arc.get_v2()[0]-stateMars[c,3:6])
        if(dv<33):
            dvHolder[c,d] = dv
        else:
            dvHolder[c,d] = np.nan

    print(c/(steps+1)) 
        

fig1, ax1 = plt.subplots()
ax1.contour(start_times,end_times,dvHolder)
ax1.set_aspect('equal')


plt.show()

#scipy.io.savemat('porkchopPlot.mat', mdict={'start_times': start_times, 'end_times':end_times,'dvHolder':dvHolder})
breakpoint()