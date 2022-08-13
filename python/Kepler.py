# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 12:33:50 2021

@author: gerde
"""

import pykep as pk
from pykep.orbit_plots import plot_planet, plot_lambert
from pykep import AU, DAY2SEC
import pygmo as pg
import numpy as np
import spiceypy as spice

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

spice.furnsh("Spice/BasicKernel.txt")

steps = 200
utc_start_1 = ['Apr 30, 2024']
utc_start_2 = ['Jan 1, 2025']
utc_end_1 = ['Mar 30, 2025'] #Old ['Oct 1, 2027']
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

statePH27 = spice.spkezr('54186922', end_times, 'J2000', 'NONE', '0')[0]


t1 = pk.epoch(10000)
t2 = pk.epoch(10250)
t3 = pk.epoch(10250 + 250.0)
dt = (t2.mjd2000 - t1.mjd2000) * DAY2SEC

dt2 = (t3.mjd2000 - t2.mjd2000) * DAY2SEC




pk.util.load_spice_kernel("Spice/de432s.bsp")
pk.util.load_spice_kernel("Spice/naif0012.tls")
pk.util.load_spice_kernel("Spice/gm_de431.tpc")
pk.util.load_spice_kernel("Spice/54186922.bsp")
ph27 = pk.planet.spice('54186922', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, 1, 1, 1)
earth = pk.planet.spice('EARTH', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, pk.MU_EARTH, 1, 1)
mars = pk.planet.spice('MARS BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, 0, 1, 1)
venus = pk.planet.spice('VENUS BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, 0, 1, 1)
mercury = pk.planet.spice('MERCURY BARYCENTER', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, 0, 1, 1)
earthR, earthV = earth.eph(t1)
venusR, venusV = venus.eph(t2)

lambert_1arc = pk.lambert_problem(r1 = earthR, r2 = venusR, tof = dt, mu = pk.MU_SUN, max_revs = 2)


earthR2, earthV2 = earth.eph(t3)
rf_1,vf_1 = pk.propagate_lagrangian(r0 = earthR, v0 = lambert_1arc.get_v1()[0], tof = dt, mu = pk.MU_SUN)
lambert_2arc = pk.lambert_problem(r1 = rf_1,
                                  r2 = earthR2, tof = dt2, mu = pk.MU_SUN, max_revs = 2)


t4 = pk.epoch(10250 + 250.0 + 1043.0)
dt3 = (t4.mjd2000 - t3.mjd2000) * DAY2SEC
earthR3, earthV3 = earth.eph(t4)
rf_2,vf_2 = pk.propagate_lagrangian(r0 = rf_1, v0 = lambert_2arc.get_v1()[0], tof = dt2, mu = pk.MU_SUN)
lambert_3arc = pk.lambert_problem(r1 = rf_2,
                                  r2 = earthR3, tof = dt3, mu = pk.MU_SUN, max_revs = 2)

t5 = pk.epoch(10250 + 250.0 + 1043.0 + 1043.0)
dt4 = (t5.mjd2000 - t4.mjd2000) * DAY2SEC
earthR4, earthV4 = earth.eph(t5)
rf_3,vf_3 = pk.propagate_lagrangian(r0 = rf_2, v0 = lambert_3arc.get_v1()[0], tof = dt3, mu = pk.MU_SUN)
lambert_4arc = pk.lambert_problem(r1 = rf_3,
                                  r2 = earthR4, tof = dt4, mu = pk.MU_SUN, max_revs = 2)

t6 = pk.epoch(10250 + 250.0 + 1043.0 + 1043.0 + 580.0)
dt5 = (t6.mjd2000 - t5.mjd2000) * DAY2SEC
venusR5, venusV5 = venus.eph(t6)
rf_4,vf_4 = pk.propagate_lagrangian(r0 = rf_3, v0 = lambert_4arc.get_v1()[0], tof = dt4, mu = pk.MU_SUN)
lambert_5arc = pk.lambert_problem(r1 = rf_4,
                                  r2 = venusR5, tof = dt5, mu = pk.MU_SUN, max_revs = 2)

t7 = pk.epoch(10250 + 250.0 + 1043.0 + 1043.0 + 580.0 + 580.0)
dt6 = (t7.mjd2000 - t6.mjd2000) * DAY2SEC
venusR6, venusV6 = venus.eph(t7)
rf_5,vf_5 = pk.propagate_lagrangian(r0 = rf_4, v0 = lambert_5arc.get_v1()[0], tof = dt5, mu = pk.MU_SUN)
lambert_6arc = pk.lambert_problem(r1 = rf_5,
                                  r2 = venusR6, tof = dt6, mu = pk.MU_SUN, max_revs = 2)

t8 = pk.epoch(10250 + 250.0 + 1043.0 + 1043.0 + 580.0 + 580.0 + 100)
dt7 = (t8.mjd2000 - t7.mjd2000) * DAY2SEC
PH27R7, PH27V7 = ph27.eph(t8)
rf_6,vf_6 = pk.propagate_lagrangian(r0 = rf_5, v0 = lambert_6arc.get_v1()[0], tof = dt6, mu = pk.MU_SUN)
print(rf_6)
lambert_7arc = pk.lambert_problem(r1 = rf_6,
                                  r2 = PH27R7, tof = dt7, mu = pk.MU_SUN, max_revs = 2)

total_time = (250.0 + 1043.0 + 1043.0 + 580.0 + 580.0 + 100)/365.
Dv1 = abs(np.linalg.norm(np.asarray(lambert_1arc.get_v1()[0]) - np.asarray(lambert_1arc.get_v2()[0])))
Dv2 = abs(np.linalg.norm(np.asarray(lambert_2arc.get_v1()[0]) - np.asarray(lambert_2arc.get_v2()[0])))
Dv3 = abs(np.linalg.norm(np.asarray(lambert_3arc.get_v1()[0]) - np.asarray(lambert_3arc.get_v2()[0])))
Dv4 = abs(np.linalg.norm(np.asarray(lambert_4arc.get_v1()[0]) - np.asarray(lambert_4arc.get_v2()[0])))
Dv5 = abs(np.linalg.norm(np.asarray(lambert_5arc.get_v1()[0]) - np.asarray(lambert_5arc.get_v2()[0])))
Dv6 = abs(np.linalg.norm(np.asarray(lambert_6arc.get_v1()[0]) - np.asarray(lambert_6arc.get_v2()[0])))
Dv7 = abs(np.linalg.norm(np.asarray(lambert_7arc.get_v1()[0]) - np.asarray(lambert_7arc.get_v2()[0])))


totalDv = Dv1 + Dv2+ Dv3+ Dv4+Dv5+Dv6+Dv7
print(totalDv/1000.0)


# We plot
mpl.rcParams['legend.fontsize'] = 10

# Create the figure and axis
fig = plt.figure(figsize = (9,9))
ax = fig.add_subplot(projection='3d')
ax.set(xlim = (-1.0, 1.0), ylim=(-1.0, 1.0), zlim = (-1.0, 1.0))
ax.scatter(0.0, .0, .0, color = "k")

# Plot the planet orbits
plot_planet(earth, t0=t1, color=(0, 1, 0), legend=False, units=AU, axes=ax)
plot_planet(mars, t0=t2, color=(1, 0.0, 0), legend=False, units=AU, axes=ax)
plot_planet(venus, t0=t2, color=(1, 1, 0), legend=False, units=AU, axes=ax)
plot_planet(mercury, t0=t2, color=(.5, 0, 0), legend=False, units=AU, axes=ax)
plot_planet(ph27, t0=t2, color=(0.0, 0., 1), legend=False, units=AU, axes=ax)
plot_planet(ph27, t0=t8, color=(0.0, 0., .5), legend=False, units=AU, axes=ax)
# Plot the Lambert solutions

plot_lambert(lambert_1arc, sol = 0, color = (.1, .1, 0), legend = False, units=AU, axes=ax)
plot_lambert(lambert_2arc, sol = 0, color = (.2, .2, 0), legend = False, units=AU, axes=ax)
plot_lambert(lambert_3arc, sol =1, color = (.3, .3, 0), legend = False, units=AU, axes=ax)
plot_lambert(lambert_4arc, sol =1, color = (.4, .4, 0), legend = False, units=AU, axes=ax)
plot_lambert(lambert_5arc, sol =1, color = (.5, .5, 0), legend = False, units=AU, axes=ax)
plot_lambert(lambert_6arc, sol =1, color = (.6, .6, 0), legend = False, units=AU, axes=ax)
plot_lambert(lambert_7arc, sol =0, color = (.7, .7, 0), legend = False, units=AU, axes=ax)


plt.show()
