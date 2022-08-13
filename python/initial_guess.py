import pykep as pk
from pykep.orbit_plots import plot_planet, plot_lambert
from pykep import AU, DAY2SEC
import pygmo as pg
import numpy as np

# Plotting imports
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# We define the Lambert problem
# t1 = pk.epoch(0)
# t2 = pk.epoch(640)
t1 = pk.epoch_from_string('2012-04-10 23:59:54.003') # Depart Earth
t2 = pk.epoch_from_string('2012-10-04 23:59:54.003') # Venus GA 1
t3 = pk.epoch_from_string('2014-03-11 23:59:54.003') # Earth GA 1
t4 = pk.epoch_from_string('2016-03-10 23:59:54.003') # Earth GA 2
t5 = pk.epoch_from_string('2017-02-22 23:59:54.003') # Venus GA 2
t6 = pk.epoch_from_string('2018-05-17 23:59:54.003') # Venus GA 3
t7 = pk.epoch_from_string('2019-08-09 23:59:54.003') # Venus GA 4
t8 = pk.epoch_from_string('2020-11-01 23:59:54.003') # Venus GA 5
t9 = pk.epoch_from_string('2022-02-24 23:59:54.003') # Venus GA 6

dt1 = (t2.mjd2000 - t1.mjd2000) * DAY2SEC
dt2 = (t3.mjd2000 - t2.mjd2000) * DAY2SEC
dt3 = (t4.mjd2000 - t3.mjd2000) * DAY2SEC
dt4 = (t5.mjd2000 - t4.mjd2000) * DAY2SEC
dt5 = (t6.mjd2000 - t5.mjd2000) * DAY2SEC
dt6 = (t7.mjd2000 - t6.mjd2000) * DAY2SEC
dt7 = (t8.mjd2000 - t7.mjd2000) * DAY2SEC
dt8 = (t9.mjd2000 - t8.mjd2000) * DAY2SEC


mu_sun=132712440041.940e9
cAU = 1.496e11

earth = pk.planet.jpl_lp('earth')
rE, vE = earth.eph(t1) # Earth at departure
vEarth = np.asarray(vE)

venus = pk.planet.jpl_lp('venus')
rV, vV = venus.eph(t2) # Venus at arrival
vVenus = np.asarray(vV)
rVenus = np.asarray(rV)

mars = pk.planet.jpl_lp('mars')

# We solve the Lambert problem
l = pk.lambert_problem(r1 = rE, r2 = rV, tof = dt1, mu = pk.MU_SUN, max_revs=4)
# v1 = np.asarray(l.get_v1()[0])
v1 = np.asarray(l.get_v1()[0])
vf1 = np.asarray(l.get_v2()[0])
dv1 = np.linalg.norm(v1[0:2]-vEarth[0:2]) # arrival relative velocity
# departdv = np.linalg.norm(v1[0:2]-vEarth[0:2]) # departure relative velocity

# Earth arrival GA 1, Venus-Earth
r3, v3 = earth.eph(t3)
l2 = pk.lambert_problem(r1 = rV, r2 = r3, tof = dt2, mu = pk.MU_SUN, max_revs=4)
v2 = np.asarray(l2.get_v1()[2])
vf2 = np.asarray(l2.get_v2()[2])
print(l2.get_Nmax())

rV3, vV2 = venus.eph(t2)
rE3, vE3 = earth.eph(t3)
dv2 = np.abs(np.linalg.norm(vf2[0:2]-vE3[0:2])-np.linalg.norm(v2[0:2]-vV2[0:2])) # arrival relative velocity

# Earth arrival GA 2
r4, v4= earth.eph(t4)
l3 = pk.lambert_problem(r1 = r3, r2 = r4, tof = dt3, mu = pk.MU_SUN, max_revs=2)
v3 = np.asarray(l3.get_v1()[0])
vf3 = np.asarray(l3.get_v2()[0])

rE3, vE3 = earth.eph(t3)
rE4, vE4 = earth.eph(t4)
dv3 = np.abs(np.linalg.norm(vf3[0:2]-vE4[0:2])-np.linalg.norm(v3[0:2]-vE3[0:2]))


# Venus arrival GA 2
r5, v5= venus.eph(t5)
l4 = pk.lambert_problem(r1 = r4, r2 = r5, tof = dt4, mu = pk.MU_SUN, max_revs=2)
v4 = np.asarray(l4.get_v1()[0])
vf4 = np.asarray(l4.get_v2()[0])

rE4, vE4 = earth.eph(t4)
rV5, vV5 = venus.eph(t5)
dv4 = np.abs(np.linalg.norm(vf4[0:2]-vV5[0:2])-np.linalg.norm(v4[0:2]-vE4[0:2]))

# Venus arrival GA 3
r6, v6= venus.eph(t6)
l5 = pk.lambert_problem(r1 = r5, r2 = r6, tof = dt5, mu = pk.MU_SUN, max_revs=2)
v5 = np.asarray(l5.get_v1()[0])
vf5 = np.asarray(l5.get_v2()[0])

rV5, vV5 = venus.eph(t5)
rV6, vV6 = venus.eph(t6)
dv5 = np.abs(np.linalg.norm(vf5[0:2]-vV6[0:2])-np.linalg.norm(v5[0:2]-vV5[0:2]))

# Venus arrival GA 4
r7, v7= venus.eph(t7)
l6 = pk.lambert_problem(r1 = r6, r2 = r7, tof = dt6, mu = pk.MU_SUN, max_revs=2)
v6 = np.asarray(l6.get_v1()[0])
vf6 = np.asarray(l6.get_v2()[0])

rV6, vV6 = venus.eph(t6)
rV7, vV7 = venus.eph(t7)
dv6 = np.abs(np.linalg.norm(vf6[0:2]-vV7[0:2])-np.linalg.norm(v6[0:2]-vV6[0:2]))

# Venus arrival GA 5
r8, v8= venus.eph(t8)
l7 = pk.lambert_problem(r1 = r7, r2 = r8, tof = dt7, mu = pk.MU_SUN, max_revs=2)
v7 = np.asarray(l7.get_v1()[2])
vf7 = np.asarray(l7.get_v2()[2])

rV7, vV7 = venus.eph(t7)
rV8, vV8 = venus.eph(t8)
dv7 = np.abs(np.linalg.norm(vf7[0:2]-vV8[0:2])-np.linalg.norm(v7[0:2]-vV7[0:2]))


# Load in PH27
pk.util.load_spice_kernel("54186922.bsp")
asteroid = pk.planet.spice('54186922','SOLAR_SYSTEM_BARYCENTER','ECLIPJ2000','NONE')
tA = pk.epoch_from_string('2022-01-01 23:59:54.003')
tAf = pk.epoch_from_string('2023-06-06 23:59:54.003')
# We plot
mpl.rcParams['legend.fontsize'] = 10

# Venus arrival GA 4
r9, v9= asteroid.eph(t9)
l8 = pk.lambert_problem(r1 = r8, r2 = r9, tof = dt8, mu = pk.MU_SUN, max_revs=3)
v8 = np.asarray(l8.get_v1()[2])
vf8 = np.asarray(l8.get_v2()[2])

rV8, vV8 = venus.eph(t8)
rA9, vA9 = asteroid.eph(t9)
dv8 = np.abs(np.linalg.norm(vf8[0:2]-vA9[0:2])-np.linalg.norm(v8[0:2]-vV8[0:2]))

print(f"total DV: {(dv1+dv2+dv3+dv4+dv5+dv6+dv7+dv8)/1000} km/s")
print(f"dv1: {dv1/1000}")
print(f"dv2: {dv2/1000}")
print(f"dv3: {dv3/1000}")
print(f"dv8: {dv8/1000}")
# Create the figure and axis
fig = plt.figure(figsize = (16,5))
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.scatter([0], [0], [0], color=['y'])

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.scatter([0], [0], [0], color=['y'])
ax2.view_init(90, 0)

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.scatter([0], [0], [0], color=['y'])
ax3.view_init(0,0)


for ax in [ax1, ax2, ax3]:
    # Plot the planet orbits
    ax.set(xlim = (-1.5, 1.5),ylim = (-1.5, 1.5),zlim = (-1.5, 1.5))
    plot_planet(earth, t0=t1, color='g', units=AU, axes=ax)
    plot_planet(mars, t0=t1, color='m', units=AU, axes=ax)
    plot_planet(venus, t0=t2, color='r', units=AU, axes=ax)
    plot_planet(earth, t0=t3, color='g', units=AU, axes = ax)
    plot_planet(earth, t0=t4, color='g', units=AU, axes = ax)
    ax.scatter(r9[0]/cAU, r9[1]/cAU, r9[2]/cAU, color=['k'])
    # plot_planet(asteroid, t0=tA, tf=tAf, N = 500, units=AU, axes = ax, color='k')
    plot_planet(asteroid, t0=t9, tf=tAf, N = 500, units=AU, axes = ax, color='k')

    # Plot the Lambert solutions
    axis = plot_lambert(l, color='y', units=AU, axes=ax)
    axis = plot_lambert(l2, sol=2, color='y', units=AU, axes=ax)
    axis = plot_lambert(l3, sol=0, color='y', units=AU, axes=ax)
    axis = plot_lambert(l4, sol=0, color='y', units=AU, axes=ax)
    axis = plot_lambert(l5, sol=0, color='y', units=AU, axes=ax)
    axis = plot_lambert(l6, sol=0, color='y', units=AU, axes=ax)
    axis = plot_lambert(l7, sol=2, color='y', units=AU, axes=ax)
    axis = plot_lambert(l8, sol=2, color='y', units=AU, axes=ax)
    
plt.show()

breakpoint()

