# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 12:13:35 2021

@author: Jared
"""

# Imports
import pykep as pk
import pygmo as pg
import pygmo_plugins_nonfree as ppnf
from pykep.orbit_plots import plot_planet, plot_lambert
import numpy as np
from pykep.examples import add_gradient
from pykep import AU, DAY2SEC
from AstroModels import TwoBody

# Plotting imports
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os.path
import sys
import asset as ast
import Date


pk.util.load_spice_kernel("Spice/de432s.bsp")
pk.util.load_spice_kernel("Spice/naif0012.tls")
pk.util.load_spice_kernel("Spice/gm_de431.tpc")
pk.util.load_spice_kernel("Spice/54186922.bsp")
ph27 = pk.planet.spice('54186922', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, 1, 1, 1)
seq= [pk.planet.jpl_lp('earth'), pk.planet.jpl_lp('venus'), pk.planet.jpl_lp('earth'),
           pk.planet.jpl_lp('venus'), pk.planet.jpl_lp('venus'), pk.planet.jpl_lp('venus'), 
           pk.planet.jpl_lp('venus'), ph27]
udp = pk.trajopt.mga(
     seq = seq,
     t0 = [8000, 12000], # This is in mjd2000
     tof = [[.4*365, 1.4*365.],[.8*365., 1.8*365.], [.8*365., 1.8*365.],
            [.8*365., 1.8*365.], [.8*365., 1.8*365.], [.8*365., 1.8*365.],
            [.8*365., 1.8*365.]], # This is in days
     vinf = 3., # This is in km/s
     orbit_insertion = False,
     multi_objective = False
)


prob = pg.problem(udp)

prob.c_tol = 1e-6

uda = pg.nlopt("slsqp")

uda2 = pg.mbh(uda, 10, 0.35)
algo = pg.algorithm(uda2)
algo.set_verbosity(1)

pop = pg.population(prob, 500)
# And optimize
#pop = algo.evolve(pop)


# We plot
mpl.rcParams['legend.fontsize'] = 10

# Create the figure and axis
fig = plt.figure(figsize = (13,9))
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
udp.plot(pop.champion_x, axes = ax1)
udp.plot(pop.champion_x, axes = ax3)
ax1.set(xlim = (-1.0, 1.0), ylim=(-1.0, 1.0), zlim = (-1.0, 1.0))
ax2.set(xlim = (-1.0, 1.0), ylim=(-1.0, 1.0), zlim = (-1.0, 1.0))
ax3.set(xlim = (-1.0, 1.0), ylim=(-1.0, 1.0), zlim = (-1.0, 1.0))


ode = TwoBody(1.32712440018e20, 1.496e11)
tbinteg = ode.integrator(.001)

t = pop.champion_x[0]
lambert_vec = []
states = []
dts = np.array(pop.champion_x[1:])
statetime = 0
for i in range(1, len(pop.champion_x)):
    t += pop.champion_x[i]
    t2 = pk.epoch(t)
    t1 = pk.epoch(t - pop.champion_x[i])
    dt = (t2.mjd2000 - t1.mjd2000)*DAY2SEC
    planet1R, planet1V = seq[i-1].eph(t1)
    planet2R, planet2V = seq[i].eph(t2)
    lambert_arc = pk.lambert_problem(r1 = planet1R, r2 = planet2R, tof = dt, mu = pk.MU_SUN, max_revs = 0)
    lambert_vec.append(lambert_arc)
    r1 = np.asarray(planet1R)
    v1 = np.asarray(lambert_arc.get_v1()[0])
    planetv = np.asarray(planet1V)
    states.append(np.concatenate((r1/ode.lstar, v1/ode.vstar, [statetime])))
    statetime += dts[i-1]*86400.0/ode.tstar
    plot_lambert(lambert_arc, sol = 0, color = (.5, .5, 0), legend = False, units=AU, axes=ax2)
    plot_lambert(lambert_arc, sol = 0, color = (.5, .5, 0), legend = False, units=AU, axes=ax3)
plt.show()
vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags
    



arc1 = tbinteg.integrate_dense(states[0], states[1][6], 500)
T1 = np.array(arc1).T

arc2 = tbinteg.integrate_dense(states[1], states[2][6], 500)
T2 = np.array(arc2).T

arc3 = tbinteg.integrate_dense(states[2], states[3][6], 500)
T3 = np.array(arc3).T

arc4 = tbinteg.integrate_dense(states[3], states[4][6], 500)
T4 = np.array(arc4).T

arc5 = tbinteg.integrate_dense(states[4], states[5][6], 500)
T5 = np.array(arc5).T

arc6 = tbinteg.integrate_dense(states[5], states[6][6], 500)
T6 = np.array(arc6).T

arc7 = tbinteg.integrate_dense(states[6], statetime, 500)
T7 = np.array(arc7).T


vec = [float(pk.epoch(pop.champion_x[0]).jd), arc1, arc2, arc3, arc4, arc5, arc6, arc7]

vec = np.load("IG.npy", allow_pickle = True)



#This is the final combined states
FinalArc = np.concatenate((vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7]))
Traj = np.array(FinalArc).T

print(FinalArc[-1][6])

fig = plt.figure(figsize = (13,9))
ax = fig.add_subplot(projection = "3d")
ax.set(xlim = (-1.0, 1.0), ylim=(-1.0, 1.0), zlim = (-1.0, 1.0))
ax.plot(Traj[0], Traj[1], Traj[2])
'''
ax.plot(T1[0], T1[1], T1[2], label = "Arc 1")
ax.plot(T2[0], T2[1], T2[2], label = "Arc 2")
ax.plot(T3[0], T3[1], T3[2], label = "Arc 3")
ax.plot(T4[0], T4[1], T4[2], label = "Arc 4")
ax.plot(T5[0], T5[1], T5[2], label = "Arc 5")
ax.plot(T6[0], T6[1], T6[2], label = "Arc 6")
ax.plot(T7[0], T7[1], T7[2], label = "Arc 7")
'''
ax.legend()

plt.show()










