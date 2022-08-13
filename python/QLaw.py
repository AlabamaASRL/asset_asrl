# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:30:04 2021

@author: Jared
"""
import asset as ast
import numpy as np
import MKgSecConstants as c
vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments
'''
def QLaw(accel, target, weights): 
    args = Args(8)
    mu = 1.0
    #p, f, g, h, k, L
    x = args.head(6)
    u = args.tail(2)
    sina = vf.sin(u[0])
    sinb = vf.sin(u[1])
    cosa = vf.cos(u[0])
    cosb = vf.cos(u[1])
    
    q = 1.0 + x[1]*vf.cos(x[5]) + x[2]*vf.sin(x[5])
    A = np.zeros((6, 3))
    sqrtpu = vf.sqrt(x[0]/mu)
    cosL =vf.cos(x[5])
    sinL = vf.sin(x[5])
    A[0, 1] = 2.0*x[0]
    A[1, 0] = q*sinL
    A[1, 1] = (q + 1)*cosL + x[1]
    A[1, 2] = -x[2]*(x[3]*sinL - x[4]*cosL)
    A[2, 0] = -q*cosL
    A[2, 1] = (q + 1)*sinL + x[1]
    A[2, 2] = x[1]*(x[3]*sinL - x[4]*cosL)
    A[3, 2] = .5*(1 + x[3]**2 + x[4]**2)*cosL
    A[4, 2] = .5*(1 + x[3]**2 + x[4]**2)*sinL
    A[5, 2] = x[3]*sinL - x[4]*cosL
    A = (1.0/q)*vf.sqrt(x[0]/mu) * A
    
    b = np.array([0, 0, 0, 0, 0, vf.sqrt(mu*x[0]) * (q/x[0])**2])
    s = 1.0 + x[3]**2 + x[4]**2
    dpa = (-2.0*x[0]/q)*sqrtpu*accel*sina*cosb
    dpb = (-2.0*x[0]/q)*sqrtpu*accel*cosa*sinb  
    maxp  = (2*x[0]/q)*sqrtpu*accel
    
    dfa = sqrtpu * (cosa*cosb*sinL - ((q + 1.0) * cosL + x[1])*sina*cosb*(1/q))*accel
    dfb = sqrtpu * (-sina*sinb*sinL- ((q + 1.0) * cosL + x[1])*cosa*sinb*(1/q) - (x[3]*sinL - x[4]*cosL)*(x[2]/q)*cosb)*accel
    maxf = sqrtpu * (accel/q)*vf.sqrt(q**2*sinL**2 + ((q+1.0)*sinL + x[1])**2 + (x[3]*sinL - x[4]*cosL)**2 * x[2]**2)
    
    dga = sqrtpu * (-cosa*cosb*cosL - ((q + 1.0) * cosL + x[1])*sina*cosb*(1.0/q))*accel
    dgb = sqrtpu *(sina*sinb*cosL - ((q + 1.0)*sinL + x[2])*(1.0/q)*cosa*sinb + (x[3]*sinL - x[4]*cosL)*(x[1]/q)*cosb)*accel
    maxg = sqrtpu * (accel/q)*vf.sqrt(q**2*sinL**2 + ((q+1.0)*sinL + x[2])**2 + (x[3]*sinL - x[4]*cosL)**2 * x[1]**2)
    
    dha = 0
    dhb = sqrtpu * (s**2*cosL/(2.0*q))*cosb*accel
    maxh = sqrtpu * (s**2*cosL/(2.0*q))*accel
    
    dka = 0
    dkb = sqrtpu * (s**2*sinL/(2.0*q))*cosb*accel
    maxk = sqrtpu * (s**2*sinL/(2.0*q))*accel
    
    maxhkL = sqrtpu * (s**2/(2.0*q))*accel
'''
def func(x):
    return 1

def LyapSteer(tab):
    mu = 1.0
    args = Args(10)
    x = args.head(6)
    t = args[6]

    
    L = vf.cross(x.head(3), x.tail(3))
    A = vf.cross(x.tail(3), vf.cross(x.head(3), x.tail(3))) - mu * x.head(3).normalized()
    energy = .5 * x.tail(3).squared_norm() - mu/(x.head(3).norm())
    
    TargtPosVel =  oc.InterpFunction(tab,range(0,6)).vf().eval(t)
    
    Lt = vf.cross(TargtPosVel.head(3), TargtPosVel.tail(3))
    At = vf.cross(TargtPosVel.tail(3), vf.cross(TargtPosVel.head(3),
                                                TargtPosVel.tail(3))) - mu * TargtPosVel.head(3).normalized()
    energyt = .5 * TargtPosVel.tail(3).squared_norm() - mu/(TargtPosVel.head(3).norm())
    
    DeltaL = L - Lt
    DeltaA = A - At
    
    k = 1.0

    F = -func(x) * (k*vf.cross(DeltaL, x.head(3)) + vf.cross(L, DeltaA) + vf.cross(vf.cross(DeltaA, x.tail(3)), x.head(3)))
    
    Q = vf.RowMatrix([x.head(3).normalized(),
                  vf.cross(vf.cross(x.head(3), x.tail(3)), x.head(3)).normalized(),
                  vf.cross(x.head(3), x.tail(3)).normalized()])
    
    Vnorm = (DeltaL.squared_norm()/L.squared_norm()) + (DeltaA.squared_norm()/A.squared_norm())
    u = F.normalized()
    return u
    
    

    
    
    
   