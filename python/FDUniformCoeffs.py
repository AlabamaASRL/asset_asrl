import numpy as np
import math
from fractions import Fraction


def cnt(ord, acc):
    p = int((2*math.floor((ord+1)/2) - 2 + acc)/2)
    idx = np.array([i for i in range(-p, p+1)])
    matList = []
    vecList = []
    for i in range(0, (2*p)+1):
        vecList.append(0)
        matList.append(list(idx**i))

    vecList[ord] = math.factorial(ord)
    vec = np.array(vecList)
    mat = np.array(matList)

    cff_float = np.linalg.solve(mat, vec)
    cff = []
    for f in cff_float:
        cff.append(Fraction(f).limit_denominator())

    return cff


def fwd(ord, acc):
    p = ord + acc
    idx = np.array([i for i in range(0, p)])
    matList = []
    vecList = []
    for i in idx:
        vecList.append(0)
        matList.append(list(idx**i))

    vecList[ord] = math.factorial(ord)
    vec = np.array(vecList)
    mat = np.array(matList)

    cff_float = np.linalg.solve(mat, vec)
    cff = []
    for f in cff_float:
        cff.append(Fraction(f).limit_denominator())

    return cff


def bck(ord, acc):
    return list(((-1)**ord)*np.flip(np.array(fwd(ord, acc))))


def arb(ord, idx):
    p = len(idx)
    matList = []
    vecList = []
    for i in range(0, p):
        vecList.append(0)
        matList.append(list(idx**i))

    vecList[ord] = math.factorial(ord)
    vec = np.array(vecList)
    mat = np.array(matList)

    cff_float = np.linalg.solve(mat, vec)
    cff = []
    for f in cff_float:
        cff.append(Fraction(f).limit_denominator())

    return cff
