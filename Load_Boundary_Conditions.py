from __future__ import division
import numpy as np 
from Global_Variables import *
import math as mth

def Input_bienchuoisolieu():
	pass
    
    #lim = [(dauj[M, 0], cuoij[M, 0]), (dauj[2, 0], cuoij[2, 0]), (daui[2, 0], cuoii[2, 0]), (daui[N, 0], cuoii[N, 0])]
    #bientrai[lim[2][1]:lim[2][2]] = np.loadtxt("Inputs/bientrai.txt")i
    #bienphai[lim[3][1]:lim[3][2]] = np.loadtxt("Inputs/bienphai.txt")



def boundary_cannal(t):
    t1 = t // 3600
    t2 = t / 3600 - t1 # phan le cua t / 3600
    t1 = int(t1)
    
    if bienQ[0]:
        for k in range(dauj[M, 0], cuoij[M, 0] + 1):
            vbt[k] = 0
    else:
        for k in range(dauj[M, 0], cuoij[M, 0] + 1):
            t_z[k, M] = 0.01 * cos(2 * np.pi / 27.75 * t)  * cos(2 * (np.pi / 100) *  (100 - dY / 2))
            t_z[k, M + 1] = 0.01 * cos(2 * np.pi / 27.75 * t)  * cos(2 * (np.pi / 100) *  (100 + dY / 2))
            #if t >= 6.75: print "tz[" ,k, M, "] = ", t_z[k, M] 
    
    if bienQ[1]:
        for k in range(dauj[2, 0], cuoij[2, 0] + 1):
            vbd[k] = 0
    else:
        for k in range(dauj[2, 0], cuoij[2, 0] + 1):
            
            t_z[k, 2] = 0.01 * cos(2 * np.pi / 27.75 * t ) * cos(2 * (np.pi / 100) * dY / 2)
            t_z[k, 1] = 0.01 * cos(2 * np.pi / 27.75 * t ) * cos(2 * (np.pi / 100) * (- dY) / 2)
            #if t >= 6.75: print "tz[", k, 2, "] = ", t_z[k, 2] 
    
    
    if bienQ[2]:
        for k in range(daui[2, 0], cuoii[2, 0] + 1):
            ubt[k] = 0
    else:
        for k in range(daui[2, 0], cuoii[2, 0] + 1):
            
            t_z[2, k] = 0.01 * cos(2 * np.pi / 27.75 * t ) * cos(2 * (np.pi / 100) * dX / 2)
            t_z[1, k] = 0.01 * cos(2 * np.pi / 27.75 * t ) * cos(2 * (np.pi / 100) * (- dX) / 2)
            #if t >= 6.75: print "tz[",2, k, "] = ", t_z[2, k]
            
            #t_z[2, k] = 0.01 * cos(2 * 3.141592654 / 27.75 * t) * cos(2 * 3.141592654 / 100 * dX / 2)
            #t_z[1, k] = 0.01 * cos(2 * 3.141592654 / 27.75 * t) * cos(2 * 3.141592654 / 100 * (-dX / 2))

        #t_z[1] = t_z[2]

    if bienQ[3]:
        for k in range(daui[N, 0], cuoii[N, 0] + 1):
            ubp[k] = 0
    else:
        for k in range(daui[N, 0], cuoii[N, 0] + 1):
            t_z[N, k] = 0.01 * cos(2 * np.pi / 27.75 * t)  * cos(2 * (np.pi / 100) *  (100 - dX / 2))
        #t_z[N + 1] = t_z[N] 
            t_z[N + 1, k] = 0.01 * cos(2 * np.pi / 27.75 * t)  * cos(2 * (np.pi / 100) *  (100 + dX / 2))
            #print "tz[", N, k, "] = ", t_z[N, k]
            
            #t_z[N, k] = 0.01 * cos(2 * 3.141592654 / 27.75 * t) * cos(2 * 3.141592654 / 100 * (100 - (dX / 2)))
            #t_z[N + 1, k] = 0.01 * cos(2 * 3.141592654 / 27.75 * t) * cos(2 * 3.141592654 / 100 * (100 + (dX / 2)))


    #cfg.boundaries[t1] + (cfg.boundaries[t1 + 1] - cfg.boundaries[t1]) * t2

def Boundary_at(t, cannal=0):
    #print "Boundary_at"
    if cannal == 1:
        boundary_cannal(t)
    #    print t_z[2]
    else:
        # check different types of boundaries here
        pass
