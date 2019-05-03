from __future__ import division

import numpy as np
import math as mth
import matplotlib as plt
from Reynolds_Equation_Solver import *
from Coeff import *
from Global_Variables import *
from Supplementary_Functions import *
from Load_Boundary_Conditions import *
import time
import matplotlib.pyplot as plt
# input need : topography of the river, velocity field u, v, water level z
# Boundary condition: change after each iteration
# global variables: N, M: size of calculation mesh.
# u, v, z: velocity in x, y direction, water level
# Phase I: Translate from VB6. In this phase, all variable names will be kept in Vietnamese. Changes can be made later on for clearer understanding


# dong mo o
def Reset_state(isI):
    
    # this snippet is corresponding to GiatriH_Chuan() in the original code
    for i in range(1, N + 1):
    	cond = np.where(khouot[i] == 0)[0]
    	H_moi[i][cond] = htaiz[i][cond] + t_z[i][cond]
        #print "tz tai ", i, " = ", t_z[i]
    #print "h_moi tai ", i , " =", H_moi[] 

    
    # This part is corresponding to dongmoo() in the original code
    # isI mean dongmoj
    if isI:                           
        for i in range(1, N + 1):
            for j in range(2, M + 1):
                if t_z[i, j] > z[i, j]:
                    if khouot[i, j - 1] == 1:
                        t_z[i, j - 1] = t_z[i, j]
                        H_moi[i, j - 1] = htaiz[i, j - 1] + t_z[i, j]
                        khouot[i, j - 1] = 0
                if t_z[i, j] > z[i, j]:
                    if khouot[i, j + 1] == 1:
                        t_z[i, j + 1] = t_z[i, j]
                        H_moi[i, j + 1] = htaiz[i, j + 1] + t_z[i, j]
                        khouot[i, j + 1] = 0
        # update tz in boundaries
        #t_z[0] = t_z[1]
        #t_z[N + 1] = t_z[N]
        #t_z[:, 0] = t_z[:, 1]
        #t_z[:, M + 1] = t_z[:, M]
    else:
        for i in range(2, N + 1):
            for j in range(1, M + 1):
                if t_z[i, j] > z[i, j]:
                    if khouot[i - 1, j] == 1:
                        t_z[i - 1, j] = t_z[i, j]
                        H_moi[i - 1, j] = htaiz[i - 1, j] + t_z[i -1, j]
                        khouot[i - 1, j] = 0
                if t_z[i, j] > z[i, j]:
                    if khouot[i + 1, j] == 1:
                        t_z[i + 1, j] = t_z[i, j]
                        H_moi[i + 1, j] = htaiz[i + 1, j] + t_z[i + 1, j]
                        khouot[i + 1, j] = 0
        #t_z[0] = t_z[1]
        #t_z[:, 0] = t_z[:, 1]
        #t_z[:, M + 1] = t_z[:, M]

    for i in range(2, N + 1):
        for j in range(2, M + 1):
            if H_moi[i, j] <= H_TINH and khouot[i, j]  == 0:
                t_u[i -1, j] = 0
                t_u[i, j] = 0
                t_v[i, j - 1] = 0
                t_v[i, j] = 0
                khouot[i, j] = 1

def set_boundary(is_i, i_j, k):
    bienran1 = False
    bienran2 = False
    if is_i: 
        i = i_j
        dau = daui[i, k]
        cuoi = cuoii[i, k]
        if (dau > 2) or (dau == 2 and (h[i, dau - 1] + h[i - 1, dau - 1]) * 0.5 == NANGDAY):
            bienran1 = True 
        if (cuoi < M) or (cuoi == M and (h[i, cuoi] + h[i - 1, cuoi]) * 0.5 == NANGDAY):
            bienran2 = True
    else:
        j = i_j
        dau = dauj[j, k]
        cuoi = cuoij[j, k]
        if (dau > 2) or (dau == 2 and (h[dau - 1, j] + h[dau - 1, j - 1]) * 0.5 == NANGDAY):
            bienran1 = True
        if (cuoi < N) or (cuoi == N and (h[cuoi, j] + h[cuoi, j - 1]) * 0.5 == NANGDAY):
            bienran2 = True
    return (dau, cuoi, (bienran1, bienran2))


def update_uvz():
    u[:] = t_u[:] * (1 - kenhhepd)
    v[:] = t_v[:] * (1 - kenhhepng)     #chinh sua tam thoi cho ep bien
    z[:] = t_z[:]


#t_start: the time that calculation start, by default equals 0.
# t_start can be passed by user if the calculation is ressumed from some previous calculation
def cpu_hydraulic_Calculation(days, hours, mins, canal=0, t_start=0, sec=0.5, plot=False):

    Tmax = (days*24*60 + hours*60 + mins)* 60 + sec
    print ("Tmax = ", Tmax)

    print "cpu"
    with open('DirsOfOutputs.txt', 'r') as inf:
        Dirs = dict([line.split() for line in inf])
    fz = open(Dirs['z_V'], 'w')
    fu = open(Dirs['u_V'], 'w')
    fv = open(Dirs['v_V'], 'w')
    print ("N, M : ",  N," ", M)
    t = t_start
    count = 1;

    while t < Tmax:
        #print "begining of a time step", count; count += 1
        # determine Boundary condition at time step t   
        t = t + dT * 0.5      
        
        Boundary_at(t, canal)

        # maybe there can be conflict of the starting indices of mocj and moci array 
        # bwt old and new function
        # ep bien kenh dung
        start_idx = 2
        end_indx = M + 1
        if canal and (kenhhepd == 1):
            t_u[1 : N, 2] = 0
            t_u[1 : N, M] = 0
            start_idx = 3
            end_indx = M

        
        #---------------------------------CPU code---------------------------------------------
        for j in range(start_idx, end_indx): 
            if mocj[j] > 0 :
                for k in range(mocj[j]):
                    dau, cuoi, solidBound = set_boundary(is_i=False, i_j=j, k=k)
                    uzSolver(l=dau, r=cuoi, jpos=j, solidBound=solidBound)
       

        for i in range(2, N + 1):
            if moci[i] > 0: 
                for k in range(moci[i]):
                    dau, cuoi, solidBound = set_boundary(is_i=True, i_j=i, k=k)
                    vSolver(l=dau, r=cuoi, ipos=i, solidBound=solidBound)

        

        #Normalize_V(coeff=heso)

        # attention
        Reset_state(True)

        update_uvz()
        
        #np.savetxt(fz, np.transpose(z[2 : N + 1, 4]))o


        # variables change: moci/ mocj, daui, cuoii, dauj, cuoij. Can actually get rid of these annoying variables
        # Will use this for testing other functions. 
        Find_Calculation_limits() 
        Htuongdoi()
        #hesoK()
        #biennongdophuongi()

       
        t = t + dT*0.5
        # attention
        # =giatribien in original VB code. Update boundary condition
        Boundary_at(t, canal)
        # ep bien kenh ngang
        start_idx = 2
        end_indx = N + 1
        if canal and (kenhhepng == 1):
            t_v[2, 1 : M] = 0
            t_v[N, 1 : M] = 0
            start_idx = 3
            end_indx = N

        # second half of time step
        for i in range(start_idx, end_indx):
            if moci[i] > 0:
                for k in range(moci[i]):
                    dau, cuoi, solidBound = set_boundary(is_i=True, i_j=i, k=k)
                    vzSolver(l=dau, r=cuoi, ipos=i, solidBound=solidBound)




        #Normalize_V(coeff=heso)
        for j in range(2, M+1):
            if mocj[j] > 0:
                for k in range(mocj[j]):
                    dau, cuoi, solidBound = set_boundary(is_i=False, i_j=j, k=k)
                    uSolver(l=dau, r=cuoi, jpos=j, solidBound=solidBound)

      
        Reset_state(False)

        update_uvz()
        
        #print(t)
        #np.savetxt(fz, np.transpose(z[2 : N + 1, 4]))
        
        # attention
        Find_Calculation_limits()
        Htuongdoi()
        #VTHZ()
        #hesoK()
        #export_Result(t, canal)
        #print z[2, 4], " ", z[N, 4]

        #biennongdophuongj()
        # xuatkq:
        if plot:
            #time.sleep(2)
            print (t)
            # xuat z de kiem tra
            print ('z')
            fig1 = plt.figure()
            #plt.plot(z[4, 2 : N + 1])
            plt.plot(z[2 : N + 1, 4])
            plt.xlim(0, 400)
            plt.ylim(-0.02, 0.02)
            plt.show()
            
            #xuat u hoac v de kiem tra
            #print('u')
            #fig2 = plt.figure()
            #plt.plot(u[2 : N + 1, 4])
            #plt.xlim(0, 400)
            #plt.ylim(-0.04, 0.04)
            #plt.show()
            #fig.savefig('plot.png')
            # load some global arrays
        #print "e-16", '%.16f' % np.power(2.70134219723423422, 2.7013421972342342)
        
