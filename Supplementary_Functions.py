from __future__ import division
import numpy as np 
from Coeff import *
from Global_Variables import *
import math as mth

# This module contains functions that can be divided into 2 parts: 
# Part I: Load inputs from files and initialize values
# Part II: Updates values after each time step


########################################################################################################################
#             This part is dedicated for loading input and initializing values                                         #                                                                                                                    #
########################################################################################################################

# Load intial topological information
# h: depth of each unit in calculation mesh
def TinhKhoUot():
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if (h[i - 1, j - 1] + h[i - 1, j] + h[i, j - 1] + h[i, j]) * 0.25 > NANGDAY:
                khouot[i, j] = 0
                H_moi[i, j] = htaiz[i, j]


def Initial_condition(U_file, V_file, Z_file ):
    # This can be serve for the case that all calculations are done from scratch or from previous calculations
    # In the case that calculations are inherited from previous calculations, values of u, v, z calculation 
    # U, V, Z files contain arrays that have same shape with topographical map. 
    global u, v, z
    tmp = np.transpose(np.loadtxt(U_file))
    tmp = np.flip(tmp, 1)
    u = np.pad(tmp, ((1, 1), (1, 1)), 'edge')
    tmp = np.transpose(np.loadtxt(V_file))
    tmp = np.flip(tmp, 1)
    v = np.pad(tmp, ((1, 1), (1, 1)), 'edge')
    tmp = np.transpose(np.loadtxt(Z_file))
    tmp = np.flip(tmp, 1)
    z = np.pad(tmp, ((1, 1), (1, 1)), 'edge')


# In this load boundary version, I only program for a river segment that is non-branched. 
# When solving problem with branched river segment, future developer need to update this load boundary funtion. 
def load_boundary_condition(inputDir, limits, isZ):
    # read file to boundary_values
    # do some manipulation to generate v_boundary
    # return boundary_values and generate v_boundary so that they can be used for updating boundary condition in very time step
    # variableType: 0: z boundary. 1: u or v boundary 
    l, r = limits
    # Nhap cac thong so trong file dkbien
    # 0.02 could be change due to condition
    
    global boundary_values
    boundary_values = np.loadtxt(inputDir)
    hi = np.zeros(M)
    qi = np.zeros(M)
    hi[l:r] = h[l:r, M] - NANGDAY
    denominator = reduce((lambda x: mth.pow(x, 5 / 3) / (dX * 0.02)), hi[l:r])
    tmp = np.zeros(time)
    tmp = boundary_values / denominator
    for i in range(0, time + 1):
        qi[l:r] = tmp[i] * np.power(h[l:r], 5 / 3) / (dX * 0.02) 
        v_boundary[i, l:r] = qi[l:r] / hi[l:r]
    return (z_boundary, v_boundary)


# this is to load boundary condiitions
def _Input_bienchuoisolieu(inputDirs):
    indx = 0
    lim = {(dauj[M, 0], cuoij[M, 0]), (dauj[2, 0], cuoij[2, 0]), (dauj[2, 0], cuoij[2, 0]), (dauj[N, 0], cuoij[N, 0])}
    for i in range(0, 4):
        if boundaryType[i]:
            boundary[i], v_boundary[i] = load_boundary_condition(inputDirs[indx], lim[i], boundaryType[i] % 2)
            indx +=1

# phan nay de chay ra ket qua bai toan kenh thang


def Find_Calculation_limits(): # Danhdaumang in original program
    for i in range(2, N + 1):
        #print khouot[i]
        tmp = np.where(khouot[i] == 0)[0]

        segments = 0
        if tmp.size > 0:
            daui[i, 0] = max(tmp[0], 2)
            for j in range(0, tmp.size - 1):
                if tmp[j + 1] - tmp[j] - 1 != 0:
                    cuoii[i, segments] = tmp[j]
                    segments += 1
                    daui[i, segments] = tmp[j + 1]
            cuoii[i, segments] = min(tmp[-1], M)
            segments += 1
        moci[i] = segments

    for i in range(2, M + 1):
        tmp = np.where(khouot[:, i] == 0)[0]
        segments = 0
        if tmp.size > 0:
            dauj[i, 0] = max(tmp[0], 2)
            for j in range(0, tmp.size - 1):
                if tmp[j + 1] - tmp[j] - 1 != 0:
                    cuoij[i, segments] = tmp[j]
                    segments += 1
                    dauj[i, segments] = tmp[j + 1]
            cuoij[i, segments] = min(tmp[-1], N)
            segments += 1
        mocj[i] = segments 


########################################################################################################################
#             This part is dedicated for updating values after each time step                                          #
#                                                                                                                      #
########################################################################################################################

# Update relative factor at each time step
def Htuongdoi():
    #print z.shape
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            #print i, " ", j
            Htdu[i, j] = (h[i, j - 1] + h[i, j] + z[i + 1, j] + z[i, j]) * 0.5
            Htdv[i, j] = (h[i - 1, j] + h[i, j] + z[i, j + 1] + z[i, j]) * 0.5


def GiatriHtaiZ():
    for i in range(2, N + 1):
        for j in range(2, M + 1):
            htaiz[i, j] = (h[i - 1, j - 1] + h[i - 1, j] + h[i, j - 1] + h[i, j]) * 0.25

    for j in range(2, M + 1):
        htaiz[1, j]  = (h[1, j - 1] + h[1, j]) * 0.5

    htaiz[2 : N + 1, 1] = htaiz[2 : N + 1, 2]
    #htaiz_bd = np.copy (htaiz)



def VTHZ():
    for i in range(2, N + 1):
        for j in range (2, M + 1):
            if H_moi[i, j] > H_TINH:
                ut[i, j] = (tu[i - 1, j] + tu[i, j]) * 0.5
                vt = (tv[i, j - 1] + tv[i, j]) * 0.5
                VTH[i, j] = mth.sqrt(ut[i, j]**2 + vt[i, j]**2)
                if ut[i, j] != 0:
                    angle = 180 * mth.atan(abs(vt[i, j] / ut[i, j])) * pi**-1
                    if vt[i, j] * ut[i, j] > 0:
                        if vt[i, j] < 0:
                            angle = 180 + angle
                        else:
                            angle = 180 - angle
                    elif vt[i, j] * ut[i, j] < 0: 
                        if vt[i, j] < 0:
                            angle = 360 - angle

                else:
                    if vt[i, j] < 0:
                        angle = 270
                    elif vt[i, j] == 0:
                        angle = 0
                    else:
                        angle = 90
                goc[i, j] = angle
            else:
                VTH[i, j] = 0




# this equal to GiatriBien function in the original program
# this will return 
# this is for kenh thang only
# 

def export_Result(time, canal):
    outDir = 'Outputs/Hydraulic_Calculation/'
    hour = time // 3600
    
    np.savetxt(outDir + 'u_' + str(hour) + '.txt', u, fmt='%.5e', delimiter=' ')
    np.savetxt(outDir + 'v_' + str(hour) + '.txt', v, fmt='%.5e', delimiter=' ')
    np.savetxt(outDir + 'z_' + str(hour) + '.txt', z, fmt='%.5e', delimiter=' ')

