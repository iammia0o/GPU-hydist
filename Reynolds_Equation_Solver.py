# this module contains all functions contribute to solve Reynold equations. 
# Input: 1: Initial Condition
#           - topological information, vector field u, v (u is x-component and v is y-component of velocity vector at dept h) 
#           - Water level z
#        2: Boundary condition: 
#           Values of u, v at boundary positions
#

from __future__ import division
import numpy as np 
import math as mth
from Coeff import *
from Global_Variables import *
import matplotlib.pyplot as plt

# In this module I will declare global variables that are used within this only
a1 = np.zeros(max(M, N) + 2)
b1 = np.zeros(max(M, N) + 2)
c1 = np.zeros(max(M, N) + 2)
d1 = np.zeros(max(M, N) + 2)
a2 = np.zeros(max(M, N) + 2)
b2 = np.zeros(max(M, N) + 2)
c2 = np.zeros(max(M, N) + 2)
d2 = np.zeros(max(M, N) + 2)
flist = [np.zeros(max(M, N) + 2) for x in range(7)]
f1, f2, f3, f5, f6, f7, f8 = flist[0], flist[1], flist[2], flist[3], flist[4], flist[5], flist[6]
# load boundary type to this arra
#print bienQ
x = np.zeros(max(M, N) * 2 + 2)
fx = open("x.txt", 'w')
varNo = max(M+2, N+2)*2
AA = np.zeros(varNo)
BB = np.zeros(varNo)
CC = np.zeros(varNo)
DD = np.zeros(varNo)
#ok
def bienrandau(l, r):
    # This can be changed using numpy library
    # print "bien ran dau"
    # varNo = max(M+2, N+2)*2
    # AA = np.zeros(varNo)
    # BB = np.zeros(varNo)
    # CC = np.zeros(varNo)
    # DD = np.zeros(varNo)
    for i in range (l, r + 1):
        AA[(i - l) * 2 + 1] = a2[i]
        BB[(i - l) * 2 + 1] = 1      # b2[i] = 1
        CC[(i - l) * 2 + 1] = c2[i]
        DD[(i - l) * 2 + 1] = d2[i]
        #print "d2[", i, "] =", d2[i]

        AA[(i - l + 1) * 2] = a1[i]
        BB[(i - l + 1) * 2] = b1[i]
        CC[(i - l + 1) * 2] = c1[i]
        DD[(i - l + 1) * 2] = d1[i]
        #print "d1[", i, "] =", d1[i]
    # return AA, BB, CC, DD

#ok
def bienlongdau(l, r):
    # print "bien long dau"
    # varNo = max(M+2, N+2)*2
    # AA = np.zeros(varNo)
    # BB = np.zeros(varNo)
    # CC = np.zeros(varNo)
    # DD = np.zeros(varNo)
    for i in range (l, r):
        AA[(i - l) * 2 + 1] = a1[i]
        BB[(i - l) * 2 + 1] = b1[i]  
        CC[(i - l) * 2 + 1] = c1[i]
        DD[(i - l) * 2 + 1] = d1[i]

        AA[(i - l + 1) * 2] = a2[i + 1]
        BB[(i - l + 1) * 2] = 1               # b2[i] = 1
        CC[(i - l + 1) * 2] = c2[i + 1]
        DD[(i - l + 1) * 2] = d2[i + 1]
        #print "DD = ", d2[i + 1]
    # return AA, BB, CC, DD

# ok
def update_abcd_at_l(l, r, f4, solidBound):
    bienran1, bienran2 = solidBound
    #print l, " ", r
    if r - l > 1:
        if bienran1:
            a1[l] = f4
            b1[l] = f2[l] - (f3[l] * f7[l + 1] / f6[l + 1])
            c1[l] = - f4 - (f3[l] / f6[l + 1])
            d1[l] = f5[l] - (f3[l] * f8[l + 1] / f6[l + 1])
            #print f5[l], " ", f8[l + 1], " ", f6[l + 1]," ", f3[l]
        else:
            a1[l] = f4 - f1[l] / f7[l]
            b1[l] = f2[l] - (f1[l] * f6[l] / f7[l]) - (f3[l] * f7[l + 1] / f6[l + 1])
            c1[l] = -f4 - (f3[l] / f6[l + 1])
            d1[l] = f5[l] - (f1[l] * f8[l] / f7[l]) - (f3[l] * f8[l + 1] / f6[l + 1])
        if r - l > 2:
            for i in range(l + 1, r - 1):
                a1[i] = f4 - f1[i] / f7[i]
                b1[i] = f2[i] - (f1[i] * f6[i] / f7[i]) - (f3[i] * f7[i + 1] / f6[i + 1])
                c1[i] = -f4 - (f3[i] / f6[i + 1])
                d1[i] = f5[i] - (f1[i] * f8[i] / f7[i]) - (f3[i] * f8[i + 1] / f6[i + 1])

        if bienran2:
            a1[r - 1] = f4 - f1[r -1] / f7[r -1]
            b1[r - 1] = f2[r - 1] - (f1[r - 1] * f6[r -1] / f7[r - 1])
            c1[r - 1] = - f4
            d1[r - 1] = f5[r - 1] - (f1[r - 1] * f8[r - 1] / f7[r - 1])
        else:
            a1[r - 1] = f4 - f1[r - 1] / f7[r - 1]
            b1[r - 1] = f2[r - 1] - (f1[r - 1] * f6[r - 1] / f7[r - 1]) - (f3[r - 1] * f7[r] / f6[r])
            c1[r - 1] = -f4 - (f3[r - 1] / f6[r])
            d1[r - 1] = f5[r - 1] - (f1[r - 1] * f8[r - 1] / f7[r - 1]) - (f3[r - 1] * f8[r] / f6[r])
    else:
        a1[l] = f4
        b1[l] = f2[l]
        d1[l] = f5[l]
        c1[l] = -f4

def boundary_config(isU, i_j, l, r, solidBound):
    bienran1, bienran2 = solidBound
    j = i_j
    if isU:
        ubp_or_vbt = ubp[j]
        ubt_or_vbd = ubt[j]
        TZ_r = t_z[r, j]
        TZ_l = t_z[l, j]
        dkBienQ_1 = bienQ[2]
        dkBienQ_2 = bienQ[3]
        dkfr = N
    else:
        ubp_or_vbt = vbt[j]
        ubt_or_vbd = vbd[j]
        TZ_r = t_z[j, r]
        TZ_l = t_z[j, l]
        dkBienQ_1 = bienQ[1]
        dkBienQ_2 = bienQ[0]
        dkfr = M
    #print "dk bien: ", dkBienQ_2, dkBienQ_1
    if bienran1:
        # attention
        # AA, BB, CC, DD = bienrandau(l, r)
        bienrandau(l, r)
        DD[1] = d2[l]
        # ran - long
        if bienran2 is False:
            if dkBienQ_2 and r == dkfr:         #r == dkfr:   # Kiem tra lai phan nay
                sn = 2 * (r - l) + 1
                # attention 
                AA[sn] = a2[r]
                BB[sn] = 1
                DD[sn] = d2[r] - c2[r] * ubp_or_vbt
            else:
                sn = 2 * (r - l) 
                AA[sn] = a1[r - 1]
                BB[sn] = b1[r - 1]
                DD[sn] = d1[r - 1] - c1[r - 1] * TZ_r
        # ran - ran
        else:
            sn = 2 * (r - l) + 1
            AA[sn] = a2[r]
            BB[sn] = 1
            DD[sn] = d2[r]
    # long
    else:
        if dkBienQ_1 and l == 2:
            # AA, BB, CC, DD = bienrandau(l, r)
            bienrandau(l, r)
            DD[1] = d2[l] - a2[l] * ubt_or_vbd
            # thieu bb[1] va cc[1] cho truong hop vz, hoi lai co
            isBienran = True
        else:
            # AA, BB, CC, DD = bienlongdau(l, r)
            bienlongdau(l, r)
            BB[1] = b1[l]
            CC[1] = c1[l]
            DD[1] = d1[l] - a1[l] * TZ_l
            isBienran = False
        # long - long
        if bienran2 is False: # variable isbienran is equivalent with variable text in original code
            if dkBienQ_1 and r == dkfr:     #r == dkfr: # BienQ[0] and r == M trong truong hop giaianv
                sn = 2 * (r - l)
                if isBienran:
                    sn += 1
                AA[sn] = a2[r]
                BB[sn] = 1
                DD[sn] = d2[r] - c2[r] * ubp_or_vbt
            else:
                sn = 2 * (r - l)
                if isBienran is False:
                    sn -= 1
                AA[sn] = a1[r - 1]
                BB[sn] = b1[r - 1]
                DD[sn] = d1[r - 1] - c1[r - 1] * TZ_r
        else:
            sn = 2 * (r - l)
            if isBienran is True:
                sn += 1
            # AA[sn] = f7[r] 
            # this line is modified for the canal case
            AA[sn] = a2[r]
            BB[sn] = 1
            DD[sn] = d2[r]
    if j == 3 and isU == False:
        print sn
        np.savetxt("/home/Pearl/mia/tmp/AA.txt", AA)
        np.savetxt("/home/Pearl/mia/tmp/BB.txt", BB)
        np.savetxt("/home/Pearl/mia/tmp/CC.txt", CC)
        np.savetxt("/home/Pearl/mia/tmp/DD.txt", DD)

    if sn > 0:
        if isU is False and j == 4:
            j = 0
        truyduoi(sn, AA, BB, CC, DD, j)
    return sn

# Thomas Algorithm solving diagonal matrix
# ok!
def truyduoi(sn, AA, BB, CC, DD, j=0):
    Ap = np.zeros(sn + 1)
    Bp = np.zeros(sn + 1)
    ep = np.zeros(sn + 1)
    Ap[1] = - CC[1] / BB[1]
    Bp[1] = DD[1] / BB[1]
    #print sn
    for i in range(2, sn):
        ep[i] = AA[i] * Ap[i - 1] + BB[i]
        Ap[i] = -CC[i] / ep[i]
        Bp[i] = (DD[i] - (AA[i] * Bp[i - 1])) / ep[i]
        #print "i =", i, "AA=", AA[i], "BB:", BB[i], "CC:", CC[i], "DD:", DD[i]

    x[sn] = (DD[sn] - (AA[sn] * Bp[sn - 1])) / (BB[sn] + (AA[sn] * Ap[sn - 1]))

    for i in range(sn - 1, 0, -1):
        x[i] = Bp[i] + (Ap[i] * x[i + 1])
        #print "i: ", i, "x:", '%.15f' % x[i]
    # np.savetxt(fx, x)
    # fx.write(" ")

def K_factor():
    for i in range(2, N + 1):
        for j in range(1, M + 1):
            if h[i - 1, j ] + h[i, j] != 0:
                Ky1[i, j] = g * mth.pow(mth.pow((h[i - 1, j] + h[i, j]) * 0.5, mu_mn), -2) * mth.pow(hsnham[i, j], 2)

    for j in range(2, M + 1):
        for i in range(1, N + 1):
            if h[i, j - 1] + h[i, j] != 0:
                Kx1[i, j] = g * mth.pow(mth.pow((h[i, j - 1] + h[i, j]) * 0.5, mu_mn), -2) * mth.pow(hsnham[i, j], 2)


# Need to double check this  as well 
# ok
def uzSolver(l, r, jpos, solidBound):
    j = jpos
    f4 = 2 * g * dTchia2dX
    # print "Giai an u: l, r = ", l, r
    for i in range(l, r):
        vtb = (v[i, j - 1] + v[i, j] + v[i + 1, j - 1] + v[i + 1, j]) * 0.25
        f1[i] = dTchia2dX * u[i, j] + VISCOINDX[i, j] * dT / dXbp
        f2[i] = -(2 + Kx1[i, j] * dT * sqrt(u[i, j] ** 2 + vtb ** 2) / Htdu[i, j] + (2 * dT * VISCOINDX[i, j]) / dXbp) # chua tinh muc nuoc trung binh
        f3[i] = dT * VISCOINDX[i, j] / dXbp - dTchia2dX * u[i, j] 
        #print f1[i], " ", f3[i], " ", u[i, j]

        if H_moi[i, j - 1] <= H_TINH:
            if vtb >= 0:
                p = 0
                q = 0
            else:
                p = vtb * (-3 * u[i, j] + 4 * u[i, j + 1] - u[i, j + 2]) / dY2
                q = (u[i, j] - 2 * u[i, j + 1] + u[i, j + 2]) / dYbp
        else:
            if H_moi[i, j + 1] <= H_TINH:
                if H_moi[i, j - 2] <= H_TINH or vtb <=0:
                    p = 0
                    q = 0
                else:
                    p = vtb * (3 * u[i, j] - 4 * u[i, j - 1] + u[i, j - 2]) / dY2
                    q = (u[i, j] - 2 * u[i, j - 1] + u[i, j - 2] ) / dYbp

            else:

                p = vtb * (u[i, j + 1] - u[i, j - 1]) / dY2
                q = (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dYbp

        f5[i] = -2 * u[i, j] + dT * p  - dT * f * vtb - dT * VISCOINDX[i, j] * q - dT * (Windx - Tsxw[i, j]) / Htdu[i, j]


    for i in range(l, r + 1):
        f6[i] = dTchia2dX * Htdu[i, j]
        f7[i] = - dTchia2dX * Htdu[i - 1, j]
        f8[i] = z[i, j] - dTchia2dY * (Htdv[i, j] * v[i, j] - Htdv[i, j - 1] * v[i, j - 1])
        a2[i] = f7[i]
        c2[i] = f6[i]
        d2[i] = f8[i] 

    update_abcd_at_l(l=l, r=r, f4=f4, solidBound=solidBound)

    sn = boundary_config(isU=True, i_j=j, l=l, r=r, solidBound=solidBound)


    bienran1, bienran2 = solidBound

    if bienran1:
        for i in range(l, r):
            t_z[i, j] = x[2 * (i - l) + 1]
            t_u[i, j] = x[2 * (i - l) + 2]

        t_u[l - 1, j] = 0
    else:
        if bienQ[2] and l == 2:
            for i in range(l, r):
                t_z[i, j] = x[2 * (i - l) + 1]
                t_u[i, j] = x[2 * (i - l) + 2]
            
            t_u[l - 1, j] = ubt[j]
        else:
            #print "long z1"
            t_u[l, j] = x[1]
            t_u[l - 1, j] = (d2[l] - t_z[l, j] - c2[l] * t_u[l, j]) / a2[l]
            for i in range(l + 1, r):
                t_z[i, j] = x[2 * (i - l)]
                t_u[i, j] = x[2 * (i - l) + 1]
                #print t_u[i, j]

    if bienran2:
        t_u[r, j] = 0
        t_z[r, j] = x[sn]
    else:
        if bienQ[3] and r == N:
            t_u[r, j] = ubp[j]
            t_z[r, j] = x[sn]
        else:
            #print "long z2"
            t_u[r, j] = (d2[r] - a2[r] * t_u[r - 1, j] - t_z[r, j]) / c2[r]
 



def vzSolver(l, r, ipos, solidBound):
    i = ipos
    f4 = 2 * g * dTchia2dY
    #print "giaian v: l, r: ", l, r
    for j in range(l, r):
        utb = (u[i - 1, j] + u[i, j] + u[i - 1, j + 1] + u[i, j + 1]) * 0.25
        f1[j] = dTchia2dY * v[i, j] + VISCOINDX[i, j] * dT / dYbp
        f2[j] = -(2 + Ky1[i, j] * dT * sqrt(v[i, j] ** 2 + utb ** 2) / Htdv[i, j] + (2 * dT * VISCOINDX[i, j]) / dYbp) # not exactly like the stated formular. Need to double check
        f3[j] = dT * VISCOINDX[i, j] / dYbp - dTchia2dY * v[i, j]  
        # print i, f1[j]

        if H_moi[i - 1, j] <= H_TINH:
            if utb >= 0:
                p = 0
                q = 0
            else:
                #print H_noi[i - 1, j], " ", i, " ", j
                q = utb * (-3 * v[i, j] + 4 * v[i + 1, j] - v[i + 2, j]) / dX2
                p = (v[i, j] - 2 * v[i + 1, j] + v[i + 2, j] ) / dXbp
        else:
            if H_moi[i + 1, j] <= H_TINH:
                if H_moi[i - 2, j] <= H_TINH or utb <=0:
                    q = 0
                    p = 0
                else:
                    q = utb * (3 * v[i, j] - 4 * v[i - 1, j] + v[i - 2, j]) /dX2
                    p = (v[i, j] - 2 * v[i - 1, j] + v[i - 2, j] ) / dXbp
            else:
                q = utb * (v[i + 1, j] - v[i - 1, j]) / dX2
                p = (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / dXbp

        f5[j] = -2 * v[i, j] + dT * q + dT * f * utb - dT * VISCOINDX[i, j] * p - dT * (Windy - Tsyw[i, j]) / Htdv[i, j]
        #print f5[j]
    #print l , " ", r
    for j in range(l, r + 1):
        f6[j] = dTchia2dY * Htdv[i, j]
        f7[j] = - dTchia2dY * Htdv[i, j - 1]
        f8[j] = z[i, j] - dTchia2dX * (Htdu[i, j] * u[i, j] - Htdu[i - 1, j] * u[i - 1, j])
        #print "z ", z[i, j], "Htdu ", Htdu[i, j], "u ", u[i, j]
        a2[j] = f7[j]
        c2[j] = f6[j]
        d2[j] = f8[j]


    update_abcd_at_l(l=l, r=r, f4=f4, solidBound=solidBound)

    sn = boundary_config(isU=False, i_j=i, l=l, r=r, solidBound=solidBound)

    # if (i == 3):
        # np.savetxt("/home/Pearl/mia/tmp/DD.txt", x)
    # print sn


    bienran1, bienran2 = solidBound
    if bienran1:
        for j in range(l, r):
            t_z[i, j] = x[2 * (j - l) + 1]
            t_v[i, j] = x[2 * (j - l) + 2]

        t_v[i, l - 1] = 0
    else:
        if bienQ[1] and l == 2:
            for j in range(l, r):
                t_z[i, j] = x[2 * (j - l) + 1]
                t_v[i, j] = x[2 * (j - l) + 2]
            t_v[i, l - 1] = vbd[i]
        else:
            t_v[i, l] = x[1]
            t_v[i, l - 1] = (d2[l] - t_z[i, l] - c2[l] * t_v[i, l]) / a2[l]
            for j in range(l + 1, r):
                t_z[i, j] = x[2 * (j - l)]
                t_v[i, j] = x[2 * (j - l) + 1]

    if bienran2:
        t_v[i, r] = 0
        t_z[i, r] = x[sn]
    else:
        if bienQ[0] and r == M:
            t_v[i, r] = vbt[j]
            t_z[i, r] = x[sn]
        else:
            t_v[i, r] = (d2[r] - a2[r] * t_v[i, r - 1] - t_z[i, r]) / c2[r]

def uSolver(l, r, jpos, solidBound):


    bienran1, bienran2 = solidBound
    j = jpos
    for i in range (l, r):
        vtb = (v[i, j - 1] + v[i, j] + v[i + 1, j - 1] + v[i + 1, j]) * 0.25
        t_vtb = (t_v[i, j - 1] + t_v[i, j] + t_v[i + 1, j - 1] + t_v[i + 1, j]) * 0.25
        p = (u[i + 1, j] - u[i - 1, j]) / dX2
        p = (HaiChiadT + p + Kx1[i, j] * sqrt(vtb **2 + u[i, j] **2) / Htdu[i, j])
        #print vtb, ' ', t_vtb
        if H_moi[i, j - 1] <= H_TINH:
            if (vtb >= 0):
                q = 0
                tmp = 0
                #print 'H_moi <= H_TINH and vtb >= 0'
            else:
                q = t_vtb * (-3 * u[i, j] + 4 * u[i, j + 1] - u[i, j + 2]) / dY2
                tmp = (u[i, j] - 2 * u[i, j + 1] + u[i, j + 2] ) / dYbp
                #print 'H_moi <= H_TINH and vtb < 0'
        else:
            if H_moi[i, j + 1] <= H_TINH:
                    if H_moi[i, j - 2] <= H_TINH or vtb <=0:
                        tmp = 0
                        q = 0
                        #print 'H_moi[i, j+1] <= H_TINH and vtb < 0'
                    else:
                        q = t_vtb * (3 * u[i, j] - 4 * u[i, j - 1] + u[i, j - 2]) /dY2
                        tmp = (u[i, j] - 2 * u[i, j - 1] + u[i, j - 2] ) / dYbp

            else:
                q = t_vtb * (u[i, j + 1] - u[i, j - 1]) / dY2
                tmp = (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dYbp
                #print 'q is calculated in line 395'
        #print q
        q = HaiChiadT * u[i, j] - q + f * t_vtb
        q = (q - g * (z[i + 1, j] - z[i, j]) / dX + VISCOINDX[i, j] * ((u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dXbp + tmp)) + (Windx - Tsxw[i, j]) / Htdu[i, j]
        #print ' ', dX, ' ', dXbp, ' ', Htdu[i, j]
        t_u[i, j] = q / p

    if bienran1:
        t_u[l - 1, j]  = 0
    else:
        t_u[l - 1, j] = 2 * t_u[l, j] - t_u[l + 1, j]
    if bienran2:
        t_u[r, j] = 0
    else:
        t_u[r, j] = 2 * t_u[r - 1, j] - t_u[r - 2, j]


def vSolver(l, r, ipos, solidBound):

    bienran1, bienran2 = solidBound
    #print "giai hien v: l , r = ", l , " ", r
    i = ipos
    for j in range(l, r):
        utb = (u[i - 1, j] + u[i, j] + u[i - 1, j + 1] + u[i, j + 1]) * 0.25
        t_utb = (t_u[i - 1, j] + t_u[i, j] + t_u[i - 1, j + 1] + t_u[i, j + 1]) * 0.25
        p = (v[i, j + 1] - v[i, j - 1]) / dY2
        p = (HaiChiadT + p + Ky1[i, j] * sqrt(utb ** 2 + v[i, j] ** 2) / Htdv[i, j])
        #print "H_moi", H_moi[i - 1, j], " ", i, " ", j
        if H_moi[i - 1, j] <= H_TINH:
            #print H_moi[i - 1, j], " ", i, " ", j
            if utb >= 0:
                tmp = 0
                q = 0
            else:
                #print "here74", i
                q = t_utb * (-3 * v[i, j] + 4 * v[i + 1, j] + v[i + 2, j]) / dX2
                tmp = (v[i, j] - 2 * v[i + 1, j] + v[i + 2, j] ) / dXbp
        else:
            #print "here 17", i , j
            if H_moi[i + 1, j] <= H_TINH:
                if H_moi[i - 2, j] <= H_TINH or utb <= 0:
                    tmp = 0
                    q = 0
                else:
                    #print "here83", i, j
                    q = t_utb * (3 * v[i, j] - 4 * v[i - 1, j] + v[i - 2, j]) /dX2
                    tmp = (v[i, j] - 2 * v[i - 1, j] + v[i - 2, j] ) / dXbp
            else:
                #print "here87", i, j
                q = t_utb * (v[i + 1, j] - v[i - 1, j]) / dX2
                tmp = (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / dXbp
        #if i == 4: print "q_1: ", q, i, j
        q = HaiChiadT * v[i, j] - q - f * t_utb
        q = (q - g * (z[i, j + 1] - z[i, j]) / dY + VISCOINDX[i, j] * (tmp + (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) / dYbp)) + (Windy - Tsyw[i, j]) / Htdv[i, j]
        #if i == 4: print "p, q, tv ", p, ' ', q, ' ', utb, ' ', t_utb

        t_v[i, j] = q / p
        #if i == 4: print 'p, q:', p, q, i, j, ":", t_v[i, j]

    if bienran1:
        t_v[i, l - 1] = 0 
    else:
        t_v[i, l - 1] = 2 * t_v[i, l] - t_v[i, l + 1]
    if bienran2:
        t_v[i, r] = 0
    else:
        t_v[i, r] = 2 * t_v[i, r - 1] - t_v[i, r - 2]

    

def Normalize_UV(coeff, isU):
    if isU:
        t_uv = tu
        
        uv = np.zeros((N + 3, M + 3))
        for j in range(2, M + 1):
            for i in range(2, N + 1):
                if t_uv[i, j] != 0 and i == N:
                    uv[i, j] = coeff * t_uv[i, j] + (1 - coeff) * np.average(t_uv[i, j - 4 : j])
                if t_uv[i, j] != 0 and i < N:
                    uv[i, j] = coeff * t_uv[i, j] + (1 - coeff) * (t_uv[i - 1, j] + t_uv[i + 1, j] + t_uv[i, j - 1] + t_uv[i, j + 1]) * 0.25
    else:
        t_uv = tv
    
        uv = np.zeros((N + 3, M + 3))
        for i in range(2, N + 1):
            for j in range(2, M + 1):
                if t_uv[i, j] != 0 and j == M:
                    uv[i, j] = coeff * t_uv[i, j] + (1 - coeff) * np.average(t_uv[i, j - 4 : j])
                if t_uv[i, j] != 0 and j < M:
                    uv[i, j] = coeff * t_uv[i, j] + (1 - coeff) * (t_uv[i - 1, j] + t_uv[i + 1, j] + t_uv[i, j - 1] + t_uv[i, j + 1]) * 0.25
                

    for i in range(2, N + 1):
        for j in range(2, M + 1):
            #uv[i, j] = coeff * t_uv[i, j] + (1 - coeff) * np.average(t_uv[i - 4 : i, j])
            t_uv[:] = uv[:]

    

def Normalize_U(coeff):
    Normalize_UV(coeff, isU=True)


def Normalize_V(coeff):
    Normalize_UV(coeff, isU=False)
        
