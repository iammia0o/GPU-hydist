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
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt


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



def _gpu_boundary_at(t, ctx, source, pc, channel):
    if channel:
        offset = M  + 3;
        m_offset = 5
        up = source.get_function("boundary_up")
        down = source.get_function("boundary_down")
        left = source.get_function("boundary_left")
        right = source.get_function("boundary_right")
        b_args = [np.float64(t), np.int32(m_offset), np.int32(M), np.int32(N), pc['bienQ'], pc['t_z'],\
             pc['daui'], pc['cuoii'], pc['dauj'], pc['cuoij'], pc['vbt'], pc['vbd'], pc['ubt'], pc['ubp']]
        up(*b_args, block=(min(1024, N), 1, 1), grid=(N // min(1024, N), 1, 1))
        down(*b_args, block=(min(1024, N), 1, 1), grid=(N // min(1024, N), 1, 1))
        left(*b_args, block=(min(1024, M), 1, 1), grid=(M // min(1024, M), 1, 1))
        right(*b_args, block=(min(1024, M), 1, 1), grid=(M // min(1024, M), 1, 1))
        ctx.synchronize()



def hydraulic_Calculation(days, hours, mins, pointers, ctx, blk_size=512, canal=0, t_start=0, sec=0, plot=False):

    Tmax = (days*24*60 + hours*60 + mins)* 60 + sec
    M1 = M + 3; N1 = N + 3

    block_u = (min(M1, blk_size), 1, 1)
    grid_u = (M1 // min(blk_size, M1) + 1, 1, 1)
    block_v = (min(N1, blk_size), 1, 1)
    grid_v = (N1 // min(blk_size, N1), 1, 1)

    block_2d = (min(blk_size, M1), 1, 1)
    grid_2d = (M1 // min(blk_size, M1) + 1, N1, 1)


    print ("Tmax = ", Tmax)
    with open('DirsOfOutputs.txt', 'r') as inf:
        Dirs = dict([line.split() for line in inf])
    fz = open(Dirs['z_V'], 'w')
    fu = open(Dirs['u_V'], 'w')
    fv = open(Dirs['v_V'], 'w')
    print ("N, M : ",  N," ", M)
    t = t_start
    count = 1;

    solvers = open("__Reynold_Equation_Solver_Kernel.cu").read()
    supplement = open("Supplementary_Functions_Kernels.cu").read()
    solvermod = SourceModule(solvers, include_dirs = ["/home/huongnm/gpuver_riverbank_failure_prediction"])
    supmod = SourceModule(supplement, include_dirs = ["/home/huongnm/gpuver_riverbank_failure_prediction"])

    gpu_uzSolver = solvermod.get_function("solveUZ")
    gpu_vzSolver = solvermod.get_function("SolveVZ")
    gpu_vSolver = solvermod.get_function("solveV")
    gpu_uSolver = solvermod.get_function("solveU")

    gpu_Mark_x = supmod.get_function("Find_Calculation_limits_Horizontal")
    gpu_Mark_y = supmod.get_function("Find_Calculation_limits_Vertical")
    gpu_Htuongdoi = supmod.get_function("Htuongdoi")
    gpu_update_uvz = supmod.get_function("update_uvz")
    gpu_reset_state_x = supmod.get_function("Reset_states_horizontal")   
    gpu_reset_state_y = supmod.get_function("Reset_states_vertical")

    pc = pointers.arg_list
    pd = pointers.device_only_ptrs
    mem_offset_u = N * 2 + 1
    mem_offset_v = M * 2 + 1
    start_idx = 2
    end_indx = M + 1
    if canal and (kenhhepd == 1):
       start_idx = 3
       end_indx = M
    uzSolver_args = [ np.int32(M), np.int32(N), np.int32(mem_offset_u), np.int32(start_idx), np.int32(end_indx), pc['mocj'], pc['dauj'], pc['cuoij'], pc['bienQ'], pc['Tsxw'],\
                        pc['v'], pc['u'], pc['z'], pc['Htdu'], pc['Htdv'], pc['VISCOINDX'], pc['t_u'], pc['t_z'], pc['h'], pc['ubt'], pc['ubp'], \
                        pc['H_moi'], pc['Kx1'], pd['f1'], pd['f2'], pd['f3'], pd['f5'], pd['f6'], pd['f7'], pd['f8'],\
                        pd['a1'], pd['b1'], pd['c1'], pd['d1'], pd['a2'], pd['b2'], pd['c2'], pd['d2'], pd['AA'], pd['BB'], pd['CC'], pd['DD'], \
                        pd['Ap'], pd['Bp'], pd['ep'], pd['x']]
    start_idx = 2
    end_indx = N + 1
    if canal and (kenhhepng == 1):
       start_idx = 3
       end_indx = N
    vzSolver_args = [ np.int32(M), np.int32(N), np.int32(mem_offset_v), np.int32(start_idx), np.int32(end_indx),pc['moci'], pc['daui'], pc['cuoii'], pc['bienQ'], pc['Tsyw'], \
                        pc['v'], pc['u'], pc['z'], pc['Htdu'], pc['Htdv'], pc['VISCOINDX'], pc['t_v'], pc['t_z'], pc['h'], pc['vbt'], pc['vbd'], \
                        pc['H_moi'], pc['Ky1'], pd['f1'], pd['f2'], pd['f3'], pd['f5'], pd['f6'], pd['f7'], pd['f8'],\
                        pd['a1'], pd['b1'], pd['c1'], pd['d1'], pd['a2'], pd['b2'], pd['c2'], pd['d2'], pd['AA'], pd['BB'], pd['CC'], pd['DD'], \
                        pd['Ap'], pd['Bp'], pd['ep'], pd['x']]

    vSolver_arg = [np.int32(N), np.int32(M), np.int32(2), np.int32(N), pc['VISCOINDX'], pc['Tsyw'], pc['moci'], pc['daui'], pc['cuoii'], \
                    pc['v'], pc['t_v'], pc['u'], pc['t_u'], pc['z'], pc['t_z'], pc['Ky1'], pc['Htdv'], pc['H_moi'], pc['h']]

    uSolver_arg = [np.int32(N), np.int32(M), np.int32(2), np.int32(M), pc['VISCOINDX'], pc['Tsxw'], pc['mocj'], pc['dauj'], pc['cuoij'], \
                        pc['v'], pc['t_v'], pc['u'], pc['t_u'], pc['z'], pc['t_z'], pc['Kx1'], pc['Htdu'], pc['H_moi'], pc['h']]
    reset_arg  = [np.int32(M),np.int32(N), pc['H_moi'], pc['htaiz'], pc['khouot'], pc['z'], pc['t_z'], pc['t_u'], pc['t_v']]


    update_arg = [np.int32(M), np.int32(N), pc['u'], pc['v'], pc['z'], pc['t_u'], pc['t_v'], pc['t_z'], np.int32(kenhhepd), np.int32(kenhhepng)]

    markx_arg = [np.int32(5), np.int32(M), np.int32(N), pc['daui'], pc['cuoii'], pc['moci'], pc['khouot']]
    marky_arg = [np.int32(5), np.int32(M), np.int32(N), pc['dauj'], pc['cuoij'], pc['mocj'], pc['khouot']]

    Htuongdoi_arg = [np.int32(M), np.int32(N), pc['Htdu'], pc['Htdv'], pc['z'], pc['h']]

    gpu_tz = np.zeros(t_z.shape)
    gpu_tu = np.zeros(t_u.shape)
    gpu_tv = np.zeros(t_v.shape)
    gpu_z = np.zeros(z.shape)
    gpu_u = np.zeros(u.shape)
    gpu_v = np.zeros(v.shape)
    gpu_htdu = np.zeros(Htdu.shape)
    gpu_htdv = np.zeros(Htdv.shape)
    gpu_daui = np.zeros(daui.shape).astype(np.int32)
    gpu_cuoii = np.zeros(cuoii.shape).astype(np.int32)
    gpu_dauj = np.zeros(dauj.shape).astype(np.int32)
    gpu_cuoij = np.zeros(cuoij.shape).astype(np.int32)
    gpu_khouot = np.ones(khouot.shape).astype(np.int32)

    while t < Tmax:
        
        t = t + dT * 0.5      
        #---------------------------------CPU code---------------------------------------------
        Boundary_at(t, canal)
        
        start_idx = 2
        end_indx = M + 1
        if canal and (kenhhepd == 1):
            t_u[1 : N, 2] = 0
            t_u[1 : N, M] = 0
            start_idx = 3
            end_indx = M
            tmp = SourceModule("""__global__ void epbien(int offset, int M, int N, double* t_u){
                    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
                    t_u[i * offset + 2] = 0;
                    t_u[i * offset + M] = 0;
                } """)
            epbien = tmp.get_function("epbien")
            epbien(np.int32(M + 3), np.int32(M), np.int32(N), pc["t_u"], block=(N - 1, 1, 1))
        
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
        Reset_state(True)
        update_uvz()
        Find_Calculation_limits() 
        Htuongdoi()

        #------------------------------GPU code-------------------------------------------------
        _gpu_boundary_at(t, ctx, supmod, pc, canal)
        ctx.synchronize()

        gpu_uzSolver(*uzSolver_args, block=block_u, grid=grid_u)
        ctx.synchronize()
       
        gpu_vSolver(*vSolver_arg, block=block_v, grid=grid_v)
       
        ctx.synchronize();
       
        
        gpu_reset_state_x(*reset_arg, block=block_2d, grid=grid_2d)
        ctx.synchronize();
        
        gpu_update_uvz(*update_arg, block=block_2d, grid=grid_2d)
        ctx.synchronize()
       


        gpu_Mark_x(*markx_arg, block=block_u, grid=grid_u)
        gpu_Mark_y(*marky_arg, block=block_u, grid=grid_v)
        ctx.synchronize();

        gpu_Htuongdoi(*Htuongdoi_arg, block=block_2d, grid=grid_2d)
        ctx.synchronize();

        pointers.extract({"u": gpu_u, "v" : gpu_v, "z" : gpu_z})
        pointers.extract({"t_z" : gpu_tz, "t_u" : gpu_tu, "t_v": gpu_tv, "u": gpu_u, "v" : gpu_v, "z" : gpu_z, "Htdu" : gpu_htdu, "Htdv" : gpu_htdv,\
        "daui" : gpu_daui, "cuoii" : gpu_cuoii, "dauj" : gpu_dauj, "cuoij" : gpu_cuoij, "khouot" : gpu_khouot })
        err_tz = np.where(abs(gpu_tz - t_z) > 1e-13)
        err_tu = np.where(abs(gpu_tu - t_u) > 1e-13)
        err_tv = np.where(abs(gpu_tv - t_v) > 1e-13)
        err_z = np.where(abs(gpu_z - z) > 1e-13)
        err_u = np.where(abs(gpu_u - u) > 1e-13)
        err_v = np.where(abs(gpu_v - v) > 1e-13)
        err_htdv = np.where(abs(gpu_htdv - Htdv ) > 1e-13)
        err_htdu = np.where(abs(gpu_htdu - Htdu ) > 1e-13)
        err_daui = np.where(abs(gpu_daui - daui ) != 0)
        err_dauj = np.where(abs(gpu_dauj - dauj ) != 0)
        err_cuoii = np.where(abs(gpu_cuoii - cuoii ) != 0)
        err_cuoij = np.where(abs(gpu_cuoij - cuoij ) != 0)
        err_ku = np.where(abs(gpu_khouot - khouot) != 0)

        print "tz", err_tz
        print "tu", err_tu
        print "tv", err_tv
        print "z:",  err_z
        print "u:", err_u
        print "v:", err_v
        print 'khouot:', err_ku
        print "Htd :",  err_htdu        
        print "Htd :",  err_htdv
        print "dau :",  err_daui
        print "dau :",  err_dauj 
        print "cuoi:", err_cuoii
        print "cuoi:", err_cuoij

        # need to update all changing variables here
        #pointers.update(["t_z", "vbd", "ubt", "t_u", "t_v"])
        

        t = t + dT * 0.5      
        #---------------------------------CPU code---------------------------------------------
        Boundary_at(t, canal)
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
        for j in range(2, M+1):
            if mocj[j] > 0:
                for k in range(mocj[j]):
                    dau, cuoi, solidBound = set_boundary(is_i=False, i_j=j, k=k)
                    uSolver(l=dau, r=cuoi, jpos=j, solidBound=solidBound)
        Reset_state(False)
        update_uvz()
        
        # attention
        Find_Calculation_limits()
        Htuongdoi()

        #------------------------------GPU code------------------------------------------------- 

        _gpu_boundary_at(t, ctx, supmod, pc, canal)
        ctx.synchronize()

        gpu_vzSolver(*vzSolver_args, block=block_v, grid=grid_v)
        ctx.synchronize()
       
        gpu_uSolver(*uSolver_arg, block=block_u, grid=grid_v)
       
        ctx.synchronize();
       
        
        gpu_reset_state_y(*reset_arg, block=block_2d, grid=grid_2d)
        ctx.synchronize();
        
        gpu_update_uvz(*update_arg, block=block_2d, grid=grid_2d)
        ctx.synchronize()
       


        gpu_Mark_x(*markx_arg, block=block_u, grid=grid_u);
        gpu_Mark_y(*marky_arg, block=block_u, grid=grid_v);
        ctx.synchronize();

        gpu_Htuongdoi(*Htuongdoi_arg, block=block_2d, grid=grid_2d)
        ctx.synchronize();
        pointers.extract({"u": gpu_u, "v" : gpu_v, "z" : gpu_z}) 
        if plot:
            #time.sleep(2)
            print (t)
            # xuat z de kiem tra
            print ('z')
            fig1 = plt.figure()
            plt.plot(gpu_z[4, 2 : M + 1])
            #plt.plot(gpu_z[2 : N + 1, 4])
            plt.xlim(0, 400)
            plt.ylim(-0.02, 0.02)
            #plt.show()
            filename = 'pic/z' + str(t) + '.png'
            plt.savefig(filename)

        pointers.extract({"t_z" : gpu_tz, "t_u" : gpu_tu, "u": gpu_u, "v" : gpu_v, "z" : gpu_z, "Htdu" : gpu_htdu, "Htdv" : gpu_htdv,\
         "daui" : gpu_daui, "cuoii" : gpu_cuoii, "dauj" : gpu_dauj, "cuoij" : gpu_cuoij, "khouot" : gpu_khouot })
        err_z = np.where(abs(gpu_z - z) > 1e-13)
        err_u = np.where(abs(gpu_u - u) > 1e-13)
        err_v = np.where(abs(gpu_v - v) > 1e-13)
        err_htdv = np.where(abs(gpu_htdv - Htdv ) > 1e-13)
        err_htdu = np.where(abs(gpu_htdu - Htdu ) > 1e-13)
        err_daui = np.where(abs(gpu_daui - daui ) != 0)
        err_dauj = np.where(abs(gpu_dauj - dauj ) != 0)
        err_cuoii = np.where(abs(gpu_cuoii - cuoii ) != 0)
        err_cuoij = np.where(abs(gpu_cuoij - cuoij ) != 0)
        err_ku = np.where(abs(gpu_khouot - khouot) != 0)
        print "z:",  err_z
        print "u:", err_u
        print "v:", err_v
        print 'khouot:', err_ku
        print "Htd :",  err_htdu        
        print "Htd :",  err_htdv
        print "dau :",  err_daui
        print "dau :",  err_dauj 
        print "cuoi:", err_cuoii
        print "cuoi:", err_cuoij
