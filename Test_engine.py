from __future__ import division
import numpy as np
import math as mth
import os
import matplotlib as plt
from Coeff import *
from Global_Variables import *
from Supplementary_Functions import *
from Load_Boundary_Conditions import *
from Reynolds_Equation_Solver import *
import time
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
import time 
from Pointers import PointersStruct
from matplotlib.colors import Normalize
import copy
tol = 1e-5
floattype=np.float64


def visualize(U, V, savefile=False, stt=0):
    # im = plt.imshow(U)
    # plt.show()
    # im = plt.imshow(V)
    # plt.show()
    n, m = U.shape
    fig = plt.figure(num=None, figsize=(10, 10), facecolor='w', edgecolor='k')
    X, Y = np.mgrid[0:n, 0:m]
    EE  = np.sqrt(U * U + V * V)
    cax = plt.axis('equal')
    plt.quiver(X, Y, U, V, alpha=.5)
    normalize = Normalize ()
    cmap=normalize (EE.flatten ())
    normalize.autoscale (EE.flatten ())
    im = plt.quiver(X[::4,::4], Y[::4,::4], U[::4,::4], V[::4,::4], EE[::4,::4], scale=20, headwidth=4, pivot='tail', angles='uv')
    if savefile == True:
        fig.colorbar(im)
        plt.savefig('Outputs/TanChau/' + 'vector_field_' + str(stt) + '.png')
    plt.show()


def _gpu_boundary_at(t, ctx, source, pc, arg_struct_ptr, channel):
    if channel == 1:
        offset = M  + 3;
        m_offset = 5
        up = source.get_function("boundary_up")
        down = source.get_function("boundary_down")
        left = source.get_function("boundary_left")
        right = source.get_function("boundary_right")
        b_args = [floattype(t), np.int32(m_offset), np.int32(M), np.int32(N), pc['bienQ'], pc['t_z'],\
             pc['daui'], pc['cuoii'], pc['dauj'], pc['cuoij'], pc['vbt'], pc['vbd'], pc['ubt'], pc['ubp']]
        up(*b_args, block=(min(1024, N), 1, 1), grid=(N // min(1024, N), 1, 1))
        down(*b_args, block=(min(1024, N), 1, 1), grid=(N // min(1024, N), 1, 1))
        left(*b_args, block=(min(1024, M), 1, 1), grid=(M // min(1024, M), 1, 1))
        right(*b_args, block=(min(1024, M), 1, 1), grid=(M // min(1024, M), 1, 1))
        ctx.synchronize()
    else:
        update_boundary_value = source.get_function("Update_Boundary_Value")
        update_boundary_value(floattype(t), np.int32(total_time), arg_struct_ptr, block= (1, 1024, 1), grid= (1, max(M, N) // 1024 + 1, 1))

        # gpu_ubt = ubt.astype(floattype)
        # gpu_tv = v.astype(floattype)
        # cuda.memcpy_dtoh(gpu_ubt, pc['ubt'] )
        # cuda.memcpy_dtoh(gpu_tv, pc['t_v'])
        # print gpu_ubt
        # print gpu_tv[:, 2]

def hydraulic_Calculation(Tmax, pointers, arg_struct_ptr, arr_struct_ptr, supmod, ctx, ketdinh=True, blk_size=512, 
    debug=True, interval= 1, itv1=0, itv2=700, canal=0, t_start=0, sec=1, plot=False, export=False):

#----------------------------------------------set ups blocks and grids-----------------------------------------------------
    u_list = []
    v_list = []
    z_list = []

    M1 = M + 3; N1 = N + 3
    ketdinh = np.int8(ketdinh)

    block_u = (min(M1, blk_size), 1, 1)
    grid_u = (M1 // min(blk_size, M1) + 1, 1, 1)
    block_v = (min(N1, blk_size), 1, 1)
    grid_v = (N1 // min(blk_size, N1), 1, 1)

    block_2d = (min(blk_size, M1), 1, 1)
    grid_2d = (M1 // min(blk_size, M1), N1, 1)


#----------------------------------------------set ups kernels arguments-----------------------------------------------------
    print ("Tmax = ", Tmax)
    with open('DirsOfOutputs.txt', 'r') as inf:
        Dirs = dict([line.split() for line in inf])
    fz = open(Dirs['z_V'], 'w')
    fu = open(Dirs['u_V'], 'w')
    fv = open(Dirs['v_V'], 'w')
    print ("N, M : ",  N," ", M)
    t = t_start
    count = 1;


    gpu_Mark_x = supmod.get_function("Find_Calculation_limits_Horizontal")
    gpu_Mark_y = supmod.get_function("Find_Calculation_limits_Vertical")
    gpu_Htuongdoi = supmod.get_function("Htuongdoi")
    gpu_update_uvz = supmod.get_function("update_uvz")
    gpu_update_h_moi = supmod.get_function("update_h_moi")
    gpu_reset_state_x = supmod.get_function("Reset_states_horizontal")   
    gpu_reset_state_y = supmod.get_function("Reset_states_vertical")
    normalize = supmod.get_function("Normalize")
    update_buffer = supmod.get_function("update_buffer")

    pc = pointers.arg_list
    pd = pointers.device_only_ptrs
    start_idx = 2
    end_idx = M + 1
    

    reset_arg  = [np.int32(M),np.int32(N), pc['H_moi'], pc['htaiz'], pc['khouot'], pc['z'], pc['t_z'], pc['t_u'], pc['t_v']]


    update_arg = [np.int32(M), np.int32(N), pc['u'], pc['v'], pc['z'], pc['t_u'], pc['t_v'], pc['t_z'], np.int32(kenhhepd), np.int32(kenhhepng)]
    # update_arg_1 = [np.int32(M), np.int32(N), pc['u'], pc['v'], pc['z'], pc['t_u'], pc['t_v'], pc['t_z'], pd['AA'], pc['t_v'], np.int32(kenhhepd), np.int32(kenhhepng)]
    # update_arg_2 = [np.int32(M), np.int32(N), pc['u'], pc['v'], pc['z'], pc['t_u'], pc['t_v'], pc['t_z'], pc['t_u'], pd['BB'], np.int32(kenhhepd), np.int32(kenhhepng)]



#----------------------------------------------set up for UZ kernels--------------------------------------------------------
    UVZkernels = SourceModule(open("UVZSolver_multithread.cu").read(), include_dirs = [os.getcwd()], options=['-maxrregcount=32'])

    gpu_vSolver = UVZkernels.get_function("solveV")
    gpu_uSolver = UVZkernels.get_function("solveU")

    VZSolver_calculate_preindex =  UVZkernels.get_function('VZSolver_calculate_preindex')
    VZSolver_calculate_abcd = UVZkernels.get_function('VZSolver_calculate_abcd')
    VZSolver_calculate_matrix_coeff = UVZkernels.get_function('VZSolver_calculate_matrix_coeff')
    VZSolver_extract_solution = UVZkernels.get_function('VZSolver_extract_solution')

    UZSolver_calculate_preindex =  UVZkernels.get_function('UZSolver_calculate_preindex')
    UZSolver_calculate_abcd = UVZkernels.get_function('UZSolver_calculate_abcd')
    UZSolver_calculate_matrix_coeff = UVZkernels.get_function('UZSolver_calculate_matrix_coeff')
    UZSolver_extract_solution = UVZkernels.get_function('UZSolver_extract_solution')
    update_margin_elem_U = UVZkernels.get_function("update_margin_elem_U")
    update_margin_elem_V = UVZkernels.get_function("update_margin_elem_V")

    
    tridiagSolver = UVZkernels.get_function('tridiagSolver')

    Sediment_Transport = SourceModule(open('Sediment_Transport.cu').read(), include_dirs=[os.getcwd()], options=['-maxrregcount=32'])
    Scan_FSi = Sediment_Transport.get_function('Scan_FSi')
    Scan_FSj = Sediment_Transport.get_function('Scan_FSj')
    FSj_extract_solution = Sediment_Transport.get_function('FSj_extract_solution')
    FSi_extract_solution = Sediment_Transport.get_function('FSi_extract_solution')
    Find_VTH = Sediment_Transport.get_function('Find_VTH')
    hesoK = Sediment_Transport.get_function('hesoK')
    BedLoad = Sediment_Transport.get_function('BedLoad')
    Update_FS = Sediment_Transport.get_function('Update_FS')

    # for addr in global_attributes:
    #     print addr
    gpu_tz = np.zeros(t_z.shape,dtype=floattype)
    gpu_tu = np.zeros(t_u.shape,dtype=floattype)
    gpu_tv = np.zeros(t_v.shape,dtype=floattype)
    gpu_z = np.zeros(z.shape,dtype=floattype)
    gpu_u = np.zeros(u.shape,dtype=floattype)
    gpu_v = np.zeros(v.shape,dtype=floattype)
    gpu_htdu = np.zeros(v.shape,dtype=floattype)
    gpu_ubt = np.zeros(ubt.shape, dtype=floattype)
    


#--------------------------------------------------- Main Calculation--------------------------------------------------------
    while t < Tmax:
        
        t = t + dT * 0.5      
        _gpu_boundary_at(t, ctx, supmod, pc, arg_struct_ptr, canal)
        ctx.synchronize()
        # if (t == 3.0):
        #     pointers.extract({"ubt" : gpu_ubt})
        #     print gpu_ubt[3:51], "here 261"
        # print t

        start_idx = np.int32(2)
        end_idx = np.int32(M)
        isU = np.int8(True)
        jump_step = np.int32(2)
        if canal and (kenhhepd == 1):
            start_idx = 3
            end_idx = M - 1

        block_size = (1, N1, 1)
        grid_size = (M1, 1, 1)
        UZSolver_calculate_preindex(start_idx, end_idx, arg_struct_ptr, arr_struct_ptr, block=block_size, grid=grid_size)
        ctx.synchronize()
        UZSolver_calculate_abcd(start_idx, end_idx, arg_struct_ptr, arr_struct_ptr, block=block_size, grid=grid_size)
        ctx.synchronize()
        UZSolver_calculate_matrix_coeff(start_idx, end_idx, arg_struct_ptr, arr_struct_ptr, block=block_size, grid=grid_size)
        ctx.synchronize()
        tridiagSolver(np.int8(False), isU, start_idx, end_idx, jump_step,
                        np.int32(2 * N + 1), arg_struct_ptr, arr_struct_ptr, 
                        block=(32, 1, 1), grid=(1, M - 1 , 1))
        ctx.synchronize()
        UZSolver_extract_solution(start_idx, end_idx, arg_struct_ptr, arr_struct_ptr, block=block_size, grid=grid_size)
        ctx.synchronize()


        normalize(np.int8(1), arg_struct_ptr, arr_struct_ptr, block=block_2d, grid=grid_2d)
        ctx.synchronize()
        update_buffer(np.int8(1), arg_struct_ptr, arr_struct_ptr, block=block_2d, grid=grid_2d)
        ctx.synchronize()


        gpu_vSolver(floattype(t), np.int32(2), np.int32(N), arg_struct_ptr, block=block_2d, grid=grid_2d)
        ctx.synchronize();
        update_margin_elem_V(floattype(t), np.int32(2), np.int32(N), arg_struct_ptr, block=(32, 1, 1), grid=(1, N, 1))


        normalize(np.int8(0), arg_struct_ptr, arr_struct_ptr, block=block_2d, grid=grid_2d)
        ctx.synchronize()
        update_buffer(np.int8(0), arg_struct_ptr, arr_struct_ptr, block=block_2d, grid=grid_2d)
        ctx.synchronize()
        
        # pointers.extract({"t_u" : gpu_tu})
        # print gpu_tu[1, 3], t, "after normalize"

        gpu_update_h_moi(arg_struct_ptr,block=block_2d, grid=grid_2d)
        ctx.synchronize()
        # gpu_reset_state_x(*reset_arg, block=block_2d, grid=grid_2d)
        #int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

        # gpu_reset_state_x(*reset_arg, block=(32, 1, 1), grid=(1, N, 1))
        gpu_reset_state_x(arg_struct_ptr, block=(32, 1, 1), grid=(1, N, 1))
        # print "stucking at 200"
        ctx.synchronize();


        gpu_update_uvz(*update_arg, block=block_2d, grid=grid_2d)
        ctx.synchronize()
        # return
        block_u = (min(M1, blk_size), 1, 1)
        grid_u = (M1 // min(blk_size, M1) + 1, 1, 1)
        block_v = (min(N1, blk_size), 1, 1)
        grid_v = (N1 // min(blk_size, N1), 1, 1)

        gpu_Mark_x(arg_struct_ptr, block=(32, 1, 1), grid=(1, N, 1))
        gpu_Mark_y(arg_struct_ptr, block=(32, 1, 1), grid=(1, M, 1))
        # ctx.synchronize();
        # print "stucking at 221"
        gpu_Htuongdoi(arg_struct_ptr, block=(M, 1, 1), grid=(1, N, 1))
        ctx.synchronize()
        

        Find_VTH(arg_struct_ptr, block=block_2d, grid=grid_2d)
        hesoK(arg_struct_ptr, block=block_2d, grid=grid_2d)
        ctx.synchronize()

        # pointers.extract({"tFS" : FS})
        # plt.figure(figsize=(10, 10))
        # plt.imshow(FS)
        # plt.show()
        # Sediment Kernels come here
        # Scan FSi
        start_idx = np.int32(3)
        end_idx = np.int32(M - 1)
        Scan_FSj(ketdinh, start_idx, end_idx, arg_struct_ptr, arr_struct_ptr, block=block_size, grid=grid_size)
        ctx.synchronize();

        # Tridiag
        jump_step = 1
        tridiagSolver(np.int8(True), isU, start_idx, 
                        end_idx, np.int32(jump_step), np.int32(N + 3), 
                        arg_struct_ptr, arr_struct_ptr, block=(32, 1, 1), grid=(1, M - 1 , 1))
        ctx.synchronize();

        # Extract Solution
        FSj_extract_solution(ketdinh, start_idx, end_idx, arg_struct_ptr, arr_struct_ptr, block=block_size, grid=grid_size)
        ctx.synchronize();
        Update_FS(arg_struct_ptr, block=block_2d, grid=grid_2d)
    #----------------

        if debug:
            print t

            # pointers.extract({"t_z" : gpu_tz, "t_u" : gpu_tu, "t_v": gpu_tv, "u": gpu_u, "v" : gpu_v, "z" : gpu_z, "Htdu" : gpu_htdu})#, "Htdv" : gpu_htdv,\
            # "daui" : gpu_daui, "cuoii" : gpu_cuoii, "dauj" : gpu_dauj, "cuoij" : gpu_cuoij, "khouot" : gpu_khouot })
            pointers.extract({'FS' : FS, "u": gpu_u, "v" : gpu_v, "z" : gpu_z, "t_z" : gpu_tz, "t_u" : gpu_tu, "t_v": gpu_tv, "Htdu": gpu_htdu})
            plt.figure(figsize=(10, 10))
            plt.imshow(FS[1:N+1, 1:M+1])
            plt.show()
            if int(t) % interval == 0:
                visualize(gpu_u[1 : N + 1, 1 : M + 1], gpu_v[1 : N + 1, 1 : M + 1])
            # if t >=itv1 and t < itv2:
            v_list = v_list + [copy.deepcopy (gpu_tv)]
            u_list = u_list + [copy.deepcopy (gpu_tu)]
            z_list = z_list + [copy.deepcopy (gpu_tz)]

        # return
        t = t + dT * 0.5      
        # print t


        _gpu_boundary_at(t, ctx, supmod, pc, arg_struct_ptr, canal)
        ctx.synchronize()


        block_size = (M, 1, 1) 
        grid_size = (1, N, 1)

        start_idx = np.int32(2)
        end_idx = np.int32(N)
        jump_step = np.int32(2)
        isU = np.int8(False)
        if canal and (kenhhepng == 1):
            start_idx = 3
            end_idx = N

        VZSolver_calculate_preindex(floattype(t), start_idx, end_idx, arg_struct_ptr, arr_struct_ptr, block=block_size, grid=grid_size)
        ctx.synchronize()
            
        VZSolver_calculate_abcd(start_idx, end_idx, arg_struct_ptr, arr_struct_ptr, block=block_size, grid=grid_size)
        ctx.synchronize()
        VZSolver_calculate_matrix_coeff(start_idx, end_idx, arg_struct_ptr, arr_struct_ptr, block=block_size, grid=grid_size)
        ctx.synchronize()


        tridiagSolver(np.int8(False), isU, start_idx, end_idx, jump_step, np.int32(2 * M + 1), arg_struct_ptr, arr_struct_ptr, block=(32, 1, 1), grid=(1, N - 1, 1))
        ctx.synchronize()

        VZSolver_extract_solution(start_idx, end_idx, arg_struct_ptr, arr_struct_ptr, block=block_size, grid=grid_size)
        ctx.synchronize()      


        normalize(np.int8(0), arg_struct_ptr, arr_struct_ptr, block=block_2d, grid=grid_2d)
        ctx.synchronize()
        update_buffer(np.int8(0), arg_struct_ptr, arr_struct_ptr, block=block_2d, grid=grid_2d)
        ctx.synchronize()

       

        gpu_uSolver(floattype(t), np.int32(2), np.int32(M), arg_struct_ptr, block=block_2d, grid=grid_2d)
        ctx.synchronize()
        update_margin_elem_U(np.int32(2), np.int32(M), arg_struct_ptr, block=(32, 1, 1), grid=(1, M, 1))
        ctx.synchronize()


        normalize(np.int8(1), arg_struct_ptr, arr_struct_ptr, block=block_2d, grid=grid_2d)
        ctx.synchronize()
        update_buffer(np.int8(1), arg_struct_ptr, arr_struct_ptr, block=block_2d, grid=grid_2d)
        ctx.synchronize()


        gpu_update_h_moi(arg_struct_ptr,block=block_2d, grid=grid_2d)
        ctx.synchronize()
        gpu_reset_state_y(arg_struct_ptr, block=(1, 32, 1), grid=(M, 1, 1))
        ctx.synchronize()
        
        gpu_update_uvz(*update_arg, block=block_2d, grid=grid_2d)
        # gpu_update_uvz(*update_arg, block=block_2d, grid=grid_2d)
        ctx.synchronize()



        gpu_Mark_x(arg_struct_ptr, block=(32, 1, 1), grid=(1, N, 1))
        gpu_Mark_y(arg_struct_ptr, block=(32, 1, 1), grid=(1, M, 1))
        ctx.synchronize();

        gpu_Htuongdoi(arg_struct_ptr, block=(M, 1, 1), grid=(1, N, 1))
        # print "stucking at 325"
        
        ctx.synchronize()

        Find_VTH(arg_struct_ptr, block=block_2d, grid=grid_2d)
        hesoK(arg_struct_ptr, block=block_2d, grid=grid_2d)
        ctx.synchronize()

        # sediment kernels come here
        # Scan FSj
        start_idx = np.int32(3)
        end_idx = np.int32(N - 1)
        Scan_FSi(ketdinh, start_idx, end_idx, arg_struct_ptr, arr_struct_ptr, block=block_size, grid=grid_size)
        ctx.synchronize();

        jump_step = 1
        # Tridiag
        tridiagSolver(np.int8(True), isU, start_idx,
                        end_idx, np.int32(jump_step), np.int32(M + 3), 
                        arg_struct_ptr, arr_struct_ptr, block=(32, 1, 1), grid=(1, N - 3 , 1))
        ctx.synchronize();

        # Extract Solution
        FSi_extract_solution(ketdinh, start_idx, end_idx, arg_struct_ptr, arr_struct_ptr, block=block_size, grid=grid_size)
        ctx.synchronize();
        Update_FS(arg_struct_ptr, block=block_2d, grid=grid_2d)
        ctx.synchronize();

        if debug:
            print t

            # pointers.extract({"t_z" : gpu_tz, "t_u" : gpu_tu, "t_v": gpu_tv, "u": gpu_u, "v" : gpu_v, "z" : gpu_z, "Htdu" : gpu_htdu})#, "Htdv" : gpu_htdv,\
            # "daui" : gpu_daui, "cuoii" : gpu_cuoii, "dauj" : gpu_dauj, "cuoij" : gpu_cuoij, "khouot" : gpu_khouot })
            pointers.extract({'tFS' : FS, "u": gpu_u, "v" : gpu_v, "z" : gpu_z, "t_z" : gpu_tz, "t_u" : gpu_tu, "t_v": gpu_tv, "Htdu": gpu_htdu})
            # pointers.extract({"u": gpu_u, "v" : gpu_v, "z" : gpu_z, "t_z" : gpu_tz, "t_u" : gpu_tu, "t_v": gpu_tv, "Htdu": gpu_htdu})
            plt.figure(figsize=(10, 10))
            plt.imshow(FS[1:N+1, 1:M+1])
            plt.show()
            if int(t) % interval == 0:
                visualize(gpu_u[1 : N + 1, 1 : M + 1], gpu_v[1 : N + 1, 1 : M + 1])

            # if t >=itv1 and t < itv2:
            v_list = v_list + [copy.deepcopy (gpu_tv)]
            u_list = u_list + [copy.deepcopy (gpu_tu)]
            z_list = z_list + [copy.deepcopy (gpu_tz)]
            if np.isnan(gpu_u).any() or np.isnan(gpu_v).any() or np.isnan(gpu_z).any():
                plt.figure(figsize=(10, 10))
                plt.imshow(gpu_u)
                plt.show()
                plt.figure(figsize=(10, 10))
                plt.imshow(gpu_v)
                plt.show()
                plt.figure(figsize=(10, 10))
                plt.imshow(gpu_z)
                plt.show()
                exit()

        # return u_list, v_list,  z_list
    #-------------------------------------


        if int(t) % interval == 0 and t - int(t) == 0:
            # print int(t) // 3600
            pointers.extract({"u": gpu_u, "v" : gpu_v, "z" : gpu_z, 'FS' : FS}) 
            ctx.synchronize()
            print t
            if np.isnan(gpu_u).any() or np.isnan(gpu_v).any() or np.isnan(gpu_z).any():
                plt.figure(figsize=(10, 10))
                plt.imshow(gpu_u)
                plt.show()
                plt.figure(figsize=(10, 10))
                plt.imshow(gpu_v)
                plt.show()
                plt.figure(figsize=(10, 10))
                plt.imshow(gpu_z)
                plt.show()
                return u_list, v_list, z_list


            
            
            if plot:
                print (t)
                visualize(gpu_u[1:N+1, 1:M+1], gpu_v[1: N+1, 1: M+1])
            v_list = v_list + [copy.deepcopy (gpu_v)]
            u_list = u_list + [copy.deepcopy (gpu_u)]
            z_list = z_list + [copy.deepcopy (gpu_z)]

            
        
        if export and int(t) % 3600 == 0 and t - int(t) == 0:
            # name = 'Outputs/Song_Luy/log/' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) + '.log'
            filename = 'Outputs/Song_Luy/velocity/VTH' + str(t) + 's.data'
            v_file = open(filename, 'w')
            filename = 'Outputs/Song_Luy/water_level/mucnuoc' + str(t) + 's.grid'
            z_file = open(filename, 'w')
            for i in range(1, N + 1):
                for j in range(1, M + 1):
                    # tinh vth
                    vt = (v[i, j - 1] + v[i, j]) * 0.5
                    ut = (u[i, j] + u[i - 1, j]) * 0.5
                    vth = np.sqrt(vt * vt + ut * ut)
                    if (htaiz[i, j] <= NANGDAY):
                        vth = 0
                    # tinh goc
                    angle = 0
                    if ut != 0 :
                        angle = 180 *  np.arctan(abs(vt / ut)) * 1 / np.pi
                        if vt < 0 and ut > 0:
                            angle = 360 - angle
                        elif vt < 0 and ut < 0:
                            angl += 180
                        elif vt > 0 and ut <0:
                            angle = 180 - angle
                    else:
                        if vt > 0:
                            angle  = 90.0
                        elif vt < 0: angle = 270.0
                    #in kq
                    v_file.write('%.1f %d %d %.2f %.2f' % (t, i, j, vth, angle))
                    z_file.write('%.2f ' % (z[i , j]))
                z_file.write('\n')


       
    return u_list, v_list, z_list
