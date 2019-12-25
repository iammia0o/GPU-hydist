'''
ULSAN NATIONAL INSTIUTE OF SCIENCE AND TECHNOLOGY
Copyright (c) 2019 HVCL lab
Created by Huong Nguyen

'''


from __future__ import division
import numpy as np
import math as mth
import os
import matplotlib as plt
from Coeff import *
from Global_Variables import *
from Visualizor import *
import timeit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.compiler import DynamicSourceModule
import matplotlib.pyplot as plt
import time 
from Pointers import PointersStruct
import copy
tol = 1e-5
floattype=np.float64


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


def hydraulic_Calculation(Tmax, pointers, arg_struct_ptr, arr_struct_ptr, supmod, ctx, ketdinh=True, blk_size=1024, kenhhepd=0, kenhhepng=0,
    debug=True, interval= 1, sediment_start=36000, t_start=0, sec=1, plot=False, export=False, save_grid=True, bed_change_start=72000):

#----------------------------------------------set ups blocks and grids-----------------------------------------------------
    u_list = []
    v_list = []
    z_list = []
    fslist = []

    # need to fix this
    M1 = M + 3; N1 = N + 3
    ketdinh = np.int8(ketdinh)
    bed_change_start = floattype(bed_change_start)

    block_2d = (min(blk_size, M1), 1, 1)
    grid_2d = (int (ceil(M1 / min(blk_size, M1))), N1, 1)
    print "Simulation time = ", int(Tmax) // 3600, "hours"
    print "N, M : ",  N," ", M
    t = t_start
    channel = kenhhepd or kenhhepng


#----------------------------------------------set ups kernels arguments-----------------------------------------------------

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
    

    reset_arg  = [np.int32(M),np.int32(N), pc['H_moi'], pc['htaiz'], pc['khouot'], pc['z'], pc['t_z'], pc['t_u'], pc['t_v']]


    update_arg = [np.int32(M), np.int32(N), pc['u'], pc['v'], pc['z'], pc['t_u'], pc['t_v'], pc['t_z'], np.int32(kenhhepd), np.int32(kenhhepng)]




#----------------------------------------------set up for UZ kernels--------------------------------------------------------
    UVZkernels = SourceModule(open("UVZSolver_multithread.cu").read(), include_dirs = [os.getcwd()], options=['-maxrregcount=32'])

    # return

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
    # tridiagSolver_v2 = UVZkernels.get_function('tridiagSolver_v2')

    Sediment_Transport = SourceModule(open('Sediment_Transport.cu').read(), include_dirs=[os.getcwd()], options=['-maxrregcount=32'])
    Scan_FSi = Sediment_Transport.get_function('Scan_FSi')
    Scan_FSj = Sediment_Transport.get_function('Scan_FSj')
    FSj_extract_solution = Sediment_Transport.get_function('FSj_extract_solution')
    FSi_extract_solution = Sediment_Transport.get_function('FSi_extract_solution')
    Find_VTH = Sediment_Transport.get_function('Find_VTH')
    hesoK = Sediment_Transport.get_function('hesoK')
    BedLoad = Sediment_Transport.get_function('BedLoad')
    Update_FS = Sediment_Transport.get_function('Update_FS')
    Calculate_Qb = Sediment_Transport.get_function('Calculate_Qb')

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
        _gpu_boundary_at(t, ctx, supmod, pc, arg_struct_ptr, channel)
        ctx.synchronize()
        # if t == sediment_start: print timeit.default_timer()


        start_idx = np.int32(2)
        end_idx = np.int32(M)
        isU = np.int8(True)
        jump_step = np.int32(2)
        if channel and (kenhhepd == 1):
            start_idx = 3
            end_idx = M - 1

        block_size = (1, 1024, 1)
        grid_size = (M1, int(ceil(N/1024)), 1)
        # print block_size, grid_size
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

        # tridiagSolver_v2(np.int8(False), isU, start_idx, end_idx, jump_step, np.int32(2 * N + 1), arg_struct_ptr, arr_struct_ptr, block=(256, 1, 1), grid=(1, 1, 1))
        # ctx.synchronize()


        UZSolver_extract_solution(start_idx, end_idx, arg_struct_ptr, arr_struct_ptr, block=block_size, grid=grid_size)
        ctx.synchronize()
        # return

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
        

        gpu_update_h_moi(arg_struct_ptr,block=block_2d, grid=grid_2d)
        ctx.synchronize()
        gpu_reset_state_x(arg_struct_ptr, block=(32, 1, 1), grid=(1, N, 1))
        # print "stucking at 200"
        ctx.synchronize();


        gpu_update_uvz(*update_arg, block=block_2d, grid=grid_2d)
        ctx.synchronize()
   

        gpu_Mark_x(arg_struct_ptr, block=(32, 1, 1), grid=(1, N, 1))
        gpu_Mark_y(arg_struct_ptr, block=(32, 1, 1), grid=(1, M, 1))
        # ctx.synchronize();
        # print "stucking at 221"
        gpu_Htuongdoi(arg_struct_ptr, block=block_2d, grid=grid_2d)
        ctx.synchronize()
        
        
        if t >= sediment_start:
            Find_VTH(arg_struct_ptr, block=block_2d, grid=grid_2d)
            hesoK(arg_struct_ptr, block=block_2d, grid=grid_2d)
            ctx.synchronize()

         
            start_idx = np.int32(3)
            end_idx = np.int32(M - 1)
            Scan_FSj(floattype(t), bed_change_start,ketdinh, start_idx, end_idx, arg_struct_ptr, arr_struct_ptr, block=block_size, grid=grid_size)
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

            pointers.extract({"u": gpu_u, "v" : gpu_v, "z" : gpu_z, "t_z" : gpu_tz, "t_u" : gpu_tu, "t_v": gpu_tv, "Htdu": gpu_htdu})
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


        _gpu_boundary_at(t, ctx, supmod, pc, arg_struct_ptr, channel)
        ctx.synchronize()

        

        block_size = (1024, 1, 1) 
        grid_size = (int (ceil(M / 1024)), N, 1)
        # block_size = (M, 1, 1) 
        # grid_size = (1, N, 1)
        start_idx = np.int32(2)
        end_idx = np.int32(N)
        jump_step = np.int32(2)
        isU = np.int8(False)
        if channel and (kenhhepng == 1):
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

        # tridiagSolver_v2(np.int8(False), isU, start_idx, end_idx, jump_step, np.int32(2 * M + 1), arg_struct_ptr, arr_struct_ptr, block=(min(1024, N), 1, 1), grid=(ceil(M/ min(1024, N)), 1, 1))
        # ctx.synchronize()

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

        gpu_Htuongdoi(arg_struct_ptr, block=block_2d, grid=grid_2d)
        # print "stucking at 325"
        
        ctx.synchronize()


        if t >= sediment_start:
            Find_VTH(arg_struct_ptr, block=block_2d, grid=grid_2d)
            hesoK(arg_struct_ptr, block=block_2d, grid=grid_2d)
            ctx.synchronize()

            # sediment kernels come here
            # Scan FSj
            start_idx = np.int32(3)
            end_idx = np.int32(N - 1)
            Scan_FSi(floattype(t), bed_change_start, ketdinh, start_idx, end_idx, arg_struct_ptr, arr_struct_ptr, block=block_size, grid=grid_size)
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

        # only here is new. Verify if this work 
        if int(t) % bed_change_start == 0 and t - int(t) == 0:
            Calculate_Qb(ketdinh, arg_struct_ptr, arr_struct_ptr, block=block_2d, grid=grid_2d)
            BedLoad(floattype(t), ketdinh, start_idx, end_idx, arg_struct_ptr, arr_struct_ptr, block=block_2d, grid=grid_2d)


        if debug:
            print t

            pointers.extract({'tFS' : FS, "u": gpu_u, "v" : gpu_v, "z" : gpu_z, "t_z" : gpu_tz, "t_u" : gpu_tu, "t_v": gpu_tv, "Htdu": gpu_htdu})
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

    #-------------------------------------


        if int(t) % interval == 0 and t - int(t) == 0:
            pointers.extract({"u": gpu_u, "v" : gpu_v, "z" : gpu_z, 'FS' : FS}) 
            ctx.synchronize()
            print "hour: ",  int(t) // 3600
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
                return u_list, v_list, z_list, fslist


            if plot:
                visualize(gpu_u[1:N+1, 1:M+1], gpu_v[1: N+1, 1: M+1])
            # u v z are saved once every interval
            v_list = v_list + [copy.deepcopy (gpu_v)]
            u_list = u_list + [copy.deepcopy (gpu_u)]
            z_list = z_list + [copy.deepcopy (gpu_z)]

        # FS is saved once every 10 mins
        if save_grid and t >= sediment_start and int(t) % 360 == 0 and t - int(t) == 0:
            file_name = 'Outputs/FS_GPU/fs_' + str(t) + '.grd'
            saveFS(FS, file_name)
            pointers.extract({'FS' : FS})
            fslist = fslist + [copy.deepcopy(FS)]
            plt.figure(figsize=(10,10))
            plt.imshow(FS *2000)
            plt.show()


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

    #kernel_function to draw heat map
    # heat has argument for u,z 
       
    return u_list, v_list, z_list, fslist

