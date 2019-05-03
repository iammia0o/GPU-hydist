#!/home/huongnm/anaconda2/bin/python
from __future__ import division

import timeit

import numpy as np 
from Coeff import *
from Global_Variables import *
from Supplementary_Functions import *
from Load_Boundary_Conditions import *
from Test_engine import *
from Hydraulic_Calculation import *
import pycuda.driver as cuda
#import pycuda.autoinit
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
from Pointers import Pointers
import argparse

def load_input_and_initialize():
    TinhKhoUot()
    Htuongdoi()
    K_factor()
    GiatriHtaiZ()
    #Initial_condition(Dirs['U_file'], Dirs['V_file'], Dirs['Z_file']) # this can load new 
    # Load initial condition (tinh tiep hoac tinh tu dau)
    Find_Calculation_limits()

def gpu_init(device_no):
    dev = cuda.Device(device_no) # the number of GPU
    ctx = dev.make_context()
    
    print M, N
    kwargs = {"h": h, "hsnham": hsnham, "VISCOINDX" : VISCOINDX, "H_moi": H_moi, "bienQ" : bienQ,\
            "moci" : moci ,"mocj" : mocj, "dauj": dauj, "daui" : daui, "cuoii" : cuoii, "cuoij" : cuoij,\
            "Tsxw" : Tsxw, "Tsyw" : Tsyw, "khouot" : khouot, "boundary_type" : boundary_type,\
            "u": u, "v": v, "z" : z, "t_u": t_u, "t_v": t_v, "t_z": t_z, "Htdu": Htdu, "Htdv" : Htdv,\
            "Kx1" : Kx1, "Ky1" : Ky1, "htaiz" : htaiz, "ut" : ut, "vt": vt, \
            "bc_up": bc_up, "bc_down": bc_down, "bc_left": bc_left, "bc_right" : bc_right,\
           "ubt" : ubt, "ubp" : ubp, "vbt" : vbt, "vbd" : vbd, "hi": np.zeros((2 * (M + N + 6),), dtype=np.float32)}
    pointers = Pointers(ctx,**kwargs)
    pd = pointers.alloc_on_device_only(N, M)
    pc = pointers.alloc()
    global_attributes = [np.int32(M), np.int32(N),\
                    pc['bienQ'], pc['daui'], pc['dauj'], pc['cuoii'], pc['cuoij'], pc['moci'], pc['mocj'], pc['khouot'], pc['boundary_type'], \
                    pc['h'], pc['v'], pc['u'], pc['z'], pc['t_u'], pc['t_v'], pc['t_z'], pc['Htdu'], pc['Htdv'], pc['H_moi'], pc['htaiz'],\
                    pc['ubt'], pc['ubp'], pc['vbt'], pc['vbd'], \
                    pc['hsnham'], pc['VISCOINDX'], pc['Kx1'], pc['Ky1'], pc['Tsyw'], pc['Tsxw'],\
                    pc['bc_up'], pc['bc_down'], pc['bc_left'], pc['bc_right'], pc['hi']]
    auxilary_arrays = [pd['a1'], pd['b1'], pd['c1'], pd['d1'], pd['a2'], pd['c2'], pd['d2'],\
                    pd['f1'], pd['f2'], pd['f3'], pd['f5'],\
                    pd['AA'], pd['BB'], pd['CC'], pd['DD'],\
                    pd['x'], pd['Ap'], pd['Bp'], pd['ep'], pd['SN'] ]


    arg_struct_ptr = cuda.mem_alloc(PointersStruct.arg_struct_size)
    arr_struct_ptr = cuda.mem_alloc(PointersStruct.arr_struct_size)
    arg_struct = PointersStruct(global_attributes, arg_struct_ptr)
    arr_struct = PointersStruct(auxilary_arrays, arr_struct_ptr, structtype='ARR')

    pointers.toDevice(['h', 'hsnham', 'VISCOINDX', 'bienQ', 'Tsyw', 'Tsxw', 'boundary_type', 'bc_up', 'bc_down', 'bc_right', 'bc_left'])
    ctx.synchronize()

    supplement = open("tmp.cu").read()
    supmod = SourceModule(supplement, include_dirs = ["/home/Pearl/mia/Riverbed-Erosion-Prediction/Python_Code/test"])

    init_Kernel = supmod.get_function("Onetime_init")
    Find_Calculation_limits_x = supmod.get_function("Find_Calculation_limits_Horizontal")
    Find_Calculation_limits_y = supmod.get_function("Find_Calculation_limits_Vertical")
    gpu_Htuongdoi  = supmod.get_function("Htuongdoi")
    # preprocess = supmod.get_function("preprocess_data")

    block_2d = (min(32, M + 3), 1, 1)
    grid_2d = ((M + 3) // min(32, M + 3) + 1, N + 3, 1)
    print grid_2d, block_2d
    init_Kernel(arg_struct_ptr, block=block_2d, grid=grid_2d)
    ctx.synchronize()
    Find_Calculation_limits_x(arg_struct_ptr, block=(32, 1, 1), grid=(1, N, 1))
    Find_Calculation_limits_y(arg_struct_ptr, block=(1, 32, 1), grid=(M, 1, 1))
    gpu_Htuongdoi(arg_struct_ptr, block=block_2d, grid=grid_2d)
    ctx.synchronize()
    gpu_dauj = np.zeros(dauj.shape, dtype=np.int32)
    gpu_cuoij = np.zeros(cuoij.shape, dtype=np.int32)
    cuda.memcpy_dtoh(gpu_dauj, pc['dauj'])
    cuda.memcpy_dtoh(gpu_cuoij, pc['cuoij'])
    print gpu_dauj[M], gpu_cuoij[M]

    uvbmod = SourceModule(open("universal_boundary.cu").read(),\
                      include_dirs = ["/home/Pearl/mia/Riverbed-Erosion-Prediction/Python_Code/test"])
    preprocess = uvbmod.get_function("preprocess_data")
    preprocess(arg_struct_ptr, block=(32, 1, 1), grid = (1, 1, 1))

    update_boundary_value = uvbmod.get_function("Update_Boundary_Value")
    update_boundary_value(np.float32(300), np.int32(total_time), arg_struct_ptr, block= (1, 1024, 1), grid= (1, max(M, N) // 1024 + 1, 1))
    ctx.synchronize()


    # hi = np.zeros((2 * (M + N + 6),), dtype=np.float32)
    # gpu_h = np.zeros(h.shape, dtype=np.float32)
    # cuda.memcpy_dtoh(hi, pc['hi'])
    # cuda.memcpy_dtoh(gpu_h, pc['h'])
    # print gpu_h[:, M]
    # print hi[: M + 3]
    
    # gpu_htdu = np.zeros(Htdu.shape,dtype=np.float32)
    # gpu_htdv = np.zeros(Htdv.shape,dtype=np.float32)
    # gpu_daui = np.zeros(daui.shape,dtype=np.int32)
    # gpu_dauj = np.zeros(dauj.shape,dtype=np.int32)
    # gpu_cuoii = np.zeros(cuoii.shape,dtype=np.int32)
    # gpu_cuoij = np.zeros(cuoij.shape,dtype=np.int32)
    # gpu_khouot = np.ones(khouot.shape,dtype=np.int32)

    # pointers.extract({"Htdu" : gpu_htdu, "Htdv" : gpu_htdv, "daui" : gpu_daui, "cuoii" : gpu_cuoii, "dauj" : gpu_dauj, "cuoij" : gpu_cuoij, "khouot" : gpu_khouot })

    # err_htdv = np.where(abs(gpu_htdv - Htdv ) > 1e-6)
    # err_htdu = np.where(abs(gpu_htdu - Htdu ) > 1e-6)
    # err_daui = np.where(abs(gpu_daui - daui ) != 0)
    # err_dauj = np.where(abs(gpu_dauj - dauj ) != 0)
    # err_cuoii = np.where(abs(gpu_cuoii - cuoii ) != 0)
    # err_cuoij = np.where(abs(gpu_cuoij - cuoij ) != 0)
    # err_ku = np.where(abs(gpu_khouot - khouot) != 0)

    # print khouot[:,5]
    # print gpu_khouot[:, 5]
    # print 'khouot:', err_ku
    # print "Htd :",  err_htdu
    # print "Htd :",  err_htdv
    # print "dau :",  err_daui
    # print "dau :",  err_dauj
    # print "cuoii:", err_cuoii
    # print "cuoij:", err_cuoij


    return pointers, ctx, arg_struct_ptr, arr_struct_ptr, supmod

def main():
    

    load_input_and_initialize()
    parser = argparse.ArgumentParser()
    #parser.add_argument("--days", type=int)
    #parser.add_argument("--hours", type=int)
    parser.add_argument("GPU", type=int)
    parser.add_argument("--Device", type=int)
    parser.add_argument("--mins", type=int, default=0)
    parser.add_argument("--plot")
    parser.add_argument("--test",default=True)

    args = parser.parse_args()
    
    
    
    #days, hours, mins = args.days, args.hours, args.mins
    days = 0; hours = 0;
    mins = args.mins
    Tmax = (days*24*60 + hours*60 + mins)* 60 + 0.4

    runOnGPU = args.GPU

    if runOnGPU > 0:
        cuda.init()
        device_no = args.Device
        # by default use GPU number 0
        if args.Device == None: 
            device_no = 0

        pointers, ctx, arg_struct_ptr, arr_struct_ptr, supmod = gpu_init(device_no)
        
        start_time = timeit.default_timer()
        hydraulic_Calculation(Tmax, pointers, arg_struct_ptr, arr_struct_ptr, supmod, ctx, sec=5, canal=1,plot=args.plot)
        ctx.detach()
        stop_time = timeit.default_timer()
        print stop_time - start_time
    else:
        start_time = timeit.default_timer()

       
        cpu_hydraulic_Calculation(Tmax, ctx, canal=1,plot=args.plot)

        stop_time = timeit.default_timer()
        print stop_time - start_time
    #print time.localtime()

    
    #cuda.stop()

main()


