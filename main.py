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
float_type = np.float64

parser = argparse.ArgumentParser()
    #parser.add_argument("--days", type=int)
    #parser.add_argument("--hours", type=int)
parser.add_argument("--GPU", type=int, default=1)
parser.add_argument("--Device", type=int, default=0)
parser.add_argument("--mins", type=int, default=10)
parser.add_argument("--hours", type=int, default=0)
parser.add_argument("--plot", default=False)
parser.add_argument("--test",default=True)
parser.add_argument("--include_dirs", default="/home/Pearl/mia/VH-Project/Code/test")

args = parser.parse_args()


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
           "ubt" : ubt, "ubp" : ubp, "vbt" : vbt, "vbd" : vbd, "hi": np.zeros((2 * (M + N + 6),), dtype=float_type)}
    pointers = Pointers(ctx,dtype=np.float64,**kwargs)

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
    pointers.toDevice(['h', 'hsnham', 'VISCOINDX', 'bienQ', 'Tsyw', 'Tsxw', 'boundary_type', 'bc_up', 'bc_down', 'bc_left', 'bc_right',\
                      'u', 'v', 'z'])
    ctx.synchronize()

    supplement = open("tmp.cu").read()
    supmod = SourceModule(supplement, include_dirs = [args.include_dirs])
    init_Kernel = supmod.get_function("Onetime_init")
    Find_Calculation_limits_x = supmod.get_function("Find_Calculation_limits_Horizontal")
    Find_Calculation_limits_y = supmod.get_function("Find_Calculation_limits_Vertical")
    gpu_Htuongdoi  = supmod.get_function("Htuongdoi")
    preprocess = supmod.get_function("preprocess_data")

    block_2d = (min(32, M + 3), 1, 1)
    grid_2d = ((M + 3) // min(32, M + 3) + 1, N + 3, 1)
    init_Kernel(arg_struct_ptr, block=block_2d, grid=grid_2d)
    ctx.synchronize()
    Find_Calculation_limits_x(arg_struct_ptr, block=(32, 1, 1), grid=(1, N, 1))
    Find_Calculation_limits_y(arg_struct_ptr, block=(32, 1, 1), grid=(1, M, 1))
    gpu_Htuongdoi(arg_struct_ptr, block=block_2d, grid=grid_2d)
    ctx.synchronize()
    preprocess(arg_struct_ptr, block=(32, 1, 1), grid = (1, 1, 1))
    ctx.synchronize()
    # gpu_dauj = np.zeros(dauj.shape, dtype=np.int32)
    # gpu_cuoij = np.zeros(cuoij.shape, dtype=np.int32)
    # cuda.memcpy_dtoh(gpu_dauj, pc['dauj'])
    # cuda.memcpy_dtoh(gpu_cuoij, pc['cuoij'])
    # print gpu_dauj[M], gpu_cuoij[M]

    return pointers, ctx, arg_struct_ptr, arr_struct_ptr, supmod

def main():
       
    
    #days, hours, mins = args.days, args.hours, args.mins
    days = 0; hours = args.hours;
    mins = args.mins
    secs = 0
    Tmax = hours*3600 + mins* 60 + secs

    runOnGPU = args.GPU

    if runOnGPU > 0:
        cuda.init()
        device_no = args.Device
        # by default use GPU number 0
        if args.Device == None: 
            device_no = 0

        pointers, ctx, arg_struct_ptr, arr_struct_ptr, supmod = gpu_init(device_no)
        canal = kenhhepng or kenhhepd
        start_time = timeit.default_timer()
        ulist, vlist, zlist = hydraulic_Calculation(Tmax, pointers, arg_struct_ptr, arr_struct_ptr, supmod, ctx, blk_size=1024,canal=canal, plot=args.plot, export=False)
        stop_time = timeit.default_timer()
        print "total running time of ", hours, "hours is ",  stop_time - start_time
        ctx.detach()
    else:
        start_time = timeit.default_timer()

        load_input_and_initialize()
        
        cpu_hydraulic_Calculation(Tmax, ctx, canal=1,plot=args.plot)

        stop_time = timeit.default_timer()
        print stop_time - start_time
    #print time.localtime()

    
    #cuda.stop()

main()


