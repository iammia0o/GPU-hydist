'''
ULSAN NATIONAL INSTIUTE OF SCIENCE AND TECHNOLOGY
Copyright (c) 2019 HVCL lab
Created by Huong Nguyen

'''

from __future__ import division

import timeit

import numpy as np 
from Global_Variables import *
from Engine import *
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
parser.add_argument("--mins", type=int, default=0)
parser.add_argument("--hours", type=int, default=10)
parser.add_argument("--plot", default=False)
parser.add_argument("--test",default=False)
parser.add_argument("--sediment", default=10)
parser.add_argument("--bed_change", default=20)
parser.add_argument("--pick_up", default=False)
parser.add_argument("--pickup_dirs", default='/Input/Initial_Condition')
parser.add_argument("--kenhhepng",default=False)
parser.add_argument("--kenhhepd",default=False)

parser.add_argument("--include_dirs", default="/home/Pearl/mia/VH-Project/Code/test")


args = parser.parse_args()
def load_intial_condition(dirs, pointers):
    u = np.loadtxt(dirs + 'u.txt',skiprows=1)
    u = np.pad(u,((0,1), (0,1)), 'edge')
    v = np.loadtxt(dirs + 'v.txt',skiprows=1)
    v = np.pad(v, ((0,1), (0,1)), 'edge')
    z = np.loadtxt(dirs + 'z.txt',skiprows=1)
    
    z= np.pad (z, ((0,1), (0,1)), 'edge')
    khouot = np.loadtxt(dirs = 'khouot.txt', dtype=np.int32)
    khouot = np.pad(khouot, ((0,1), (0,1)), 'constant', constant_values=((2,2),(2,2)))
    FS = np.loadtxt(dirs + 'FS.txt')
    FS = np.pad(FS, ((0,1), (0,1)), 'edge')
    pointers.toDevice(['u', 'v', 'z', 'khouot', 'FS'])


def gpu_init(device_no, pick_up, pickup_dirs):
    cuda.init()
    dev = cuda.Device(device_no) # the number of GPU
    ctx = dev.make_context()
    kwargs = {"h": h, "hsnham": hsnham, "VISCOINDX" : VISCOINDX, "H_xomoi": H_moi, "bienQ" : bienQ,\
              "moci" : moci ,"mocj" : mocj, "dauj": dauj, "daui" : daui, "cuoii" : cuoii, "cuoij" : cuoij,\
              "Tsxw" : Tsxw, "Tsyw" : Tsyw, "khouot" : khouot, "boundary_type" : boundary_type,\
              "u": u, "v": v, "z" : z, "t_u": t_u, "t_v": t_v, "t_z": t_z, "Htdu": Htdu, "Htdv" : Htdv, \
              "Kx1" : Kx1, "Ky1" : Ky1, "htaiz" : htaiz, "htaiz_bd" : htaiz,\
              "bc_up": bc_up, "bc_down": bc_down, "bc_left": bc_left, "bc_right" : bc_right,\
              "ubt" : ubt, "ubp" : ubp, "vbt" : vbt, "vbd" : vbd, "hi": hi,\
              "FS" : FS, 'tFS': tFS, 'CC_u' : CC_u, 'CC_d' : CC_d, 'CC_l' : CC_l, 'CC_r' : CC_r,\
              "VTH": VTH, "Kx" : Kx, "Ky" : Ky, "Fw" : Fw, "Qbx" : Qbx, "Qby" : Qby, "dH" : dH}

    # create a pointer object that store address of pointers on device
    pointers = Pointers(ctx,dtype=np.float64,**kwargs)

    # pointers = Pointers(ctx,**kwargs)
    # hmax is used to calculate boundary condition, this will be recalculated later on 
    # in pre_processing kernel 
    hmax = np.max(h[2])
    # print hmax

    # allocate memory on device
    pd = pointers.alloc_on_device_only(N, M)
    pc = pointers.alloc()

    # store pointers on a list to transfer it to gpu
    # hmax here are just dummie values, for address alignment 
    # so that pointers of other arrays can be copied to the right place in memory
    global_attributes = [np.int32(M), np.int32(N), floattype(hmax), floattype(hmax), floattype(hmax), floattype(hmax),\
                    pc['bienQ'], pc['daui'], pc['dauj'], pc['cuoii'], pc['cuoij'], pc['moci'], pc['mocj'], pc['khouot'], pc['boundary_type'],\
                    pc['h'], pc['v'], pc['u'], pc['z'], pc['t_u'], pc['t_v'], pc['t_z'], pc['Htdu'], pc['Htdv'], pc['H_moi'], pc['htaiz'],\
                    pc['htaiz_bd'], pc['ubt'], pc['ubp'], pc['vbt'], pc['vbd'], \
                    pc['hsnham'], pc['VISCOINDX'], pc['Kx1'], pc['Ky1'], pc['Tsyw'], pc['Tsxw'],\
                    pc['bc_up'], pc['bc_down'], pc['bc_left'], pc['bc_right'], pc['hi'],\
                    pc['FS'], pc['tFS'], pc['CC_u'], pc['CC_d'], pc['CC_l'], pc['CC_r'],\
                    pc['VTH'], pc['Kx'], pc['Ky'], pc['Fw'], pc['Qbx'], pc['Qby'], pc['dH']]

    auxilary_arrays = [pd['a1'], pd['b1'], pd['c1'], pd['d1'], pd['a2'], pd['c2'], pd['d2'], \
                   pd['f1'], pd['f2'], pd['f3'], pd['f5'],\
                   pd['AA'], pd['BB'], pd['CC'], pd['DD'],\
                   pd['x'], pd['Ap'], pd['Bp'], pd['ep'], pd['SN'] ]


    # copy struct to gpu: struct that store attribute arrays
    arg_struct_ptr = cuda.mem_alloc(np.intp(0).nbytes * (len(global_attributes) - 6) + 8 + 4 * np.dtype(floattype).itemsize)
    
    # copy struct to gpu: struct that store supporting arrays (i.e. arrays that only exist on device and don't have corresponding arrays on host)
    arr_struct_ptr = cuda.mem_alloc(np.intp(0).nbytes * len(auxilary_arrays))
    arg_struct = PointersStruct(global_attributes, arg_struct_ptr)
    arr_struct = PointersStruct(auxilary_arrays, arr_struct_ptr, structtype='ARR')
    pointers.toDevice(['h', 'hsnham', 'VISCOINDX', 'bienQ', 'Tsyw', 'Tsxw', 'boundary_type', 'bc_up', 'bc_down', 'bc_left', 'bc_right',\
                      'u', 'v', 'z', 'CC_u', 'CC_d', 'CC_l', 'CC_r', 'Fw'])
    ctx.synchronize()


    supplement = open("support_funcs.cu").read()
    supmod = SourceModule(supplement, include_dirs = [os.getcwd()])

    # get functions from cuda file
    init_Kernel = supmod.get_function("Onetime_init")
    Find_Calculation_limits_x = supmod.get_function("Find_Calculation_limits_Horizontal")
    Find_Calculation_limits_y = supmod.get_function("Find_Calculation_limits_Vertical")
    gpu_Htuongdoi  = supmod.get_function("Htuongdoi")
    preprocess = supmod.get_function("preprocess_data")

    # declare block size and grid size
    block_2d = (min(32, M + 3), 1, 1)
    grid_2d = ((M + 3) // min(32, M + 3) + 1, N + 3, 1)

    # call intialize kernels
    init_Kernel(arg_struct_ptr, block=block_2d, grid=grid_2d)
    ctx.synchronize()

    if pick_up is True:
        load_intial_condition(dirs, pointers)
    Find_Calculation_limits_x(arg_struct_ptr, block=(32, 1, 1), grid=(1, N, 1))
    Find_Calculation_limits_y(arg_struct_ptr, block=(32, 1, 1), grid=(1, M, 1))
    gpu_Htuongdoi(arg_struct_ptr, block=block_2d, grid=grid_2d)
    ctx.synchronize()
    preprocess(arg_struct_ptr, block=(32, 1, 1), grid = (1, 1, 1))
  

    return pointers, ctx, arg_struct_ptr, arr_struct_ptr, supmod


def main():
       
    
    #days, hours, mins = args.days, args.hours, args.mins
    hours = args.hours;
    mins = args.mins
    secs = 0
    Tmax = hours*3600 + mins* 60 + secs
    plot = args.plot
    debug = False
    sediment_start = args.sediment * 3600
    bedchange_start = args.bed_change * 3600
    interval = 3600
    export = False
    save_grid = False

    cuda.init()
    device_no = args.Device
    # by default use GPU number 0
    pointers, ctx, arg_struct_ptr, arr_struct_ptr, supmod = gpu_init(device_no, args.pick_up, args.pickup_dirs)
    start_time = timeit.default_timer()
    ulist, vlist, zlist, fslist = hydraulic_Calculation(Tmax, pointers, arg_struct_ptr, arr_struct_ptr, supmod, ctx, kenhhepd=args.kenhhepd, kenhhepng=args.kenhhepng,\
                                save_grid=save_grid, sediment_start=sediment_start, debug=debug, plot=plot, interval=interval, export=export, bed_change_start=bedchange_start)
    stop_time = timeit.default_timer()
    # print stop_time - start_time
    print "total running time of ", hours, "hours is ",  stop_time - start_time
    ctx.detach()

main()