
# coding: utf-8

# In[1]:


from __future__ import division
import timeit
import numpy as np 
import os
from Coeff import *
from Global_Variables import *
from Test_engine import *
import pycuda.driver as cuda
#import pycuda.autoinit
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
from Pointers import Pointers
from Pointers import PointersStruct
import glob
import skimage.io as io
import timeit



cuda.init()
dev = cuda.Device(1) # the number of GPU
ctx = dev.make_context()
kwargs = {"h": h, "hsnham": hsnham, "VISCOINDX" : VISCOINDX, "H_moi": H_moi, "bienQ" : bienQ,          "moci" : moci ,"mocj" : mocj, "dauj": dauj, "daui" : daui, "cuoii" : cuoii, "cuoij" : cuoij,          "Tsxw" : Tsxw, "Tsyw" : Tsyw, "khouot" : khouot, "boundary_type" : boundary_type,          "u": u, "v": v, "z" : z, "t_u": t_u, "t_v": t_v, "t_z": t_z, "Htdu": Htdu, "Htdv" : Htdv,          "Kx1" : Kx1, "Ky1" : Ky1, "htaiz" : htaiz, "htaiz_bd" : htaiz,           "bc_up": bc_up, "bc_down": bc_down, "bc_left": bc_left, "bc_right" : bc_right,          "ubt" : ubt, "ubp" : ubp, "vbt" : vbt, "vbd" : vbd, "hi": hi,           "FS" : FS, 'tFS': tFS, 'CC_u' : CC_u, 'CC_d' : CC_d, 'CC_l' : CC_l, 'CC_r' : CC_r,          "VTH": VTH, "Kx" : Kx, "Ky" : Ky, "Fw" : Fw, "Qbx" : Qbx, "Qby" : Qby, "dH" : dH}
pointers = Pointers(ctx,dtype=np.float64,**kwargs)
# pointers = Pointers(ctx,**kwargs)
hmax = np.max(h[2])
# print hmax
pd = pointers.alloc_on_device_only(N, M)
pc = pointers.alloc()
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



arg_struct_ptr = cuda.mem_alloc(np.intp(0).nbytes * (len(global_attributes) - 6) + 8 + 4 * np.dtype(floattype).itemsize)
arr_struct_ptr = cuda.mem_alloc(np.intp(0).nbytes * len(auxilary_arrays))
arg_struct = PointersStruct(global_attributes, arg_struct_ptr)
arr_struct = PointersStruct(auxilary_arrays, arr_struct_ptr, structtype='ARR')
pointers.toDevice(['h', 'hsnham', 'VISCOINDX', 'bienQ', 'Tsyw', 'Tsxw', 'boundary_type', 'bc_up', 'bc_down', 'bc_left', 'bc_right',                  'u', 'v', 'z', 'CC_u', 'CC_d', 'CC_l', 'CC_r', 'Fw'])
ctx.synchronize()

supplement = open("tmp.cu").read()
supmod = SourceModule(supplement, include_dirs = [os.getcwd()])
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


# In[7]:



one_hour = 3600
hours = 30
mins = 0
secs = 0
Tmax = hours*3600 + mins* 60 + secs
plot = True
debug = False
sediment_start = 10 * 3600 + 1
interval = 3600
export = False
save_grid = False


start_time = timeit.default_timer()
# ulist, vlist, zlist, fslist = 
hydraulic_Calculation(Tmax, pointers, arg_struct_ptr, arr_struct_ptr, supmod, ctx,\
                    save_grid=save_grid, sediment_start=sediment_start, debug=debug,\
                    plot=plot, interval=interval, export=export)
stop_time = timeit.default_timer()
print stop_time - start_time
# print stop_time
