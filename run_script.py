
# coding: utf-8

# In[1]:


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
from Pointers import PointersStruct
import skimage.io as io
# get_ipython().magic(u'matplotlib inline')


# In[2]:


U = io.imread ("U.tif")
V = io.imread ("V.tif")
Z = io.imread ("Z.tif")


# In[3]:


cuda.init()
dev = cuda.Device(1) # the number of GPU
ctx = dev.make_context()
kwargs = {"h": h, "hsnham": hsnham, "VISCOINDX" : VISCOINDX, "H_moi": H_moi, "bienQ" : bienQ,            "moci" : moci ,"mocj" : mocj, "dauj": dauj, "daui" : daui, "cuoii" : cuoii, "cuoij" : cuoij,            "Tsxw" : Tsxw, "Tsyw" : Tsyw, "khouot" : khouot, "boundary_type" : boundary_type,            "u": u, "v": v, "z" : z, "t_u": t_u, "t_v": t_v, "t_z": t_z, "Htdu": Htdu, "Htdv" : Htdv,            "Kx1" : Kx1, "Ky1" : Ky1, "htaiz" : htaiz, "ut" : ut, "vt": vt,             "bc_up": bc_up, "bc_down": bc_down, "bc_left": bc_left, "bc_right" : bc_right,           "ubt" : ubt, "ubp" : ubp, "vbt" : vbt, "vbd" : vbd, "hi": np.zeros((2 * (M + N + 6),), dtype=np.float32)}
pointers = Pointers(ctx,**kwargs)
# pointers = Pointers(ctx,**kwargs)

pd = pointers.alloc_on_device_only(N, M)
pc = pointers.alloc()
global_attributes = [np.int32(M), np.int32(N),                pc['bienQ'], pc['daui'], pc['dauj'], pc['cuoii'], pc['cuoij'], pc['moci'], pc['mocj'], pc['khouot'], pc['boundary_type'],                 pc['h'], pc['v'], pc['u'], pc['z'], pc['t_u'], pc['t_v'], pc['t_z'], pc['Htdu'], pc['Htdv'], pc['H_moi'], pc['htaiz'],                pc['ubt'], pc['ubp'], pc['vbt'], pc['vbd'],                 pc['hsnham'], pc['VISCOINDX'], pc['Kx1'], pc['Ky1'], pc['Tsyw'], pc['Tsxw'],                pc['bc_up'], pc['bc_down'], pc['bc_left'], pc['bc_right'], pc['hi']]
auxilary_arrays = [pd['a1'], pd['b1'], pd['c1'], pd['d1'], pd['a2'], pd['c2'], pd['d2'],                pd['f1'], pd['f2'], pd['f3'], pd['f5'],                pd['AA'], pd['BB'], pd['CC'], pd['DD'],                pd['x'], pd['Ap'], pd['Bp'], pd['ep'], pd['SN'] ]


# In[4]:


arg_struct_ptr = cuda.mem_alloc(PointersStruct.arg_struct_size)
arr_struct_ptr = cuda.mem_alloc(PointersStruct.arr_struct_size)
arg_struct = PointersStruct(global_attributes, arg_struct_ptr)
arr_struct = PointersStruct(auxilary_arrays, arr_struct_ptr, structtype='ARR')
pointers.toDevice(['h', 'hsnham', 'VISCOINDX', 'bienQ', 'Tsyw', 'Tsxw', 'boundary_type', 'bc_up', 'bc_down', 'bc_left', 'bc_right',                  'u', 'v', 'z'])
ctx.synchronize()

supplement = open("tmp.cu").read()
supmod = SourceModule(supplement, include_dirs = ["/home/Pearl/mia/Riverbed-Erosion-Prediction/Python_Code/test"])
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


# In[5]:


def load_input_and_initialize():
    TinhKhoUot()
    Htuongdoi()
    K_factor()
    GiatriHtaiZ()
    #Initial_condition(Dirs['U_file'], Dirs['V_file'], Dirs['Z_file']) # this can load new 
    # Load initial condition (tinh tiep hoac tinh tu dau)
    Find_Calculation_limits()
load_input_and_initialize()


# In[6]:


gpu_boundary_type = np.zeros(boundary_type.shape, dtype=np.int32)
cuda.memcpy_dtoh(gpu_boundary_type, pc['boundary_type'])
gpu_left = np.zeros(bc_left.shape, dtype=np.float32)
cuda.memcpy_dtoh(gpu_left, pc['bc_left'])
gpu_bienQ = np.zeros(bienQ.shape, dtype=np.int32)
cuda.memcpy_dtoh(gpu_bienQ, pc['bienQ'])
gpu_htdu = np.zeros(Htdu.shape,dtype=np.float32)
gpu_htdv = np.zeros(Htdv.shape,dtype=np.float32)
gpu_daui = np.zeros(daui.shape,dtype=np.int32)
gpu_dauj = np.zeros(dauj.shape,dtype=np.int32)
gpu_cuoii = np.zeros(cuoii.shape,dtype=np.int32)
gpu_cuoij = np.zeros(cuoij.shape,dtype=np.int32)
gpu_khouot = np.ones(khouot.shape,dtype=np.int32)
gpu_tz = np.zeros(t_u.shape, dtype=np.float32)
gpu_tu = np.zeros(t_u.shape, dtype=np.float32)
gpu_tv = np.zeros(t_u.shape, dtype=np.float32)

pointers.extract({"Htdu" : gpu_htdu, "Htdv" : gpu_htdv, "daui" : gpu_daui, "cuoii" : gpu_cuoii, "dauj" : gpu_dauj, "cuoij" : gpu_cuoij, "khouot" : gpu_khouot })
# print np.where(gpu_htdu[:,10] != 0)
err_htdv = np.where(abs(gpu_htdv - Htdv ) > 1e-6)
err_htdu = np.where(abs(gpu_htdu - Htdu ) > 1e-6)
err_daui = np.where(abs(gpu_daui - daui ) != 0)
err_dauj = np.where(abs(gpu_dauj - dauj ) != 0)
err_cuoii = np.where(abs(gpu_cuoii - cuoii ) != 0)
err_cuoij = np.where(abs(gpu_cuoij - cuoij ) != 0)
err_ku = np.where(abs(gpu_khouot - khouot) != 0)   
# print khouot[:,5]
# print gpu_khouot[:, 5]
# print 'khouot:', err_ku
# print "Htdu:",  err_htdu
# print "Htdv:",  err_htdv
# print "daui :",  err_daui
# print "dauj :",  err_dauj
# print "cuoii:", err_cuoii
# print "cuoij:", err_cuoij   


# In[7]:


h.shape
print dauj[3], cuoij[3]
(h[1, 3] + h[1, 1]) / 2 - NANGDAY
print cuoij[3]


# In[8]:


# uvbmod = SourceModule(open("universal_boundary.cu").read(),\
#                       include_dirs = ["/home/Pearl/mia/Riverbed-Erosion-Prediction/Python_Code/test"])
# preprocess = uvbmod.get_function("preprocess_data")
# preprocess(arg_struct_ptr, block=(32, 1, 1), grid = (1, 1, 1))
# hi = np.zeros((2 * (M + N + 6),), dtype=np.float32)
# cuda.memcpy_dtoh(hi, pc['hi'])
# hi[: M + 3]


# In[9]:


# cuda.memcpy_dtoh(gpu_dauj, pc['dauj'])
# cuda.memcpy_dtoh(gpu_cuoij, pc['cuoij'])
# print M,  gpu_cuoij[M]


# In[10]:


# update_boundary_value = uvbmod.get_function("Update_Boundary_Value")
# update_boundary_value(np.float32(0.5), np.int32(total_time), arg_struct_ptr, block= (1, 1024, 1), grid= (1, max(M, N) // 1024 + 1, 1))
# ctx.synchronize()
# # gpu_boundary_type = np.zeros(boundary_type.shape, dtype=np.int32)
# # cuda.memcpy_dtoh(gpu_boundary_type, pc['boundary_type'])
# gpu_ubt = np.zeros(ubt.shape, dtype=np.int32)
# cuda.memcpy_dtoh(gpu_ubt, pc['ubt'])


# In[11]:


ulist, vlist, zlist = hydraulic_Calculation(10, pointers, arg_struct_ptr, arr_struct_ptr, supmod, ctx, debug=True, plot=True)


# In[12]:


# print ulist[2][1] 
ulist = np.array(ulist)
vlist = np.array(vlist)
zlist = np.array(zlist)



# In[13]:


# print ulist[:, 1, 5]
# print ulist[2, 1]
# print np.max(abs(U[0] - ulist[0, 1 : N + 1, 1 : M + 1]))
for i in range(U.shape[0]):
    print i, 0.25 + i * 0.25
    print 'u', np.max(abs(U[i] - ulist[i, 1 : N + 1, 1 : M + 1]))
#     print 'z', np.max(abs(Z[i] - zlist[i, 1 : N + 1, 1 : M + 1]))
    print 'v', np.max(abs(V[i] - vlist[i, 1 : N + 1, 1 : M + 1]))
    print np.where(abs(U[i] - ulist[i, 1 : N + 1, 1 : M + 1]) > 1e-6)

print vlist[8,3,1]
#     print np.max(abs(U[i, 0] - ulist[i, 1, 1 : M + 1]))
#     if np.max(abs(U[i, 0] - ulist[i, 1, 1 : M + 1])) > 1e-5:
#         print 0.25 + i * 0.25, i , np.max(abs(U[i, 0] - ulist[i, 1, 1 : M + 1])), np.where(abs(U[i, 0] - ulist[i, 1, 1 : M + 1]) > 1e-5)


# In[ ]:


print ulist[11,1, 3:51] - U[11, 0, 2:50]
# print U[11, 0, 2:50]


# In[ ]:


print np.max(U[23, 0, 2: 50] - ulist[23,1, 3:51])
print U[23, 0, 2:50]
print ulist[23,1, 3:51]


# In[ ]:


gpu_ubt = np.zeros(ubt.shape, dtype=np.float32)
cuda.memcpy_dtoh(gpu_ubt, pc['ubt'])
cuda.memcpy_dtoh(gpu_htdu,pc['Htdu'])


# In[ ]:


plt.imshow(gpu_htdu)


# In[ ]:


cuda.memcpy_dtoh(gpu_tz, pc['t_z'])
cuda.memcpy_dtoh(gpu_tu, pc['t_u'])
cuda.memcpy_dtoh(gpu_tv, pc['t_v'])
# pointers.extract({"Htdu" : gpu_htdu, "Htdv" : gpu_htdv, "daui" : gpu_daui, "cuoii" : gpu_cuoii, "dauj" : gpu_dauj, "cuoij" : gpu_cuoij, "khouot" : gpu_khouot })
# print np.where(gpu_htdu[:,10] != 0)
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
# print "Htdu:",  err_htdu
# print "Htdv:",  err_htdv
# print "daui :",  err_daui
# print "dauj :",  err_dauj
# print "cuoii:", err_cuoii
# print "cuoij:", err_cuoij  


# In[ ]:


plt.imshow(gpu_tu[1:5, :50])
# plt.imshow(gpu_tz)


# In[ ]:


# u1 = np.loadtxt("Outputs/Song_Luy/Test_u_1/u_0.25.txt")
# generated from code that Tai give
# u2 = np.loadtxt("Outputs/Song_Luy/U/u_0.25.txt")
# z2 = np.loadtxt("Outputs/Song_Luy/Z/v_0.75.txt")
# v2 = np.loadtxt("Outputs/Song_Luy/V/v_39.txt")
#given by Tai
# u3 = np.loadtxt("Outputs/Song_Luy/Test_u/u_0.25s.txt")


# In[ ]:


# plt.imshow(u2)
# plt.imshow(u2[20:21,:20])
# plt.show()
# plt.imshow(gpu_tu[21:22, 1:51])
# plt.show()
# plt.imshow(z1[:5, :50])
# plt.show()


# plt.imshow(gpu_tu[1:5, 2:51])


# In[ ]:


# print cuoij[8]
# print U


# In[ ]:


u2 = U[19]
z2 = Z[19]
v2 = V[19]
# print u2[1, 2:50]
# print v2[1, 2:50]
# print z2[1, 31:50]
# print gpu_tz[2, 32:51]
# print gpu_tu[2, 3:51]
# print gpu_tv[2, 3:51]
# print np.where(abs(u2 -  gpu_tu[1: N + 1, 1: M + 1]) > 1e-5)
print np.max(abs(u2 - gpu_tu[1: N + 1, 1: M + 1]))
# print np.where(abs(z2 - gpu_tz[1: N + 1, 1: M + 1]) > 1e-5)
print np.max(abs(z2 - gpu_tz[1: N + 1, 1: M + 1]))
# print np.where(abs(v2 - gpu_tv[1:N + 1, 1 : M + 1]) > 1e-5)
print np.max(abs(v2 - gpu_tv[1:N + 1, 1 : M + 1]))
# g_htdu = np.loadtxt("Htdu.txt")
# g_htdu = g_htdu.reshape((N, M))
# pointers.extract({"Htdu" : gpu_htdu})
# print g_htdu[1, 2: 50]
# print gpu_htdu[2, 3:51]
# print np.where(abs(g_htdu - gpu_htdu[1 : N + 1, 1 : M + 1]) > 1e-5)
# [0.06593714 0.08957479 0.1032139  0.10478815 0.10292074 0.10115254
#  0.10046777 0.10221902 0.10644172 0.10743125 0.09413933 0.06584784
#  0.04135729 0.02753221 0.01145492 0.00155177
# print cuoij[35,0]
# print U[0, 2, 34]


# In[ ]:


# print gpu_ubt[3:51]
# print z1[2, 2:50]
print gpu_tz[3, 3:51]
# print np.max(abs(z1[2, 2:50] - gpu_tz[3, 3:51]))
#(array([  8,  16,  21,  27,  32,  37,  42,  47,  51,  55,  59,  63,  67,
#         72,  81,  91, 101, 110, 118, 120, 122, 124, 127, 129, 131, 133,
#        135, 137, 138, 140, 142, 143, 144, 146, 147, 149, 150, 151, 153,
#        154, 155, 156, 157, 158, 159, 160, 161, 162]), array([ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
#        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
#        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]))


# In[ ]:


# mauso = 0
#                             For k = daui(2, j) To cuoii(2, j)
#                                 hi(k) = (h(1, k - 1) + h(1, k)) / 2 - NANGDAY
#                                 mauso = mauso + hi(k) ^ (5 / 3)
#                             Next k
                            
#                             For jj = 0 To time
#                                 temp = bientrai(j, jj) / mauso
#                                 For k = daui(2, j) To cuoii(2, j)
#                                     qi(k) = temp * hi(k) ^ (2 / 3)
#                                     vbientrai(jj, k) = qi(k) / dX
#                                 Next k
#                             Next jj


# In[ ]:


hi = np.zeros(h.shape[1], dtype=np.float32)
acc = 0
for k in range(daui[2, 0], cuoii[2, 0] + 1):
    hi[k] = (h[1, k -1] + h[1, k]) * 0.5 
    acc += mth.pow(hi[k], 5.0/3.0)
tmp = bc_left[0] / acc
qi = np.zeros(hi.shape, dtype=np.float32)
qi_1 = tmp * np.power(hi, 2.0/3.0)
qi_2 = bc_left[1] / acc * np.power(hi, 2.0/3.0)
# print (qi_1 / dX)[3:51]
res = 1.0 / dX * (qi_1 * (1 - 6.0/3600) + qi_2 * (6.0/3600))
print res[3:51]


# In[ ]:


print ulist[23, 1, 3:51]
print U[23, 0, 2:50]
print np.where(abs(ulist[23, 1, 1: M + 1] - U[23, 0]) > 1e-6)


# In[ ]:


def sort_file (names):
    tmp = []
    for name in names:
        name_ = [float (name [21:-4]), name]
        tmp += [name_]
    tmp.sort ()
    ret = []
    for name in tmp:
        ret += [name [1]]
    return ret


# In[ ]:


# import glob
# Z = []
# U = []
# V = []



# Z_files = glob.glob ('Outputs/Song_Luy/Z/*.txt')
# Z_files = sort_file (Z_files)
# U_files = glob.glob ('Outputs/Song_Luy/U/*.txt')
# U_files = sort_file (U_files)
# V_files = glob.glob ('Outputs/Song_Luy/V/*.txt')
# V_files = sort_file (V_files)


# for file_name in Z_files:
# #     print (file_name)
#     tmp = np.loadtxt (file_name)
#     Z += [tmp]
# for file_name in U_files:
# #     print (file_name)
#     tmp = np.loadtxt (file_name)
#     U += [tmp]
# for file_name in V_files:
# #     print (file_name)
#     tmp = np.loadtxt (file_name)
#     V += [tmp]
# U = np.array (U)
# V = np.array (V)
# Z = np.array (Z)
# import skimage.io as io
# io.imsave ("U.tif", U.astype (np.float32))
# io.imsave ("V.tif", V.astype (np.float32))
# io.imsave ("Z.tif", Z.astype (np.float32))

