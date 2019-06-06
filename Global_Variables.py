# this module is to declare all the variables that being used across different modules
# phan tinh song va thuy luc da so gom tao mang trong.
# o day chi de cap cac thong so la hang so dung trong tinh toan
#global variable

import numpy as np
from math import * 
from Coeff import *
import time
    
#thong so thoi gian tinh
year = 0
month = 0
day = 0
hour = 1

# tong thoi gian tinh theo giay
Tmax = ((year * 365 + month * 30 + day) * 24 + hour) * 3600    
t1 = 0
thoigian = 0
    
######################################
# Arrays
# N rows, M columns
# load some global arrays
with open('DirsOfInputs_1.txt', 'r') as inf:
        Dirs = dict([line.split() for line in inf])

boundary_type = np.zeros((4, segment_limit), dtype=np.int32)
with open(Dirs['boundary_type'], 'r') as inf:
	i = 0
	for line in inf:
		inp = line.strip()
		if inp != 'NA':
			boundary_type[i, :len(inp.split())] = [value for value in inp.split()]
		i += 1

bc_up = np.transpose(np.loadtxt(Dirs['up']))
bc_down = np.transpose(np.loadtxt(Dirs['down']))
bc_left = np.transpose(np.loadtxt(Dirs['left']))
bc_right = np.transpose(np.loadtxt(Dirs['right']))

CC_u = np.transpose(np.loadtxt(Dirs['CC_u']))
CC_d = np.transpose(np.loadtxt(Dirs['CC_d']))
CC_l = np.transpose(np.loadtxt(Dirs['CC_l']))
CC_r = np.transpose(np.loadtxt(Dirs['CC_r']))


blist = [bc_up, bc_down, bc_left, bc_right]
# print blist[ np.where(boundary_type != 0) [0] [0] ]
total_time = len(blist[ np.where(boundary_type != 0) [0] [0] ] ) / (np.count_nonzero(boundary_type[np.where(boundary_type != 0)[0][0]]))
print total_time

bienQ = np.zeros((4,), dtype=np.int32)
print bienQ
#boundary_type:
#0: bien ran
#1: bien Z
#2: bien Q
#1 va 2 deu tuong ung voi bien long, 
#tuy nhien tuy vao dk thuc the do dac ma co the la bien muc nuoc hoac la bien luu luong
# danh dau bienQ de xac dinh xem bien la bien luu luong hay khong, neu la bien luu luong
# thi se interprete dieu kien bien khac di
bienQ[np.where(boundary_type == 2)[0]] = 1

h = np.loadtxt(Dirs['topography'])
h = np.transpose(h)
N, M = h.shape
h = np.flip(h, 1)
h = np.pad(h, ((1, 2), (1, 2)), 'edge')
h = h + NANGDAY

hsnham = np.loadtxt(Dirs['roughness_map'])
hsnham = np.transpose(hsnham)
hsnham = np.flip(hsnham, 1)
hsnham = np.pad(hsnham, ((1, 2), (1, 2)), 'edge')
# VISCOINDX: hsnhot
VISCOINDX = np.loadtxt(Dirs['visco_map'])
VISCOINDX = np.transpose(VISCOINDX)
VISCOINDX = np.flip(VISCOINDX, 1)
VISCOINDX = np.pad(VISCOINDX, ((1, 2), (1, 2)), 'edge')


shape = (N + 3, M + 3)
khouot = np.ones(shape).astype(np.int32) * 2
u = np.zeros(shape)
v = np.zeros(shape)
z = np.zeros(shape)
t_u = np.zeros(shape)
t_v = np.zeros(shape)
t_z = np.zeros(shape)
Kx1 = np.zeros(shape)
Ky1 = np.zeros(shape)
Htdu = np.zeros(shape)
Htdv = np.zeros(shape)
htaiz = np.zeros(shape)
moci = np.zeros(N + 2, dtype=np.int32)
mocj = np.zeros(M + 2, dtype=np.int32)
daui = np.zeros((N + 2, segment_limit), dtype=np.int32)
dauj = np.zeros((M + 2, segment_limit), dtype=np.int32)
cuoij = np.zeros((M + 2, segment_limit), dtype=np.int32)
cuoii = np.zeros((N + 2, segment_limit), dtype=np.int32)
Tsxw = np.zeros(shape)
Tsyw = np.zeros(shape)
H_moi = np.zeros(shape)
hi = np.zeros((2 * (M + N + 6),), dtype=np.float32)

# For Sediment Transport 
VTH = np.zeros(shape)
Qbx = np.zeros(shape)
Qby = np.zeros(shape)
FS = np.zeros(shape)
Kx = np.zeros(shape)
Ky = np.zeros(shape)
Fw = np.zeros(shape)
dH = np.zeros(shape)



ubt = np.zeros(M + 2)
ubp = np.zeros(M + 2)
vbt = np.zeros(N + 2)
vbd = np.zeros(N + 2)



