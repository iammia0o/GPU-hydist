from __future__ import division
import numpy as np

#so dong va so cot cua mien tinh
# dau = l, cuoi = r


# thong so mien tinh
#dX = 0.25  #thong so dx (m) 
#dY = 0.25 #thong so dy (m)
#dT = 5   #thong so dt (s)
dX = 5.0
dY = 5.0
dT = 0.5

kenhhepd = 0 
kenhhepng = 0
# gioi han tinh
NANGDAY = 0 # thong so nang day
H_TINH = 0.03  # do sau gioi han (m)
    
# thong so ve gio
Wind = 0     # van toc gio (m/s)
huonggio = 0  # huong gio (degree)
        
# Zban dau
Zbandau = 0
    
# He so lam tron va he so mu manning
heso = 1
mu_mn = 0.2
    
# ND number (kg/m3)
NDnen = 0.02
NDbphai = 0.5
NDbtrai = 0.5
NDbtren = 0.5
NDbduoi = 0.5

# tod(toi han boi), toe(toi han xoi)
# hstoe (he so tinh ung suat tiep toi han xoi theo do sau)
# ghtoe (gioi han do sau tinh toe(m))
# Mbochat (kha nang boc hat M(kg/m2/s))
tod = 1
toe = 1
hstoe = 0
ghtoe = 3
Mbochat = 0.0001

# khoi luong rieng cua nuoc (ro) va khoi luong rieng cua hat (ros) (kg/m3)
ro = 1000
ros = 2690

# duong kinh trung binh cua hat 50% (m) (dm)
dm = 0.0002
# duong kinh hat trung binh 90% (m) 
d90 = 0.002

# he so nhot dong hoc cua nuoc sach
muy = 1.01e-06

# Do rong cua hat (dorong) va Ty trong (KLR cua hat va nuoc) (Sx)
dorong = 0.443
Sx = 2.69
    
#tong so do sau de tinh he so nham
sohn = 8  
#tong so do sau de tinh he so nhot
soha = 3  
# tong so do sau de tinh Fw
sohn1 = 3  

Windx = 0.0013 * (0.00075 + 0.000067 * np.absolute(Wind)) * np.absolute(Wind) * Wind * np.cos(huonggio * (np.pi / 180))
Windy = 0.0013 * (0.00075 + 0.000067 * np.absolute(Wind)) * np.absolute(Wind) * Wind * np.sin(huonggio * (np.pi / 180))
    
#luc coriolis
f = 0
g = 9.81 #gia toc trong truong = 9.81 m2/s

#lan truyen
Ks = 2.5 * dm       
# hoac bang Ks = 3 * d 90% 
    
#Boixoi
Ufr = 0.25 * pow((Sx - 1) * g, (8/15)) * pow(dm, (9 / 15)) * pow(muy, (-1/15))
Dxr = dm * pow(g * (Sx - 1) / muy * muy,(1 / 3))
wss = 10 * muy / dm * (pow(1 + (0.01 * (Sx - 1) * 9.81 * pow(dm, 3) / pow(muy, 2)), 0.5) - 1) # cong thuc nay phu thuoc vao duong kinh hat

#Dung de toi uu hoa phan tinh toan cua chuong trinh
dXbp = dX * dX
dX2 = 2 * dX
    
dYbp = dY * dY
dY2 = 2 * dY
    
dTchia2dX = dT / (2 * dX)
dTchia2dY = dT / (2 * dY)
    
QuyDoiTime = 1 / 3600
QuyDoiPi = 1 / np.pi
HaiChiadT = 2 / dT
