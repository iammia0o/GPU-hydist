'''
ULSAN NATIONAL INSTIUTE OF SCIENCE AND TECHNOLOGY
Copyright (c) 2019 HVCL lab
Created by Huong Nguyen

'''

import numpy as np
from math import * 
from Coeff import *
import time

class Variables(object):
	"""docstring for Variables"""
	def __init__(self, input_dir):
		segment_limit = 20
		# with open('DirsOfInputs_1.txt', 'r') as inf:
		#         Dirs = dict([line.split() for line in inf])

		self.boundary_type = np.zeros((4, segment_limit), dtype=np.int32)
		with open(input_dirs + 'boundary_type.txt', 'r') as inf:
			i = 0
			for line in inf:
				inp = line.strip()
				if inp != 'NA':
					self.boundary_type[i, :len(inp.split())] = [value for value in inp.split()]
				i += 1

		self.bc_up = np.transpose(np.loadtxt(input_dir + 'bientren.txt'))
		self.bc_down = np.transpose(np.loadtxt(input_dir + 'bienduoi.txt'))
		self.bc_left = np.transpose(np.loadtxt(input_dir + 'bientrai.txt'))
		self.bc_right = np.transpose(np.loadtxt(input_dir + 'bienphai.txt'))

		self.CC_u = np.transpose(np.loadtxt(input_dir + 'FS_bientren.txt'))
		self.CC_d = np.transpose(np.loadtxt(input_dir + 'FS_bienduoi.txt'))
		self.CC_l = np.transpose(np.loadtxt(input_dir + 'FS_bientrai.txt'))
		self.CC_r = np.transpose(np.loadtxt(input_dir + 'FS_bienphai.txt'))



		blist = [bc_up, bc_down, bc_left, bc_right]
		# print blist[ np.where(boundary_type != 0) [0] [0] ]
		total_time = len(blist[ np.where(boundary_type != 0) [0] [0] ] ) / (np.count_nonzero(boundary_type[np.where(boundary_type != 0)[0][0]]))
		print total_time

		self.bienQ = np.zeros((4,), dtype=np.int32)
		print self.bienQ
		#boundary_type:
		#0: bien ran
		#1: bien Z
		#2: bien Q
		#1 va 2 deu tuong ung voi bien long, 
		#tuy nhien tuy vao dk thuc the do dac ma co the la bien muc nuoc hoac la bien luu luong
		# danh dau bienQ de xac dinh xem bien la bien luu luong hay khong, neu la bien luu luong
		# thi se interprete dieu kien bien khac di
		self.bienQ[np.where(boundary_type == 2)[0]] = 1

		h = np.loadtxt(input_dir + 'bandodosau.txt')
		h = np.transpose(h)
		N, M = h.shape
		h = np.flip(h, 1)
		h = np.pad(h, ((1, 2), (1, 2)), 'edge')
		h = h + NANGDAY
		self.h = h
		hsnham = np.loadtxt(input_dir + 'bandohsnham.txt')
		hsnham = np.transpose(hsnham)
		hsnham = np.flip(hsnham, 1)
		hsnham = np.pad(hsnham, ((1, 2), (1, 2)), 'edge')
		self.hsnham = hsnham
		# VISCOINDX: hsnhot
		VISCOINDX = np.loadtxt(input_dir + 'bandohsnhotroiA.txt')
		VISCOINDX = np.transpose(VISCOINDX)
		VISCOINDX = np.flip(VISCOINDX, 1)
		VISCOINDX = np.pad(VISCOINDX, ((1, 2), (1, 2)), 'edge')
		self.VISCOINDX = VISCOINDX

		# hs ma sat day
		Fw = np.loadtxt(input_dir + 'bandoFw.txt')
		Fw = np.transpose(Fw)
		Fw = np.flip(Fw, 1)
		Fw = np.pad(Fw, ((1, 2), (1, 2)), 'edge')
		self.Fw = Fw

		shape = (N + 3, M + 3)
		self.khouot = np.ones(shape).astype(np.int32) * 2
		self.u = np.zeros(shape)
		self.v = np.zeros(shape)
		self.z = np.zeros(shape)
		self.t_u = np.zeros(shape)
		self.t_v = np.zeros(shape)
		self.t_z = np.zeros(shape)
		self.Kx1 = np.zeros(shape)
		self.Ky1 = np.zeros(shape)
		self.Htdu = np.zeros(shape)
		self.Htdv = np.zeros(shape)
		self.htaiz = np.zeros(shape)
		self.moci = np.zeros(N + 2, dtype=np.int32)
		self.mocj = np.zeros(M + 2, dtype=np.int32)
		self.daui = np.zeros((N + 2, segment_limit), dtype=np.int32)
		self.dauj = np.zeros((M + 2, segment_limit), dtype=np.int32)
		self.cuoij = np.zeros((M + 2, segment_limit), dtype=np.int32)
		self.cuoii = np.zeros((N + 2, segment_limit), dtype=np.int32)
		self.Tsxw = np.zeros(shape)
		self.Tsyw = np.zeros(shape)
		self.H_moi = np.zeros(shape)
		self.hi = np.zeros((2 * (M + N + 6),), dtype=np.float32)

		# For Sediment Transport 
		self.VTH = np.zeros(shape)
		self.Qbx = np.zeros(shape)
		self.Qby = np.zeros(shape)
		self.FS = np.zeros(shape)
		self.tFS = np.zeros(shape)
		self.Kx = np.zeros(shape)
		self.Ky = np.zeros(shape)
		self.dH = np.zeros(shape) 


		self.ubt = np.zeros(M + 2)
		self.ubp = np.zeros(M + 2)
		self.vbt = np.zeros(N + 2)
		self.vbd = np.zeros(N + 2)
