import numpy as np
import pycuda.driver as cuda
#import pycuda.autoinit
from pycuda.compiler import SourceModule

INT_SIZE = 4
DOUBLE_SIZE = 8

class Pointers:
    # store pointers in host and corresponding pointers to device
    def __init__(self,ctx, dtype=np.float32, **kwargs):
        self.arg_list = {}  # GPU pointers list
        self.host_ptrs = {}  # CPU pointers list
        self.device_only_ptrs = {}
        if dtype == np.float32:
            DOUBLE_SIZE = 4
        for key, value in kwargs.iteritems():            
            if type(value[0]) == np.ndarray and type(value[0, 0]) == np.float64:
                value = value.astype(dtype)
            if type(value[0]) == np.float64:
                value = value.astype(dtype)
            if type(value[0]) == np.ndarray and type(value[0, 0]) == np.int64:
                value = value.astype(np.int32)
            if type(value[0]) == np.int64:
                value = value.astype(np.int32)
            if type(value[0]) == int:
                value = value.astype(np.int32)

            self.host_ptrs[key] = value 
        self.ctx = ctx

    # allocate corresponding arrays from host to device
    def alloc(self):
        # allocate memory on device for pointers that that correspondance in host
        for key in self.host_ptrs.keys():
            host_ptr = self.host_ptrs[key]
            #print key, host_ptr.nbytes
            self.arg_list[key] = cuda.mem_alloc(host_ptr.nbytes)
        return self.arg_list

    def alloc_on_device_only(self, m, n):
        # allocate memory on device for pointers that does not have correspondance in host
        # Momory for Thomas Algorithm: x, Ap, Bp, ep, AA, BB, CC, DD
        # Problem 1 : in a spare topography map, allocating Ap,Bp, Ep according to m, n can be very costy
        # since amount of memmory need to be allocated is tremendous. (2*(max(m, n)^2) Need to read paper about this
        
        # DOUBLE_SIZE = DOUBLE_SIZE;
        array_size = 2 * m * n + 4 * max (m, n)
        self.device_only_ptrs['x'] = cuda.mem_alloc( DOUBLE_SIZE * array_size)
        self.device_only_ptrs['AA'] = cuda.mem_alloc(DOUBLE_SIZE * array_size)
        self.device_only_ptrs['BB'] = cuda.mem_alloc(DOUBLE_SIZE * array_size)
        self.device_only_ptrs['CC'] = cuda.mem_alloc(DOUBLE_SIZE * array_size)
        self.device_only_ptrs['DD'] = cuda.mem_alloc(DOUBLE_SIZE * array_size)
        self.device_only_ptrs['Ap'] = cuda.mem_alloc(DOUBLE_SIZE * array_size)
        self.device_only_ptrs['Bp'] = cuda.mem_alloc(DOUBLE_SIZE * array_size)
        self.device_only_ptrs['ep'] = cuda.mem_alloc(DOUBLE_SIZE * array_size)
        self.device_only_ptrs['SN'] = cuda.mem_alloc(INT_SIZE * 5 * max(m, n))


        # Auxilary memory for calculating coefficients for thomas algorithm f1 .. f5, a1, .. d2,
        array_size = m * n + 2 * max(m, n)

        self.device_only_ptrs['a1'] = cuda.mem_alloc(DOUBLE_SIZE * array_size)
        self.device_only_ptrs['b1'] = cuda.mem_alloc(DOUBLE_SIZE * array_size)
        self.device_only_ptrs['c1'] = cuda.mem_alloc(DOUBLE_SIZE * array_size)
        self.device_only_ptrs['d1'] = cuda.mem_alloc(DOUBLE_SIZE * array_size)
        self.device_only_ptrs['a2'] = cuda.mem_alloc(DOUBLE_SIZE * array_size)
        self.device_only_ptrs['c2'] = cuda.mem_alloc(DOUBLE_SIZE * array_size)
        self.device_only_ptrs['d2'] = cuda.mem_alloc(DOUBLE_SIZE * array_size)
        self.device_only_ptrs['f1'] = cuda.mem_alloc(DOUBLE_SIZE * array_size)
        self.device_only_ptrs['f2'] = cuda.mem_alloc(DOUBLE_SIZE * array_size)
        self.device_only_ptrs['f3'] = cuda.mem_alloc(DOUBLE_SIZE * array_size)
        self.device_only_ptrs['f5'] = cuda.mem_alloc(DOUBLE_SIZE * array_size)

        return self.device_only_ptrs
        #print "changed"


    def add_host_pts(self, **kwargs):
        for key, value in kwargs.iteritems():       
            if type(value[0]) == np.ndarray and type(value[0, 0]) == np.float64:
                value = value.astype(dtype)
            if type(value[0]) == np.float64:
                value = value.astype(dtype)
            if type(value[0]) == np.ndarray and type(value[0, 0]) == np.int64:
                value = value.astype(np.int32)
            if type(value[0]) == np.int64:
                value = value.astype(np.int32)
            if type(value[0]) == int:
                value = value.astype(np.int32)

            self.host_ptrs[key] = value 
            self.arg_list[key] = cuda.mem_alloc(value.nbytes)


    #auxilary memory to support Thomas Algorithm
    def add_device_only_pts(self, *args):
        for arg in args:
            self.device_only_ptrs.append(arg)

    # synchronize memory of host and device after each major change (i.e calculation)
    def update(self, keyList=None):
        if keyList == None:
            #print "key list is none"
            self.toDevice()
        else:
            for key in keyList:
                if key in self.host_ptrs:
                    source = self.host_ptrs[key]
                    des = self.arg_list[key]
                    cuda.memcpy_htod(des, source)
        self.ctx.synchronize()

    # list of host pts and arg list must be one-to-one correspondance
    # transfer data to GPU for calculation
    def toDevice(self, copyList=None):
        # assert length of the 2 list are same here
        if copyList == None:
            for key in self.arg_list.keys():
                source = self.host_ptrs[key]
                des = self.arg_list[key]
                # assert des is non zero here
                cuda.memcpy_htod(des, source)
        else:
            for key in copyList:
                source = self.host_ptrs[key]
                des = self.arg_list[key]
                # assert des is non zero here
                cuda.memcpy_htod(des, source)
        self.ctx.synchronize()

    # copy back results to host: retrieve one array or all array
    def toHost(self, keyList=None, all=False):
        if keyList == None :
            for key, value in self.host_ptrs.iteritems():
                des = value
                source = self.arg_list[key]
                cuda.memcpy_dtoh(des, source)
        else:
            for key in keyList:
                source = self.arg_list[key]
                des = self.host_ptrs[key]
                cuda.memcpy_dtoh(des, source)

        self.ctx.synchronize()

    #extract some particular array to validate
    def extract(self, argmap):
        for key, value in argmap.iteritems():
            if key in self.arg_list:
                source = self.arg_list[key]
                # print value
                #if key == "khouot": print source
                cuda.memcpy_dtoh(value, source)

    # extract on device only memory, this is for debugging Thomas algorithm
    def debug(self, keyMap):
        for key, des in keyMap.iteritems():
            source = self.device_only_ptrs[key]
            cuda.memcpy_dtoh(des, source)


class PointersStruct:
    arg_struct_size = 44 * np.intp(0).nbytes + 8;
    arr_struct_size = 20 * np.intp(0).nbytes;
    def __init__(self, ptrList, struct_ptr, structtype='ARG'):
        if structtype == 'ARG':
            cuda.memcpy_htod(int(struct_ptr), np.getbuffer(np.int32(ptrList[0])))
            #print np.int32(ptrList[0])
            cuda.memcpy_htod(int(struct_ptr) + 4, np.getbuffer(np.int32(ptrList[1])))
            struct_ptr = int (struct_ptr) + 8
            #print np.int32(ptrList[1])
            for value in ptrList[2:]:
                cuda.memcpy_htod(int(struct_ptr), np.getbuffer(np.intp(int(value))))
                struct_ptr = int (struct_ptr) + np.intp(0).nbytes
        else:
            for value in ptrList:
                cuda.memcpy_htod(int(struct_ptr), np.getbuffer(np.intp(int(value))))
                struct_ptr = int (struct_ptr) + np.intp(0).nbytes
