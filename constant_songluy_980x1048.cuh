#ifndef CONSTANT_CUH__
#define CONSTANT_CUH__
#define PI 3.14159265359
#define DOUBLE double

struct Argument_Pointers{
    int M, N;
    int *bienQ;
    int* daui, *dauj, *cuoii, *cuoij, *moci, *mocj, *khouot, *boundary_type;
    DOUBLE* h,*v, *u, *z, *t_u, *t_v, *t_z, *Htdu, *Htdv, *H_moi, *htaiz;
    DOUBLE* ubt, *ubp, *vbt, *vbd; //*vt, *ut;
    DOUBLE* hsnham, *VISCOIDX, *Kx1, *Ky1, *Tsyw, *Tsxw;
    DOUBLE* bc_up, *bc_down, *bc_left, *bc_right;
    DOUBLE* hi;

    DOUBLE* FS,  *CC_u, *CC_d, *CC_l, *CC_r;
    DOUBLE *VTH, *Kx, *Ky, *Fw;
    DOUBLE* Qbx, *Qby;
    DOUBLE* dH;
};

struct Array_Pointers{
    DOUBLE* a1, *b1, *c1, *d1, *a2, *c2, *d2;
    DOUBLE* f1, *f2, *f3, *f5;
    DOUBLE* AA, *BB, *CC, *DD;
    DOUBLE* x;
    DOUBLE* Ap, *Bp, *ep;
    int *SN;
};

__constant__ int segment_limit = 10;
__constant__ DOUBLE dX = 1.25;
__constant__ DOUBLE dY = 1.25;
__constant__ DOUBLE dT = 0.125;

__constant__ DOUBLE dXbp = 1.25 * 1.25;
__constant__ DOUBLE dX2 = 2 * 1.25;
    
__constant__ DOUBLE dYbp = 1.25 * 1.25;
__constant__ DOUBLE dY2 = 2 * 1.25;
    
__constant__ DOUBLE dTchia2dX = 0.12500 / (2 * 1.25);
__constant__ DOUBLE dTchia2dY = 0.12500 / (2 * 1.25);
    
__constant__ DOUBLE QuyDoiTime = 1.0 / 3600;
__constant__ DOUBLE QuyDoiPi = 1.0 / PI;
__constant__ DOUBLE HaiChiadT = 2.0 / 0.125;

// __constant__ DOUBLE dX = 2.5;
// __constant__ DOUBLE dY = 2.5;
// __constant__ DOUBLE dT = 0.25;

// __constant__ DOUBLE dXbp = 2.5 * 2.5;
// __constant__ DOUBLE dX2 = 2 * 2.5;
    
// __constant__ DOUBLE dYbp = 2.5 * 2.5;
// __constant__ DOUBLE dY2 = 2 * 2.5;
    
// __constant__ DOUBLE dTchia2dX = 0.2500 / (2 * 2.5);
// __constant__ DOUBLE dTchia2dY = 0.2500 / (2 * 2.5);
    
// __constant__ DOUBLE QuyDoiTime = 1.0 / 3600;
// __constant__ DOUBLE QuyDoiPi = 1.0 / PI;
// __constant__ DOUBLE HaiChiadT = 2.0 / 0.25;


__constant__ DOUBLE kenhhepng = 0;
__constant__ DOUBLE kenhhepd = 0;
    
// gioi han tinh
__constant__ DOUBLE NANGDAY = 0.0; // thong so nang day
__constant__ DOUBLE H_TINH = 0.03; // do sau gioi han (m)
    
// thong so ve gio
__constant__ DOUBLE Wind = 0.0;  // van toc gio (m/s)
__constant__ DOUBLE huonggio = 0.0;  // huong gio (degree)
        
// Zban dau
__constant__ DOUBLE Zbandau = 0.0;
    
// He so lam tron va he so mu manning
__constant__ DOUBLE heso = 0.96; //1.0;
__constant__ DOUBLE mu_mn = 0.2;
    
// ND number (kg/m3)
__constant__ DOUBLE NDnen = 0.02;
__constant__ DOUBLE NDbphai = 0.5;
__constant__ DOUBLE NDbtrai = 0.5;
__constant__ DOUBLE NDbtren = 0.5;
__constant__ DOUBLE NDbduoi = 0.5;

// tod(toi han boi), toe(toi han xoi)
// hstoe (he so tinh ung suat tiep toi han xoi theo do sau)
// ghtoe (gioi han do sau tinh toe(m))
// Mbochat (kha nang boc hat M(kg/m2/s))
__constant__ DOUBLE Tod = 1;
__constant__ DOUBLE Toe = 1;
__constant__ DOUBLE hstoe = 0;
__constant__ DOUBLE ghtoe = 3;
__constant__ DOUBLE Mbochat = 0.0001;


// khoi luong rieng cua nuoc (ro) va khoi luong rieng cua hat (ros) (kg/m3)
__constant__ DOUBLE ro = 1000;
__constant__ DOUBLE ros = 2690;

// duong kinh trung binh cua hat 50% (m) (dm)
__constant__ DOUBLE dm = 0.0002;
// duong kinh hat trung binh 90% (m) 
__constant__ DOUBLE d90 = 0.002;

// he so nhot dong hoc cua nuoc sach
__constant__ DOUBLE muy = 1.01e-06;

// Do rong cua hat (Dorong) va Ty trong (KLR cua hat va nuoc) (Sx)
__constant__ DOUBLE Dorong = 0.443;
__constant__ DOUBLE Sx = 2.69;
    
//tong so do sau de tinh he so nham
__constant__ DOUBLE sohn = 8;
//tong so do sau de tinh he so nhot
__constant__ DOUBLE soha = 3;  
// tong so do sau de tinh Fw
__constant__ DOUBLE sohn1 = 3;  



    
//luc coriolis
__constant__ DOUBLE  CORIOLIS_FORCE = 0.0;
__constant__ DOUBLE g = 9.81; //gia toc trong truong = 9.81 m2/s

//lan truyen
__constant__ DOUBLE Ks = 2.5 * 0.0002; // = 2.5 * dm
// hoac bang Ks = 3 * d 90% 
    
//Boixoi
//__constant__ DOUBLE Ufr = 0.25 * pow((Sx - 1) * g, (8.0 / 15.0)) * pow(dm, (9.0 / 15.0)) * pow(muy, (-1.0 / 15.0));
//__constant__ DOUBLE Dxr = dm * pow(g * (Sx - 1) / muy * muy,(1 / 3));
//__constant__ DOUBLE wss = 10 * muy / dm * (pow(1.0 + (0.01 * (Sx - 1) * 9.81 * pow(dm, 3) / pow(muy, 2.0)), 0.5) - 1.0); // cong thuc nay phu thuoc vao duong kinh hat

__device__ DOUBLE Windx() {return  0.0013 * (0.00075 + 0.000067 * abs(Wind)) * abs(Wind) * Wind * cos(huonggio * (PI / 180));}
__device__ DOUBLE Windy() {return  0.0013 * (0.00075 + 0.000067 * abs(Wind)) * abs(Wind) * Wind * sin(huonggio * (PI / 180));}

__device__ DOUBLE Ufr(){ return 0.25 * pow ((Sx - 1) * g, 8.0 / 15.0) * pow(dm , 9.0 / 15.0) * pow(muy , -1.0 / 15.0);}
__device__ DOUBLE Dxr(){ return dm * pow(g * (Sx - 1) / muy * muy, (1.0 / 3.0));}
__device__ DOUBLE wss(){ 
    return 10 * muy / dm * ( sqrt(1 + 0.01 * (Sx - 1) * 9.81 * pow(dm , 3) / pow(muy, 2) ) - 1);
}

__device__ int locate_segment_v(int N, int M, bool* bienran1, bool* bienran2, int* first, int* last, int row, int col,  int* daui, int* cuoii, int* moci, DOUBLE* h){
    
    for (int k = 0; k < moci[row]; k++){
        int width = segment_limit;
        if ((daui[row * width +  k] <= col) && (col <= cuoii[row * width + k])) 
        {
            *first = daui[row * width + k];
            *last = cuoii[row * width + k];
            //printf("thread: %d A: dau: %d, cuoi: %d\n", threadIdx.x, *first, *last);
            //printf("first %d\n", *first);
            
            width = M + 3;
            if ((*first > 2) || ((*first == 2) && ((h[row * width + *first - 1] + h[(row - 1) * width + *first - 1]) * 0.5 == NANGDAY))) 
               *bienran1 = true;
            if ((*last < M) || ( (*last == M) && ((h[row * width +  *last] + h[(row - 1) * width + *last]) * 0.5 == NANGDAY) ) )
               *bienran2 = true;
            return k;
        }
    }
}

__device__ int locate_segment_u(int N, int M, bool* bienran1, bool* bienran2, int* first, int* last, int row, int col,  int* dauj, int* cuoij, int* mocj, DOUBLE* h){
    
    for (int k = 0; k < mocj[col]; k++){
        int width = segment_limit;
        if ((dauj[col * width +  k] <= row) && (row <= cuoij[col * width + k])) 
        {
            *first = dauj[col * width +  k];
            *last = cuoij[col * width + k];
    
            width = M + 3;
    
            if ((*first > 2) || ( (*first == 2) && ((h[1 * width + col] + h[1 * width + col - 1]) * 0.5 == NANGDAY )) ){
                *bienran1 = true;
                
            }
    
            if ((*last < N) || ((*last == N) && ((h[N * width + col] + h[N * width + col - 1]) * 0.5 == NANGDAY)))
                *bienran2 = true;
        return k;
        }
    }

}
#endif
