#include "constant.cuh"
#include "cuda_runtime.h"

#define DOUBLE float

// now need to solve problem with 2d array. 

// suppelmentary functions that need to be solved first:
// this is optimazation. Will do after testing all the main calculation in solver is done
// Tinh KhoUot
// Find Calculation Limit
// Htuongdoi
// GiatriHtaiZ
// VTHZ()
// K_factor()

__device__ int calculate_index(int M){
    int thrx = blockIdx.x * blockDim.x + threadIdx.x;
    int thry = blockIdx.y * blockDim.y + threadIdx.y;
    return thrx * (blockDim.y * gridDim.y) + thry;
}

//-----------------------------------Reynold Eq Solver--------------------------------------------
__device__ void tridiag(int sn, DOUBLE* AA, DOUBLE* BB, DOUBLE* CC, DOUBLE*DD, DOUBLE *x, 
    DOUBLE *Ap, DOUBLE *Bp, DOUBLE *ep){
    Ap[1] = - CC[1] / BB[1];
    Bp[1] = DD[1] / BB[1];
    //printf("hello from Thomas, sn = % d\n", sn);
    for (int i = 2; i < sn; i++){
        //printf("i = %d AA: %f , BB: %f , CC: %f , DD: % f\n", i, AA[i], BB[i], CC[i], DD[i] );
        ep[i] = AA[i] * Ap[i - 1] + BB[i];
        Ap[i] = -CC[i] / ep[i];
        Bp[i] = (DD[i] - (AA[i] * Bp[i - 1])) / ep[i];
    }

    x[sn] = (DD[sn] - (AA[sn] * Bp[sn - 1])) / (BB[sn] + (AA[sn] * Ap[sn - 1]));

    for (int i = sn - 1; i > 0; i--){
        x[i] = Bp[i] + (Ap[i] * x[i + 1]);
        //printf("thread: %d , i: %d x: %.15f \n", (blockIdx.x*blockDim.x + threadIdx.x) + 3, i, x[i]);
    }
}

__device__ void bienrandau(int first, int last, DOUBLE* AA, DOUBLE* BB, DOUBLE* CC, DOUBLE*DD,
    DOUBLE *a1, DOUBLE *b1, DOUBLE *c1, DOUBLE *d1, DOUBLE *a2, DOUBLE *b2, DOUBLE *c2, DOUBLE *d2){

    //printf("bienrandau is called \n");
    for (int i = first; i <= last; i++){
        AA[(i - first) * 2 + 1] = a2[i];
        BB[(i - first) * 2 + 1] = 1;
        CC[(i - first) * 2 + 1] = c2[i];
        DD[(i - first) * 2 + 1] = d2[i];
     
        AA[(i - first + 1) * 2] = a1[i];
        BB[(i - first + 1) * 2] = b1[i];
        CC[(i - first + 1) * 2] = c1[i];
        DD[(i - first + 1) * 2] = d1[i];
    }
}


__device__ void bienlongdau(int first, int last, DOUBLE* AA, DOUBLE* BB, DOUBLE* CC, DOUBLE*DD,
    DOUBLE *a1, DOUBLE *b1, DOUBLE *c1, DOUBLE *d1, DOUBLE *a2, DOUBLE *b2, DOUBLE *c2, DOUBLE *d2){

    //printf("bienlongdau is called \nFirst = %d , last = %d\n", first, last);
    for (int i = first; i < last; i++){
        AA[(i - first) * 2 + 1] = a1[i];
        BB[(i - first) * 2 + 1] = b1[i];  
        CC[(i - first) * 2 + 1] = c1[i];
        DD[(i - first) * 2 + 1] = d1[i];

        AA[(i - first + 1) * 2] = a2[i + 1];
        BB[(i - first + 1) * 2] = 1;
        CC[(i - first + 1) * 2] = c2[i + 1];
        DD[(i - first + 1) * 2] = d2[i + 1];
    }

}

__device__ void update_abcd_at_l(int first, int last, bool bienran1, bool bienran2, DOUBLE *a1, DOUBLE *b1, DOUBLE *c1, DOUBLE *d1,
    DOUBLE* f1, DOUBLE* f2, DOUBLE* f3, DOUBLE f4, DOUBLE* f5, DOUBLE* f6, DOUBLE* f7, DOUBLE* f8){
    if (last - first > 1){
        if (bienran1){
            a1[first] = f4;
            b1[first] = f2[first] - (f3[first] * f7[first+ 1] / f6[first+ 1]);
            c1[first] = - f4 - (f3[first] / f6[first+ 1]);
            d1[first] = f5[first] - (f3[first] * f8[first+ 1] / f6[first+ 1]);
        }
            
        else{
            a1[first] = f4 - f1[first] / f7[first];
            b1[first] = f2[first] - (f1[first] * f6[first] / f7[first]) - (f3[first] * f7[first+ 1] / f6[first+ 1]);
            c1[first] = -f4 - (f3[first] / f6[first+ 1]);
            d1[first] = f5[first] - (f1[first] * f8[first] / f7[first]) - (f3[first] * f8[first+ 1] / f6[first+ 1]);
        }
        if (last - first > 2){
            for(int i = first + 1; i < last - 1; i++){
                a1[i] = f4 - f1[i] / f7[i];
                b1[i] = f2[i] - (f1[i] * f6[i] / f7[i]) - (f3[i] * f7[i + 1] / f6[i + 1]);
                c1[i] = -f4 - (f3[i] / f6[i + 1]);
                d1[i] = f5[i] - (f1[i] * f8[i] / f7[i]) - (f3[i] * f8[i + 1] / f6[i + 1]);
            }
        }

        if (bienran2){
            a1[last - 1] = f4 - f1[last -1] / f7[last -1];
            b1[last - 1] = f2[last - 1] - (f1[last - 1] * f6[last -1] / f7[last - 1]);
            c1[last - 1] = - f4;
            d1[last - 1] = f5[last - 1] - (f1[last - 1] * f8[last - 1] / f7[last - 1]);
        }
        else{
            a1[last - 1] = f4 - f1[last - 1] / f7[last - 1];
            b1[last - 1] = f2[last - 1] - (f1[last - 1] * f6[last - 1] / f7[last - 1]) - (f3[last - 1] * f7[last] / f6[last]);
            c1[last - 1] = -f4 - (f3[last - 1] / f6[last]);
            d1[last - 1] = f5[last - 1] - (f1[last - 1] * f8[last - 1] / f7[last - 1]) - (f3[last - 1] * f8[last] / f6[last]);
        }
    }
    else{
        //printf("Overflow, comparision btw DOUBLE and int\n");
        a1[first] = f4;
        b1[first] = f2[first];
        d1[first] = f5[first];
        c1[first] = -f4;
    }
}



__device__ int boundary_config(bool isU, int first, int last, bool bienran1, 
    bool bienran2, DOUBLE ubp_or_vbt, DOUBLE ubt_or_vbd, DOUBLE TZ_r, DOUBLE TZ_l, bool dkBienQ_1, bool dkBienQ_2, int dkfr,
    DOUBLE *a1, DOUBLE *b1, DOUBLE *c1, DOUBLE *d1, DOUBLE *a2, DOUBLE *b2, DOUBLE *c2, DOUBLE *d2, DOUBLE* AA, DOUBLE* BB, DOUBLE* CC, DOUBLE* DD){
    int sn;
    bool isBienran;
    if (bienran1){
        // attention
        //printf("gpu bien: %d, %d\n", dkBienQ_2, dkBienQ_1);
        bienrandau(first, last, AA, BB, CC, DD, a1, b1, c1, d1, a2, b2, c2, d2);
        DD[1] = d2[first];
        // ran - long
        if (bienran2 == false){
            if ((dkBienQ_2) && (last == dkfr)){         //r == dkfr:   // Kiem tra lai phan nay
                sn = 2 * (last - first) + 1;
                // attention 
                AA[sn] = a2[last];
                BB[sn] = 1;
                DD[sn] = d2[last] - c2[last] * ubp_or_vbt;
            }
            else{
                //printf("ran - long \n");
                sn = 2 * (last - first) ;
                AA[sn] = a1[last - 1];
                BB[sn] = b1[last - 1];
                DD[sn] = d1[last - 1] - c1[last - 1] * TZ_r;
            }
        }
        // ran - ran
        else{
            //printf("ran - ran \n");
            sn = 2 * (last - first) + 1;
            AA[sn] = a2[last];
            BB[sn] = 1;
            DD[sn] = d2[last];
        }
    }
    // long
    else{
        if ((dkBienQ_1) && (first == 2)){
            bienrandau(first, last, AA, BB, CC, DD, a1, b1, c1, d1, a2, b2, c2, d2);
            DD[1] = d2[first] - a2[first] * ubt_or_vbd;
            // thieu bb[1] va cc[1] cho truong hop vz, hoi lai co
            isBienran = true;
        }
        else{
            bienlongdau(first, last, AA, BB, CC, DD, a1, b1, c1, d1, a2, b2, c2, d2);
            BB[1] = b1[first];
            CC[1] = c1[first];
            DD[1] = d1[first] - a1[first] * TZ_l;
            isBienran = false;
        }
        // long - long
        if (bienran2 == false){ // variable isbienran is equivalent with variable text in original code
            if ((dkBienQ_1) && (last == dkfr)){     //r == dkfr: // BienQ[0] && r == M trong truong hop giaianv
                sn = 2 * (last - first);
                if (isBienran)
                    sn += 1;
                AA[sn] = a2[last];
                BB[sn] = 1;
                DD[sn] = d2[last] - c2[last] * ubp_or_vbt;
            }
            else{
                sn = 2 * (last - first);

                if (!isBienran)
                    sn -= 1;
                AA[sn] = a1[last - 1];
                BB[sn] = b1[last - 1];
                DD[sn] = d1[last - 1] - c1[last - 1] * TZ_r;
            }
        }
        else{
            sn = 2 * (last - first);
            if (isBienran)
                sn += 1;
            // AA[sn] = f7[last] 
            // this line is modified for the canal case
            AA[sn] = a2[last];
            BB[sn] = 1;
            DD[sn] = d2[last];
        }
    }
   
    return sn;

}

// missing: dXbp dT dTchia2dX g dYbp Windx() Tsxw DX2

// if change to  2d: need to change all parameter, 2d array fix element access
// if use 1d: need to change element access, which involve offset
__device__ void uzSolver(int offset, int N, int M, int first, int last, int jpos, bool bienran1, bool bienran2, 
    DOUBLE dXbp, DOUBLE dYbp, DOUBLE dT, DOUBLE dTchia2dX, DOUBLE dTchia2dY, DOUBLE* Tsxw, // constants
    DOUBLE* v, DOUBLE* u, DOUBLE* z, DOUBLE* Htdu, DOUBLE* Htdv, DOUBLE* VISCOIDX, DOUBLE* t_u, DOUBLE* t_z, 
    DOUBLE* ubt, DOUBLE* ubp, DOUBLE* H_moi, DOUBLE* Kx1, bool* bienQ,
    DOUBLE *f1, DOUBLE *f2, DOUBLE *f3, DOUBLE *f5, DOUBLE *f6, DOUBLE *f7, DOUBLE *f8,
    DOUBLE *a1, DOUBLE* b1, DOUBLE *c1, DOUBLE *d1, DOUBLE *a2, DOUBLE *b2, DOUBLE *c2, DOUBLE *d2,
    DOUBLE *AA, DOUBLE *BB, DOUBLE *CC, DOUBLE *DD, DOUBLE *Ap, DOUBLE *Bp, DOUBLE* ep, DOUBLE* x){
    int j = jpos;
    DOUBLE f4 = 2 * g * dTchia2dX;
    DOUBLE p = 0.0;
    DOUBLE q = 0.0;
    // memory offset for 2d arrays:
    // 2d arrays in use: v, u, VISCOIDX, Htdu, Htdv, Kx1, Ky1, H_moi,  Tsxw, t_u, t_z 
    // same offset: u, v, Htdu, Htdv, t_u, t_z, Hmoi, Tsxw : N + 3
    // diffent offset: VISCOIDX, Kx1, Ky1, : MUST change to make these has same offset

    for (int i = first; i < last; i++){
        DOUBLE vtb = (v[i * offset +  j - 1] + v[i * offset + j] + v[(i + 1) * offset + j - 1] + v[(i + 1) * offset + j]) * 0.25;
        f1[i] = dTchia2dX * u[i * offset + j] + VISCOIDX[i * offset + j] * dT / dXbp;
        f2[i] = -(2 + Kx1[i * offset + j] * dT * sqrt(u[i * offset + j] * u[i * offset + j] + vtb * vtb) / Htdu[i * offset + j] + (2 * dT * VISCOIDX[i * offset + j]) / dXbp); // chua tinh muc nuoc trung binh
        f3[i] = dT * VISCOIDX[i * offset + j] / dXbp - dTchia2dX * u[i * offset + j];
    

        if (H_moi[i * offset + j - 1] <= H_TINH){
            if (vtb < 0){
                p = vtb * (-3 * u[i * offset + j] + 4 * u[i * offset + j + 1] - u[i * offset + j + 2]) / dY2;
                q = (u[i * offset + j] - 2 * u[i * offset + j + 1] + u[i * offset + j + 2]) / dYbp;
            }
        }
        else{
            if (H_moi[i * offset + j + 1] <= H_TINH){
                if ((H_moi[i * offset + j - 2] > H_TINH) && (vtb > 0)){
                    p = vtb * (3 * u[i * offset + j] - 4 * u[i * offset + j - 1] + u[i * offset + j - 2]) / dY2;
                    q = (u[i * offset + j] - 2 * u[i * offset + j - 1] + u[i * offset + j - 2] ) / dYbp;
                }
            }else{
                p = vtb * (u[i * offset + j + 1] - u[i * offset + j - 1]) / dY2;
                q = (u[i * offset + j + 1] - 2 * u[i * offset + j] + u[i * offset + j - 1]) / dYbp;
            }
        }

        f5[i] = -2 * u[i * offset + j] + dT * p  - dT * CORIOLIS_FORCE * vtb - dT * VISCOIDX[i * offset + j] * q - dT * (Windx() - Tsxw[i * offset + j]) / Htdu[i * offset + j];
    }

    for (int i = first; i <= last; i++){
        f6[i] = dTchia2dX * Htdu[i * offset + j];
        f7[i] = - dTchia2dX * Htdu[(i - 1) * offset + j];
        //printf("Htdu[%d, %d]: %f\n", i - 1, j, Htdu[(i - 1) * offset + j]);
        f8[i] = z[i * offset + j] - dTchia2dY * (Htdv[i * offset + j] * v[i * offset + j] - Htdv[i * offset + j - 1] * v[i * offset + j - 1]);
    
        a2[i] = f7[i];
        c2[i] = f6[i];
        d2[i] = f8[i];
    }

    update_abcd_at_l(first, last, bienran1, bienran2, a1, b1, c1, d1, f1, f2, f3, f4, f5, f6, f7, f8);


    int sn = boundary_config(true, first, last, bienran1, bienran2, 
            ubp[j], ubt[j], t_z[last * offset + j], t_z[first * offset + j], bienQ[2], bienQ[3], N, a1, b1, c1, d1, a2, b2, c2, d2, AA, BB, CC, DD);

    if (sn > 0)
        tridiag(sn, AA, BB, CC, DD, x, Ap, Bp, ep);

    if (bienran1){
        for (int i = first; i < last; i++) {
            t_z[i * offset + j] = x[2 * (i - first) + 1];
            t_u[i * offset + j] = x[2 * (i - first) + 2];
        }
        t_u[(first - 1) * offset + j] = 0;
    }else{
        if ((bienQ[2]) && (first == 2)){
            for (int i = first; i < last; i++){
                t_z[i * offset + j] = x[2 * (i - first) + 1];
                t_u[i * offset + j] = x[2 * (i - first) + 2];
            }
            t_u[(first - 1) * offset + j] = ubt[j];
        }
        else{
       
            t_u[first * offset + j] = x[1];
            t_u[(first - 1) * offset + j] = (d2[first] - t_z[first * offset + j] - c2[first] * t_u[first * offset + j]) / a2[first];
            for (int i  = first + 1; i < last; i ++){
                t_z[i * offset + j] = x[2 * (i - first)];
                t_u[i * offset + j] = x[2 * (i - first) + 1];
            }
        }
    }       

    if (bienran2){
        t_u[last * offset + j] = 0;
        t_z[last * offset + j] = x[sn];
    }
    else{
        if ((bienQ[3]) && (last == N)){
            t_u[last * offset + j] = ubp[j];
            t_z[last * offset + j] = x[sn];
        }
        else
            //print "long z2"
            t_u[last * offset + j] = (d2[last] - a2[last] * t_u[(last - 1) * offset + j] - t_z[last * offset + j]) / c2[last];
    }
}

__device__ void vzSolver(int offset, int N, int M, int first, int last, int ipos, bool bienran1, bool bienran2, 
    DOUBLE dXbp, DOUBLE dYbp, DOUBLE dT, DOUBLE dTchia2dX, DOUBLE dTchia2dY, DOUBLE* Tsyw, // constants
    DOUBLE* v, DOUBLE* u, DOUBLE* z, DOUBLE* Htdu, DOUBLE* Htdv, DOUBLE* VISCOIDX, DOUBLE* t_v, DOUBLE* t_z, 
    DOUBLE* vbt, DOUBLE* vbd, DOUBLE* H_moi, DOUBLE* Ky1, bool* bienQ,
    DOUBLE *f1, DOUBLE *f2, DOUBLE *f3, DOUBLE *f5, DOUBLE *f6, DOUBLE *f7, DOUBLE *f8,
    DOUBLE *a1, DOUBLE* b1, DOUBLE *c1, DOUBLE *d1, DOUBLE *a2, DOUBLE *b2, DOUBLE *c2, DOUBLE *d2,
    DOUBLE *AA, DOUBLE *BB, DOUBLE *CC, DOUBLE *DD, DOUBLE *Ap, DOUBLE *Bp, DOUBLE* ep, DOUBLE* x){
    int i = ipos;
    DOUBLE f4 = 2 * g * dTchia2dY;
    DOUBLE p = 0.0;
    DOUBLE q = 0.0;

  

    for (int j = first; j < last; j++){
        DOUBLE utb = (u[(i - 1) * offset + j] + u[i * offset + j] + u[(i - 1) * offset + j + 1] + u[i * offset + j + 1]) * 0.25;
        f1[j] = dTchia2dY * v[i * offset + j] + VISCOIDX[i * offset + j] * dT / dYbp;
        f2[j] = -(2 + Ky1[i * offset + j] * dT * sqrt(v[i * offset + j] * v[i * offset + j] + utb * utb) / Htdv[i * offset + j] + (2 * dT * VISCOIDX[i * offset + j]) / dYbp);
        f3[j] = dT * VISCOIDX[i * offset + j] / dYbp - dTchia2dY * v[i * offset + j];

        if (H_moi[(i - 1) * offset + j] <= H_TINH){
            if (utb < 0){
                q = utb * (-3 * v[i * offset + j] + 4 * v[(i + 1) * offset + j] - v[(i + 2) * offset + j]) / dX2;
                p = (v[i * offset + j] - 2 * v[(i + 1) * offset + j] + v[(i + 2) * offset + j] ) / dXbp;
            }
        }else{
            if (H_moi[(i + 1) * offset + j] <= H_TINH){
                if ((H_moi[(i - 2) * offset + j] > H_TINH) && (utb > 0)){
                    q = utb * (3 * v[i * offset + j] - 4 * v[(i - 1) * offset + j] + v[(i - 2) * offset + j]) / dX2;
                    p = (v[i * offset + j] - 2 * v[(i - 1) * offset + j] + v[(i - 2) * offset + j] ) / dXbp;
                }
            }else{
                q = utb * (v[(i + 1) * offset + j] - v[(i - 1) * offset + j]) / dX2;
                p = (v[(i + 1) * offset + j] - 2 * v[i * offset + j] + v[(i - 1) * offset + j]) / dXbp;
            }
        }
        f5[j] = -2 * v[i * offset + j] + dT * q + dT * CORIOLIS_FORCE * utb - dT * VISCOIDX[i * offset + j] * p - dT * (Windy() - Tsyw[i * offset + j]) / Htdv[i * offset + j];
    }


    for (int j = first; j <= last; j++){
        f6[j] = dTchia2dY * Htdv[i * offset + j];             
        f7[j] = - dTchia2dY * Htdv[i * offset + j - 1];
        f8[j] = z[i * offset + j] - dTchia2dX * (Htdu[i * offset + j] * u[i * offset + j] - Htdu[(i - 1) * offset + j] * u[(i - 1) * offset + j]);
        //printf("gpu: z %f, htdu: %f, u: %f \n", z[i * offset + j], Htdu[i * offset + j], u[i * offset + j] );
        a2[j] = f7[j];
        c2[j] = f6[j];
        d2[j] = f8[j];
    }
   
    update_abcd_at_l(first, last, bienran1, bienran2, a1, b1, c1, d1, f1, f2, f3, f4, f5, f6, f7, f8);

    int sn = boundary_config(false, first, last, bienran1, bienran2, 
        vbt[i], vbd[i], t_z[i * offset + last], t_z[i * offset + first], bienQ[1], bienQ[0], M, a1, b1, c1, d1, a2, b2, c2, d2, AA, BB, CC, DD);
    if (sn > 0)
        tridiag(sn, AA, BB, CC, DD, x, Ap, Bp, ep);

    if (bienran1){
        for (int j = first; j < last; j++){
            t_z[i * offset + j] = x[2 * (j - first) + 1];
            t_v[i * offset + j] = x[2 * (j - first) + 2];
        }
        t_v[i * offset + first - 1] = 0;
    }
    else{
        if( (bienQ[1]) && (first == 2)){
            for (int j = first; j < last; j++){
                t_z[i * offset + j] = x[2 * (j - first) + 1];
                t_v[i * offset + j] = x[2 * (j - first) + 2];
            }
            t_v[i * offset + first - 1] = vbd[i];
        }
        else{
            t_v[i * offset + first] = x[1];
            t_v[i * offset + first - 1] = (d2[first] - t_z[i * offset + first] - c2[first] * t_v[i * offset + first]) / a2[first];
            for (int j = first + 1; j < last; j++){
                t_z[i * offset + j] = x[2 * (j - first)];
                t_v[i * offset + j] = x[2 * (j - first) + 1];
            }
        }
    }

    if (bienran2){
        t_v[i * offset + last] = 0;
        t_z[i * offset + last] = x[sn];
    }
    else{
        if ((bienQ[0]) && (last == M)){
            t_v[i * offset + last] = vbt[i];
            t_z[i * offset + last] = x[sn];
        }
        else{
            t_v[i * offset + last] = (d2[last] - a2[last] * t_v[i * offset + last - 1] - t_z[i * offset + last]) / c2[last];
        }
    }
}

__device__ void uSolver(int offset, int first, int last, int jpos, bool bienran1, bool bienran2, DOUBLE* VISCOIDX, DOUBLE* Tsxw,
    DOUBLE *v, DOUBLE *t_v, DOUBLE *u, DOUBLE *t_u, DOUBLE *z, DOUBLE *t_z, DOUBLE *Kx1, DOUBLE *Htdu, DOUBLE *H_moi){

    int j = jpos;
    DOUBLE p, q, tmp;
    for (int i = first; i < last; i++){
        p = 0; q = 0; tmp = 0;
        DOUBLE vtb = (v[i * offset + j - 1] + v[i * offset + j] + v[(i + 1) * offset +  j - 1] + v[(i + 1) * offset +  j]) * 0.25;
        DOUBLE t_vtb = (t_v[i * offset + j - 1] + t_v[i * offset + j] + t_v[(i + 1) * offset +  j - 1] + t_v[(i + 1) * offset +  j]) * 0.25;
        p = (u[(i + 1) * offset +  j] - u[(i - 1) * offset +  j]) / dX2;
        p = (HaiChiadT + p + Kx1[i * offset + j] * sqrt(vtb * vtb + u[i * offset + j] * vtb) / Htdu[i * offset + j]);
        //print vtb, ' ', t_vtb
        if (H_moi[i * offset + j - 1] <= H_TINH){
            if (vtb < 0){
                q = t_vtb * (-3 * u[i * offset + j] + 4 * u[i * offset + j + 1] - u[i * offset +  j + 2]) / dY2;
                tmp = (u[i * offset +  j] - 2 * u[i * offset +  j + 1] + u[i * offset +  j + 2] ) / dYbp;
            }
        }
        else
            if (H_moi[i * offset +  j + 1] <= H_TINH)
                    if ((H_moi[i * offset +  j - 2] > H_TINH) && (vtb > 0)){ 
                        q = t_vtb * (3 * u[i * offset +  j] - 4 * u[i * offset +  j - 1] + u[i * offset +  j - 2]) /dY2;
                        tmp = (u[i * offset +  j] - 2 * u[i * offset +  j - 1] + u[i * offset +  j - 2] ) / dYbp;
                    }
            else{
                q = t_vtb * (u[i * offset +  j + 1] - u[i * offset +  j - 1]) / dY2;
                tmp = (u[i * offset +  j + 1] - 2 * u[i * offset +  j] + u[i * offset +  j - 1]) / dYbp;
            }
                //print 'q is calculated in line 395'
        //print q
        q = HaiChiadT * u[i * offset +  j] - q + CORIOLIS_FORCE * t_vtb;
        q = (q - g * (z[(i + 1) * offset +  j] - z[i * offset +  j]) / dX + VISCOIDX[i * offset +  j] * ((u[(i + 1) * offset +  j] - 2 * u[i * offset +  j] + u[(i - 1) * offset +  j]) / dXbp + tmp)) + (Windx() - Tsxw[i * offset +  j]) / Htdu[i * offset +  j];
        //print ' ', dX, ' ', dXbp, ' ', Htdu[i, j]
        t_u[i * offset +  j] = q / p;
    }
    if (bienran1)
        t_u[(first - 1) * offset +  j]  = 0;
    else
        t_u[(first - 1) * offset +  j] = 2 * t_u[first * offset + j] - t_u[(first + 1) * offset + j];
    if (bienran2)
        t_u[last * offset +  j] = 0;
    else
        t_u[last * offset +  j] = 2 * t_u[(last - 1) * offset +  j] - t_u[(last - 2) * offset +  j];
    
}

__device__ void vSolver(int offset, int first, int last, int ipos, bool bienran1, bool bienran2, DOUBLE* VISCOIDX, DOUBLE* Tsyw, 
    DOUBLE *v, DOUBLE *t_v, DOUBLE *u, DOUBLE *t_u, DOUBLE *z, DOUBLE *t_z, DOUBLE *Ky1, DOUBLE *Htdv, DOUBLE *H_moi){

    DOUBLE p, q, tmp;
    int i = ipos;
    for (int j = first; j < last; j++){
        q = 0;
        p = 0;
        tmp = 0;
        DOUBLE utb = (u[(i - 1) * offset +  j] + u[i * offset +  j] + u[(i - 1) * offset +  j + 1] + u[i * offset +  j + 1]) * 0.25;

        DOUBLE t_utb = (t_u[(i - 1) * offset +  j] + t_u[i * offset +  j] + t_u[(i - 1) * offset +  j + 1] + t_u[i * offset +  j + 1]) * 0.25;
        //printf("utb: %.14f, %.14f, %.14f, %d, %d\n", utb, t_utb, q, i, j);
        p = (v[i * offset +  j + 1] - v[i * offset +  j - 1]) / dY2;
        p = (HaiChiadT + p + Ky1[i * offset +  j] * sqrt(utb * utb + v[i * offset +  j] * v[i * offset +  j]) / Htdv[i * offset +  j]);
        //printf("H_moi[%d %d], %.10f \n", i, j, H_moi[i - 1, j]);
        if (H_moi[(i - 1) * offset +  j] <= H_TINH){
           // printf("here484, %d %d\n", i, j);
            if (utb < 0){
                q = t_utb * (-3 * v[i * offset +  j] + 4 * v[(i + 1) * offset +  j] + v[(i + 2) * offset +  j]) / dX2;
                tmp = (v[i * offset +  j] - 2 * v[(i + 1) * offset +  j] + v[(i + 2) * offset +  j] ) / dXbp;
            }
        }
        else{
            //printf("here491, %d %d\n", i, j);
            if (H_moi[(i + 1) * offset +  j] <= H_TINH){
                if ((H_moi[(i - 2) * offset +  j] > H_TINH) && (utb > 0)){
                    //printf("here487, %d %d\n", i, j);
                    q = t_utb * (3 * v[i * offset +  j] - 4 * v[(i - 1) * offset +  j] + v[(i - 2) * offset +  j]) /dX2;
                    tmp = (v[i * offset +  j] - 2 * v[(i - 1) * offset +  j] + v[(i - 2) * offset +  j] ) / dXbp;
                }
            }
            else{
                //printf("here489, %d %d\n", i, j);
                q = t_utb * (v[(i + 1) * offset +  j] - v[(i - 1) * offset +  j]) / dX2;
                tmp = (v[(i + 1) * offset +  j] - 2 * v[i * offset +  j] + v[(i - 1) * offset +  j]) / dXbp;
            }
        }

        //if (i == 4) printf("q1: %.14f %d %d\n", q, i, j);
        q = HaiChiadT * v[i * offset +  j] - q - CORIOLIS_FORCE * t_utb;
        q = (q - g * (z[i * offset +  j + 1] - z[i * offset +  j]) / dY + VISCOIDX[i * offset +  j] * (tmp + (v[i * offset +  j + 1] - 2 * v[i * offset +  j] + v[i * offset +  j - 1]) / dYbp)) + (Windy() - Tsyw[i * offset +  j]) / Htdv[i * offset +  j];
        

        t_v[i * offset +  j] = q / p ;
        //if (i == 4) printf("p: %.14f, q: %.14f, t_v[%d, %d] : %.14f \n", p, q, i, j, t_v[i * offset + j] );

    }

    if (bienran1)
        t_v[i * offset +  first - 1] = 0;
    else{
        t_v[i * offset +  first - 1] = 2 * t_v[i * offset +  first] - t_v[i * offset +  first + 1];
    }
    if (bienran2){
        t_v[i * offset +  last] = 0;
    }
    else{
        t_v[i * offset +  last] = 2 * t_v[i * offset +  last - 1] - t_v[i * offset +  last - 2];
    }
    //printf("t_v[%d, %d] = %f\n",i, j, t_v[i * offset + j] );
    //printf("Htdv[%d, %d] = %f\n",i, j, Htdv[i * offset + j] );
    
}


__device__ void set_boundary_vslice(int N, int M, bool* bienran1, bool* bienran2, int* first, int* last, int index, int k, int h_offset, int* daui, int* cuoii, DOUBLE* h){
    int i = index;
    int offset = 5;
    //printf("mem_offset: %d \n", i * offset + k);
    *first = daui[i * offset + k];
    *last = cuoii[i * offset + k];
    //printf("thread: %d A: dau: %d, cuoi: %d\n", threadIdx.x, *first, *last);

    offset = h_offset;
    if ((*first > 2) || ((*first == 2) && (h[i * offset + *first - 1] + h[(i - 1) * offset + *first - 1]) * 0.5 == NANGDAY))
        *bienran1 = true;
    if ((*last < M) || ((*last == M) && (h[i * offset +  *last] + h[(i - 1) * offset + *last]) * 0.5 == NANGDAY))
        *bienran2 = true;
    //printf("thread: %d B: dau: %d, cuoi: %d\n", threadIdx.x, *first, *last);
}

__device__ void set_boundary_uslice(int N, int M, bool* bienran1, bool* bienran2, int* first, int* last, int index, int k, int h_offset, int* dauj, int* cuoij, DOUBLE* h ){
    int offset = 5;
    int j = index;

    *first = dauj[j * offset +  k];
    *last = cuoij[j * offset + k];

    offset = h_offset;

    DOUBLE *depth = &h[(*first - 1) * offset];
    if ((*first > 2) || ((*first == 2) && (depth[j] + depth[j - 1]) * 0.5 == NANGDAY))
        *bienran1 = true;

    depth = &h[*last * h_offset];
    if ((*last < N) || ((*last == N) && (depth[j] + depth[j - 1]) * 0.5 == NANGDAY))
        *bienran2 = true;
    //printf("thread: %d D: dau: %d, cuoi: %d\n", threadIdx.x, *first, *last);
}


// mem_offset: Memory offset for each 
__global__ void solveUZ(int M, int N, int mem_offset, int startidx, int endidx, int* mocj, int* dauj, int* cuoij,  bool*bienQ, DOUBLE* Tsxw,
    DOUBLE* v, DOUBLE* u, DOUBLE* z, DOUBLE* Htdu, DOUBLE* Htdv, DOUBLE* VISCOIDX, DOUBLE* t_u, DOUBLE* t_z, DOUBLE* h,
    DOUBLE* ubt, DOUBLE* ubp, DOUBLE* H_moi, DOUBLE* Kx1, DOUBLE *f1, DOUBLE *f2, DOUBLE *f3, DOUBLE *f5, DOUBLE *f6, DOUBLE *f7, DOUBLE *f8,
    DOUBLE *a1, DOUBLE* b1, DOUBLE *c1, DOUBLE *d1, DOUBLE *a2, DOUBLE *b2, DOUBLE *c2, DOUBLE *d2,
    DOUBLE *AA, DOUBLE *BB, DOUBLE *CC, DOUBLE *DD, DOUBLE *Ap, DOUBLE *Bp, DOUBLE* ep, DOUBLE* x ){

        int j = calculate_index(M) + startidx;
        if (j >= endidx) return;
        //if (j == 2) printf("precision check %.16f\n", pow (2.70134219723423422342334134, 2.70134219723423422342334134));
        int row_Thomas = (blockIdx.x*blockDim.x + threadIdx.x) * mem_offset;
        int row_f = (blockIdx.x*blockDim.x + threadIdx.x) * (N + 2);
        //printf("hello from first kernel %d\n", threadIdx.x);
        for (int k = 0; k < mocj[j]; k++){
            // set boundary here
            bool bienran1 = false;
            bool bienran2 = false;
            int first, last;
            int h_offset = M + 3; 
            set_boundary_uslice(N, M, &bienran1, &bienran2, &first, &last, j, k, h_offset, dauj, cuoij, h);
            // debug

            //cudaError_t Errtype = cudaGetLastError();
            //printf("%s\n",cudaGetErrorString(Errtype));


            uzSolver(M + 3, N, M, first, last, j, bienran1, bienran2, dXbp, dYbp, dT, dTchia2dX, dTchia2dY, Tsxw,
            v, u, z, Htdu, Htdv, VISCOIDX, t_u, t_z, ubt, ubp, H_moi, Kx1, bienQ, &f1[row_f], &f2[row_f], &f3[row_f], &f5[row_f], 
            &f6[row_f], &f7[row_f], &f8[row_f], &a1[row_f], &b1[row_f], &c1[row_f], &d1[row_f], &a2[row_f], &b2[row_f], &c2[row_f], &d2[row_f], 
            &AA[row_Thomas], &BB[row_Thomas], &CC[row_Thomas], &DD[row_Thomas], &Ap[row_Thomas], &Bp[row_Thomas], &ep[row_Thomas], &x[row_Thomas]);
        }

    }

__global__ void SolveVZ(int M, int N, int mem_offset,int startidx, int endidx, int* moci, int* daui, int* cuoii,  bool*bienQ, DOUBLE* Tsyw,
    DOUBLE* v, DOUBLE* u, DOUBLE* z, DOUBLE* Htdu, DOUBLE* Htdv, DOUBLE* VISCOIDX, DOUBLE* t_v, DOUBLE* t_z, DOUBLE* h,
    DOUBLE* vbt, DOUBLE* vbd, DOUBLE* H_moi, DOUBLE* Ky1, DOUBLE *f1, DOUBLE *f2, DOUBLE *f3, DOUBLE *f5, DOUBLE *f6, DOUBLE *f7, DOUBLE *f8,
    DOUBLE *a1, DOUBLE* b1, DOUBLE *c1, DOUBLE *d1, DOUBLE *a2, DOUBLE *b2, DOUBLE *c2, DOUBLE *d2,
    DOUBLE *AA, DOUBLE *BB, DOUBLE *CC, DOUBLE *DD, DOUBLE *Ap, DOUBLE *Bp, DOUBLE* ep, DOUBLE* x){
    
    //int i = (blockIdx.x*blockDim.x + threadIdx.x) + startidx;
    int i = calculate_index(M) + startidx;
    if (i >= endidx) return;

    //printf("thread no %d say hello from second kernel\n", blockIdx.x*blockDim.x + threadIdx.x);
    int row_Thomas = (blockIdx.x*blockDim.x + threadIdx.x) * mem_offset;
    int row_f = (blockIdx.x*blockDim.x + threadIdx.x) * (M + 2);
    for (int k = 0; k < moci[i]; k++){
        bool bienran1 = false;
        bool bienran2 = false;
        int first, last;
        int h_offset = M + 3;
        set_boundary_vslice(N, M, &bienran1, &bienran2, &first, &last, i, k, h_offset, daui, cuoii, h);
        vzSolver(M + 3, N, M, first, last, i, bienran1, bienran2, dXbp, dYbp, dT, dTchia2dX, dTchia2dY, Tsyw,
            v, u, z, Htdu, Htdv, VISCOIDX, t_v, t_z, vbt, vbd, H_moi, Ky1, bienQ,&f1[row_f],
            &f2[row_f], &f3[row_f], &f5[row_f], &f6[row_f], &f7[row_f], &f8[row_f], &a1[row_f], &b1[row_f], &c1[row_f], &d1[row_f], &a2[row_f], &b2[row_f], &c2[row_f], &d2[row_f], 
            &AA[row_Thomas], &BB[row_Thomas], &CC[row_Thomas], &DD[row_Thomas], &Ap[row_Thomas], &Bp[row_Thomas], &ep[row_Thomas], &x[row_Thomas] );
    }

}


__global__ void solveU(int N, int M, int startidx, int endidx, DOUBLE* VISCOIDX, DOUBLE* Tsxw, int* mocj, int* dauj, int* cuoij,
    DOUBLE *v, DOUBLE *t_v, DOUBLE *u, DOUBLE *t_u, DOUBLE *z, DOUBLE *t_z, DOUBLE *Kx1, DOUBLE *Htdu, DOUBLE *H_moi, DOUBLE* h){
    int j = calculate_index(M) + startidx;
    if (j >= endidx) return;
    //printf("thread no %d say hello from third kernel\n", blockIdx.x*blockDim.x + threadIdx.x);
        for (int k = 0; k < mocj[j]; k++){
            bool bienran1 = false;
            bool bienran2 = false;
            int first, last;
            int h_offset = M + 3;
            set_boundary_uslice(N, M, &bienran1, &bienran2, &first, &last, j, k, h_offset, dauj, cuoij, h);
            uSolver(M + 3,first, last, j, bienran1, bienran2, VISCOIDX, Tsxw, v, t_v, u, t_u, z, t_z, Kx1, Htdu,H_moi);
    }

    
}


__global__ void solveV(int N, int M, int startidx, int endidx, DOUBLE* VISCOIDX, DOUBLE* Tsyw, int* moci, int* daui, int* cuoii,
    DOUBLE *v, DOUBLE *t_v, DOUBLE *u, DOUBLE *t_u, DOUBLE *z, DOUBLE *t_z, DOUBLE *Ky1, DOUBLE *Htdv, DOUBLE *H_moi, DOUBLE* h){
        int i = calculate_index(M) + startidx;
        if (i >= endidx) return;
        //printf("thread no %d say hello from forth kernel\n", blockIdx.x*blockDim.x + threadIdx.x);
        for (int k = 0; k < moci[i]; k++){
            bool bienran1 = false;
            bool bienran2 = false;
            int first, last;
            int h_offset = M + 3;
            set_boundary_vslice(N, M, &bienran1, &bienran2, &first, &last, i, k, h_offset, daui, cuoii, h);
            vSolver(M +3, first, last, i, bienran1, bienran2, VISCOIDX, Tsyw, v, t_v, u, t_u, z, t_z, Ky1, Htdv, H_moi);
        }

}