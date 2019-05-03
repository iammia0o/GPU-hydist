solveVZ 
	1. find coeffs for tridiag
	2. solve tridiah
	3. assign solutions found from tridiag to 

	__global__ void SolveVZ(int M, int N, int mem_offset,int startidx, int endidx, int* moci, int* daui, int* cuoii,  bool*bienQ, DOUBLE* Tsyw,
    DOUBLE* v, DOUBLE* u, DOUBLE* z, DOUBLE* Htdu, DOUBLE* Htdv, DOUBLE* VISCOIDX, DOUBLE* t_v, DOUBLE* t_z, DOUBLE* h,
    DOUBLE* vbt, DOUBLE* vbd, DOUBLE* H_moi, DOUBLE* Ky1, DOUBLE *f1, DOUBLE *f2, DOUBLE *f3, DOUBLE *f5, 
    DOUBLE *a1, DOUBLE* b1, DOUBLE *c1, DOUBLE *d1, DOUBLE *a2, DOUBLE *b2, DOUBLE *c2, DOUBLE *d2,
    DOUBLE *AA, DOUBLE *BB, DOUBLE *CC, DOUBLE *DD, DOUBLE *Ap, DOUBLE *Bp, DOUBLE* ep, DOUBLE* x){
    
    //int i = (blockIdx.x*blockDim.x + threadIdx.x) + startidx;
    // level 1: one thread per slice
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
        __set_boundary_vslice(N, M, &bienran1, &bienran2, &first, &last, i, k, h_offset, daui, cuoii, h);
        vzSolver(M + 3, N, M, first, last, i, bienran1, bienran2, Tsyw,
            v, u, z, Htdu, Htdv, VISCOIDX, t_v, t_z, vbt, vbd, H_moi, Ky1, bienQ,&f1[row_f], &f2[row_f], &f3[row_f], &f5[row_f], 
            &a1[row_f], &b1[row_f], &c1[row_f], &d1[row_f], &a2[row_f], &b2[row_f], &c2[row_f], &d2[row_f], 
            &AA[row_Thomas], &BB[row_Thomas], &CC[row_Thomas], &DD[row_Thomas], &Ap[row_Thomas], &Bp[row_Thomas], &ep[row_Thomas], &x[row_Thomas] );
    }




__device__ void _calculate_abcd(int first, int last, int i, bool bienran1, bool bienran2, Array_Pointers* arrp){

__device__ int _calculate_matrix_coeff(bool isU, int first, int last, bool bienran1, int idx,
    bool bienran2, DOUBLE ubp_or_vbt, DOUBLE ubt_or_vbd, DOUBLE TZ_r, DOUBLE TZ_l, bool dkBienQ_1, bool dkBienQ_2, int dkfr, Array_Pointers* arrp){
__device__ _extract_solution( int i,int j, first,last, bool bienran1, bool bienran2, Argument_Pointers *arg, Array_Pointers * arr)
