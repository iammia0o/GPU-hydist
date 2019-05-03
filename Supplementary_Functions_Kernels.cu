
// ngang
#include "constant.cuh"

__device__ void calculate_index(int *i, int *j, int M){
    int thrx = blockIdx.x * blockDim.x + threadIdx.x;
    int thry = blockIdx.y * blockDim.y + threadIdx.y;

    int thrnu =  thrx * (blockDim.y * gridDim.y) + thry;
    *i = thrnu / M;
    *j = thrnu % M;
}

__global__ void Reset_states_horizontal(int M, double* H_moi, double* htaiz, int* khouot, double* z, double* t_z, double* t_u, double* t_v ){
	//int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	//int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int i, j;
	calculate_index(&i, &j, M);
	i++; j++;
	int offset = M + 3;
	
	if (khouot[i * offset + j] == 0){
		H_moi[i * offset + j] = htaiz[i * offset + j] + t_z[i * offset + j];
	}

	j = j + 1;
	if (t_z[i * offset + j] > z[i * offset + j]){
		if (khouot[i * offset + j - 1] == 1){
			t_z[i * offset + j - 1] = t_z[i * offset + j];
			H_moi[i * offset + j - 1] = htaiz[i * offset + j - 1] + t_z[i * offset + j];
            khouot[i * offset + j - 1] = 0;

            
		}
		if (khouot[i * offset + j + 1] == 1){
			t_z[i * offset + j + 1] = t_z[i * offset + j];
            H_moi[i * offset + j + 1] = htaiz[i * offset + j + 1] + t_z[i * offset + j];
            khouot[i * offset + j + 1] = 0;

		}
	}

	if ((khouot[i * offset + j] == 0) && (H_moi[i * offset + j] <= H_TINH)){
		
		t_u[(i - 1) * offset + j] = 0;
		t_u[i * offset + j] = 0;
		t_v[i * offset + j - 1] = 0;
		t_v[i * offset + j] = 0;
		khouot[i * offset + j] = 1;
		
	}

}


// doc
__global__ void Reset_states_vertical(int M, double* H_moi,double* htaiz, int* khouot, 
	double* z, double* t_z, double* t_u, double* t_v){
	//int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	//int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int i, j;
	calculate_index(&i, &j, M);
	i++; j++;
	int offset = M + 3;
	if (khouot[i * offset + j] == 0)
		H_moi[i * offset + j] = htaiz[i * offset + j] + t_z[i * offset + j];

	i = i + 1;
	if (t_z[i * offset + j] > z[i * offset + j]){
		if (khouot[(i - 1) * offset + j] == 1){
			t_z[(i - 1) * offset + j] = t_z[i * offset + j];
			H_moi[(i - 1) * offset + j] = htaiz[(i - 1) * offset + j] + t_z[i * offset + j];
            khouot[(i - 1) * offset + j] = 0;
		}
		if (khouot[(i + 1) * offset + j] == 1){
			t_z[(i + 1) * offset + j] = t_z[i * offset + j];
            H_moi[(i + 1) * offset + j] = htaiz[(i + 1) * offset + j] + t_z[i * offset + j];
            khouot[(i + 1) * offset + j] = 0;
		}
	}
	if ((!khouot[i * offset + j]) && (H_moi[i * offset + j] <= H_TINH)){
		t_u[(i - 1) * offset + j] = 0;
		t_u[i * offset + j] = 0;
		t_v[(i - 1) * offset + j] = 0;
		t_v[i * offset + j] = 0;
		khouot[i * offset + j] = 1;
	}
}


__global__ void update_uvz(int M, double* u, double* v, double* z, double* t_u, double* t_v, double* t_z, int kenhhepd=1, int kenhhepng=0){
	int i, j;
	// calculate_index(&i, &j, M + 3);
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j > M) return;
	int offset = M + 3;
	z[i * offset + j] = t_z[i * offset + j];
	u[i * offset + j] = t_u[i * offset + j] * (1 - kenhhepd);
	v[i * offset + j] = t_v[i * offset + j] * (1 - kenhhepng);
}


__global__ void Find_Calculation_limits_Horizontal( int m_offset,  int M, int N, int* daui, int* cuoii, int* moci, int* khouot){
	int thrx = blockIdx.x * blockDim.x + threadIdx.x;
    int thry = blockIdx.y * blockDim.y + threadIdx.y;
    int i = thrx * (blockDim.y * gridDim.y) + thry + 2;
    if (i > N) return;
	moci[i] = 0;
	int start = 2;
	int end = 0;
	int offset = M + 3;
	while (start < M){
		//printf("i: %d, start %d \n",i, start );
		if (khouot[i * offset + start] != 0){
			while ((khouot[i * offset + start]) && (start < M)) start++;
			//printf("start: %d, i: %d, ku: %d\n", start, i, khouot[i * offset + start] );
		} 

		if (khouot[i * offset + start] == 0){
			daui[i * m_offset + moci[i]] = start;
			//printf("start: %d, i: %d\n", start, i );
			end  = start;
			while((khouot[i * offset + end] == 0) && (end < M)) end++;
			if ((khouot[i * offset + end] != 0) && (end <= M)){
				cuoii[i * m_offset + moci[i]] = end - 1;
				start = end;
				moci[i]++;
			} else{
				cuoii[i * m_offset + moci[i]] = M ;
				//printf("i: %d, cuoii : %d\n", i, M);
				start = M;
				moci[i]++;
			}
			
		} 
	}


}


__global__ void Find_Calculation_limits_Vertical(int m_offset, int M, int N, int* dauj, int* cuoij, int* mocj, int* khouot){
	int thrx = blockIdx.x * blockDim.x + threadIdx.x;
    int thry = blockIdx.y * blockDim.y + threadIdx.y;
    int j = thrx * (blockDim.y * gridDim.y) + thry + 2;
    if (j > M) return;
	mocj[j] = 0;
	int start = 2;
	int end = 0;
	int offset = M + 3;

	while (start < N){
		if (khouot[start * offset + j] != 0 ){
			while ((khouot[start * offset + j]) && (start < N)) start++;
		}

		if (khouot[start * offset + j] == 0){
			dauj[j * m_offset + mocj[j]] = start;
			end = start;
			while ((khouot[end * offset + j] == 0) && (end < N)) end++;
			
			if ((khouot[end * offset + j] != 0) && ( end <= N)){
				
				cuoij[j * m_offset + mocj[j]] = end - 1;
				mocj[j] ++;
				start = end;
			} else{
				cuoij[j * m_offset + mocj[j]] = N;
				start = N;
				mocj[j]++;
			}
		}
	}
}



__global__ void Htuongdoi(int M, int N, double* Htdu, double* Htdv, double* z, double* h){
	int i, j;
	calculate_index(&i, &j, M);
	i++; j++;
	if ((i > N) || (j > M)) return;
	int offset = M + 3;
	Htdu[i * offset + j] = (h[i * offset + j - 1] + h[i * offset + j] + z[(i + 1) * offset + j] + z[i * offset + j]) * 0.5;
    Htdv[i * offset + j] = (h[(i - 1) * offset + j] + h[i * offset + j] + z[i * offset + j + 1] + z[i * offset + j]) * 0.5;
	
}

__global__ void boundary_up(double t, int m_offset, int M, int N, bool* bienQ, double* t_z, int* daui, int* cuoii, 
	int* dauj, int* cuoij, double* vbt, double* vbd, double* ubt, double* ubp ){
	
	int thrx = blockIdx.x * blockDim.x + threadIdx.x;
    int thry = blockIdx.y * blockDim.y + threadIdx.y;
    int i = thrx * (blockDim.y * gridDim.y) + thry + dauj[M * m_offset];
	int offset = M + 3;
	//printf("i = %d, cuoij: %d\n", i, cuoij[M * offset] );
	if (i > cuoij[M * m_offset]) return;
	if (bienQ[0])
		{vbt[i] = 0;
				printf("here\n");
		}
	else{
		t_z[i * offset + M] = 0.01 * cos(2 * PI / 27.750 * t) * cos(2 * (PI / 100) * (100 - dY / 2));
		t_z[i * offset + M + 1] = 0.01 * cos(2 * PI / 27.75 * t)  * cos(2 * (PI / 100) *  (100 + dY / 2));
		//printf("tz[%d, %d] = %.15f\n",i, M, t_z[i * offset + M]);
	}
}

__global__ void boundary_down(double t, int m_offset, int M, int N, bool* bienQ, double* t_z, int* daui, int* cuoii, 
	int* dauj, int* cuoij, double* vbt, double* vbd, double* ubt, double* ubp ){
	
	int thrx = blockIdx.x * blockDim.x + threadIdx.x;
    int thry = blockIdx.y * blockDim.y + threadIdx.y;
    int i = thrx * (blockDim.y * gridDim.y) + thry + dauj[2 * m_offset];
	int offset = M + 3;
	if (i > cuoij[2 * m_offset]) return;
	if (bienQ[1])
		vbd[i] = 0;
	else{
		t_z[i * offset + 2] = 0.01 * cos(2 * PI / 27.75 * t ) * cos(2 * (PI / 100) * dY / 2);
        t_z[i * offset + 1] = 0.01 * cos(2 * PI / 27.75 * t ) * cos(2 * (PI / 100) * (-dY) / 2);
        //if (t >= 6.75) printf(" tz[%d, %d] = %.15f\n",i, 2, t_z[i * offset + 2]);
	}
}

__global__ void boundary_left(double t, int m_offset, int M, int N, bool* bienQ, double* t_z, int* daui, int* cuoii, 
	int* dauj, int* cuoij, double* vbt, double* vbd, double* ubt, double* ubp){

	
	int thrx = blockIdx.x * blockDim.x + threadIdx.x;
    int thry = blockIdx.y * blockDim.y + threadIdx.y;
    int i = thrx * (blockDim.y * gridDim.y) + thry + daui[2 * m_offset];
	int offset = M + 3;
	//printf("i: %d\n", cuoii[2 * offset]);
	if (i > cuoii[2 * m_offset]) return;
	if (bienQ[2])
		ubt[i] = 0;
	else{
		t_z[2 * offset + i] = 0.01 * cos(2 * PI / 27.75 * t ) * cos(2 * (PI / 100) * dX / 2);
        t_z[1 * offset + i] = 0.01 * cos(2 * PI / 27.75 * t ) * cos(2 * (PI / 100) * (- dX) / 2);
        //if (t >= 6.75) printf("tz[%d, %d] = %.15f\n",2, i, t_z[2 * offset + i]);

	}

}

__global__ void boundary_right(double t, int m_offset, int M, int N, bool* bienQ, double* t_z, int* daui, int* cuoii, 
	int* dauj, int* cuoij, double* vbt, double* vbd, double* ubt, double* ubp){
	
	int thrx = blockIdx.x * blockDim.x + threadIdx.x;
    int thry = blockIdx.y * blockDim.y + threadIdx.y;
    int i = thrx * (blockDim.y * gridDim.y) + thry + daui[2 * m_offset];
	int offset = M + 3;
	if (i > cuoii[N * m_offset]) return;
	if (bienQ[3])
		ubp[i] = 0;
	else{
		t_z[N * offset + i] = 0.01 * cos(2 * PI / 27.75 * t)  * cos(2 * (PI / 100) *  (100 - dX / 2));
        t_z[(N + 1) * offset + i] = 0.01 * cos(2 * PI / 27.75 * t)  * cos(2 * (PI / 100) *  (100 + dX / 2));
        //if (t >= 6.75) printf("tz[%d, %d] = %.15f\n",N, i, t_z[N * offset + i]);
	}
}


__global__ void Boundary_at(double t, int offset, int M, int N, bool* bienQ, double* t_z, int* daui, int* cuoii, int* dauj, int* cuoij){
	double t1 = t / 3600;
	double t2 = t / 3600.0 - t1;

}
