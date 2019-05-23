// ngang
#include "constant.cuh"
#define DOUBLE double
#define powf pow
__device__ void calculate_index(int *i, int *j, int M){
    int thrx = blockIdx.x * blockDim.x + threadIdx.x;
    int thry = blockIdx.y * blockDim.y + threadIdx.y;

    int thrnu =  thrx * (blockDim.y * gridDim.y) + thry;
    *i = thrnu / M;
    *j = thrnu % M;
}

// checked Mar-30
__global__ void Onetime_init( Argument_Pointers *arg){
	int M = arg->M;
	int N = arg->N;
	int* khouot = arg->khouot;
	DOUBLE* h = arg->h;
	DOUBLE* H_moi = arg->H_moi;
	DOUBLE* Kx1 = arg->Kx1;
	DOUBLE* Ky1 = arg->Ky1;
	DOUBLE* hsnham = arg->hsnham;
	DOUBLE* htaiz = arg->htaiz;
	int width = M + 3;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (( i >= N + 3) || (j >= M + 3)) return;
	// ATTENTION
	khouot [i * width] = khouot [j] = 2;
	i++; j++;
	khouot[i * width +  j] = 2;
	if ((i > N + 1) || (j > M + 1)) return;
	// khouot
	if ((h[(i - 1) * width + j - 1] + h[(i - 1) * width + j] + h[i * width + j - 1] + h[i * width + j]) * 0.25 > NANGDAY){
		khouot[i * width + j] = 0;
		H_moi[i * width + j] = 0;
		// htaiz[i * width + j];
	}


	// giatriHtaiZ
	if (i > N || j > M)  return;
	htaiz[i * width + j] = (h[(i - 1) * width + j - 1] + h[(i - 1) * width + j] + h[i *width + j - 1] + h[i * width + j]) * 0.25;

	// hesok
	if ( h[i * width + j - 1 ] + h[i * width + j] != 0 )
		Kx1[i * width + j] = g * powf( (h[i * width + j - 1] + h[i * width + j]) * 0.5, -2 * mu_mn) * powf( (hsnham[i * width + j - 1] + hsnham[i * width + j] ) * 0.5, 2);

	if (h[(i - 1) * width + j] + h[i * width + j] != 0)
		Ky1[i * width + j] = g * powf((h[(i - 1) * width + j] + h[i * width + j]) * 0.5, -2 * mu_mn) * powf((hsnham[(i - 1) * width + j] + hsnham[i * width + j]) * 0.5, 2);

}
__global__ void update_h_moi(Argument_Pointers* arg){
	int M = arg-> M ;
	int N = arg-> N;
	int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (( i > N) || (j > M)) return;
	int* khouot = arg-> khouot;
	DOUBLE* H_moi = arg-> H_moi;
	DOUBLE* t_z = arg-> t_z;
	DOUBLE* htaiz = arg-> htaiz;
	int grid_pos = i * (M + 3) + j;
	if (khouot[grid_pos] == 0)
		H_moi[grid_pos] = htaiz[grid_pos] + t_z[grid_pos];
}

__global__ void Reset_states_horizontal(Argument_Pointers* arg){
	int M = arg-> M;
	int N = arg-> N;
	DOUBLE* H_moi = arg->H_moi;
	DOUBLE* htaiz = arg-> htaiz;
	int* khouot = arg->khouot;
	DOUBLE* z = arg->z;
	DOUBLE* t_z = arg-> t_z;
	DOUBLE* t_u = arg-> t_u;
	DOUBLE* t_v = arg-> t_v;
	DOUBLE* FS = arg->FS;

	int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
	// int j = blockIdx.x * blockDim.x + threadIdx.x + 2;
	int offset = M + 3;
	if (i > N) return;
	// if (( i > N) || (j > M)) return;
	for (int j = 2; j <= M; j++)
	{
		if (t_z[i * offset + j] > z[i * offset + j]){
			if (khouot[i * offset + j - 1] == 1){
				// if (threadIdx.x == 0)
				// 	printf ("open 5 %d %d\n", i, j - 1);
				t_z[i * offset + j - 1] = t_z[i * offset + j];
				H_moi[i * offset + j - 1] = htaiz[i * offset + j - 1] + t_z[i * offset + j];
	            khouot[i * offset + j - 1] = 0;
	            FS[i * offset + j - 1] = FS[i * offset + j];
			}
			if (khouot[i * offset + j + 1] == 1){
				// if (threadIdx.x == 0)
				// 	printf ("open 4 %d %d\n", i, j + 1);
				t_z[i * offset + j + 1] = t_z[i * offset + j];
	            H_moi[i * offset + j + 1] = htaiz[i * offset + j + 1] + t_z[i * offset + j];
	            khouot[i * offset + j + 1] = 0;
	            FS[i * offset + j  + 1] = FS[i * offset + j ];
			}
		}
	}

	for (int j = 2; j <= M; j++){
		
		if ((khouot[i * offset + j] == 0) && (H_moi[i * offset + j] <= H_TINH) ){
			
			t_u[(i - 1) * offset + j] = 0;
			t_u[i * offset + j] = 0;
			
			t_v[i * offset + j - 1] = 0;
			t_v[i * offset + j] = 0;
			khouot[i * offset + j] = 1;
			FS[i * offset + j ] = 0;
			
		}
	}

}


// doc
// __global__ void Reset_states_vertical(int M, int N, DOUBLE* H_moi,DOUBLE* htaiz, int* khouot, DOUBLE* z, DOUBLE* t_z, DOUBLE* t_u, DOUBLE* t_v){
__global__ void Reset_states_vertical(Argument_Pointers* arg){

	int M = arg-> M;
	int N = arg-> N;
	DOUBLE* H_moi = arg->H_moi;
	DOUBLE* htaiz = arg-> htaiz;
	int* khouot = arg->khouot;
	DOUBLE* z = arg->z;
	DOUBLE* t_z = arg-> t_z;
	DOUBLE* t_u = arg-> t_u;
	DOUBLE* t_v = arg-> t_v;
	DOUBLE* FS = arg->FS;

	// int i = blockIdx.y * blockDim.y + threadIdx.y + 2;
	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (j > M) return;
	int offset = M + 3;

	for (int i = 2; i <= N; i++) {
		if (t_z[i * offset + j] > z[i * offset + j]){
			if (khouot[(i - 1) * offset + j] == 1){
				// if (threadIdx.y == 0)
				// 	printf("open 1 %d %d\n", i - 1, j);
				t_z[(i - 1) * offset + j] = t_z[i * offset + j];
				H_moi[(i - 1) * offset + j] = htaiz[(i - 1) * offset + j] + t_z[i * offset + j];
	            khouot[(i - 1) * offset + j] = 0;
	            FS[(i - 1) * offset + j] = FS[i * offset + j];
			}
			if (khouot[(i + 1) * offset + j] == 1){
				// if (threadIdx.y == 0)
				// 	printf("open 2 %d %d \n", i + 1, j);
				t_z[(i + 1) * offset + j] = t_z[i * offset + j];
	            H_moi[(i + 1) * offset + j] = htaiz[(i + 1) * offset + j] + t_z[i * offset + j];
	            khouot[(i + 1) * offset + j] = 0;
	            FS[(i + 1) * offset + j] = FS[i * offset + j];
			}
		}
	}

	for (int i = 2; i <= N; i++) {
		if ((khouot[i * offset + j] == 0) && (H_moi[i * offset + j] <= H_TINH)){
				// if (threadIdx.y == 0)
				// 	printf("close 0 %d %d\n", i, j);
			t_u[(i - 1) * offset + j] = 0;
			t_u[i * offset + j] = 0;

			t_v[i * offset + j - 1] = 0;
			t_v[i * offset + j] = 0;
			khouot[i * offset + j] = 1;
			FS[i * offset + j] = 0;
		}
	}
}



__global__ void Find_Calculation_limits_Horizontal( Argument_Pointers *arg){

	// int thrx = blockIdx.x * blockDim.x + threadIdx.x;
 //    int thry = blockIdx.y * blockDim.y + threadIdx.y;
 //    int i = thrx * (blockDim.y * gridDim.y) + thry + 2;
	int i = blockDim.y * blockIdx.y + threadIdx.y + 2;
	int M = arg->M;
	int N = arg->N;
    if (i > N) return;
    int* khouot = arg->khouot;
    int* moci = arg->moci;
    int* cuoii = arg->cuoii;
    int* daui = arg->daui;
	int number_of_seg = 0;
	int start = 2;
	int end = 0;
	int offset = M + 3;
	while (start < M){
		//printf("i: %d, start %d \n",i, start );
		if (khouot[i * offset + start] != 0){
			while ((khouot[i * offset + start]) && (start < M)) start++;
		} 
		if (start + 1 == M) start = M;

		if (khouot[i * offset + start] == 0 && start + 1 < M){
			daui[i * segment_limit + number_of_seg] = start;
			// if (threadIdx.x == 0)
				// printf("start: %d, i: %d\n", start, i );
			end  = start;
			while((khouot[i * offset + end] == 0) && (end < M)) end++;

			if ((khouot[i * offset + end] != 0) && (end <= M)){
				cuoii[i * segment_limit + number_of_seg] = end - 1;
				start = end;
				number_of_seg++;
			} else{
				cuoii[i * segment_limit + number_of_seg] = M ;
				//printf("i: %d, cuoii : %d\n", i, M);
				start = M;
				number_of_seg++;
			}
			
		} 
	}
	moci[i] = number_of_seg;
}


__global__ void Find_Calculation_limits_Vertical(Argument_Pointers *arg){

	// int thrx = blockIdx.x * blockDim.x + threadIdx.x;
 //    int thry = blockIdx.y * blockDim.y + threadIdx.y;
 //    int j = thrx * (blockDim.y * gridDim.y) + thry + 2;
	int j = blockDim.y * blockIdx.y + threadIdx.y + 2;
	int M = arg->M;
	int N = arg->N;
    if (j > M) return;
    int* khouot = arg->khouot;
    int* mocj = arg->mocj;
    int* cuoij = arg->cuoij;
    int* dauj = arg->dauj;
	int number_of_seg  = 0;
	int start = 2;
	int end = 0;
	int offset = M + 3;
	

	while (start < N){
		if (khouot[start * offset + j] != 0  ){
			while ((khouot[start * offset + j]) && (start < N)) start++;
		}
		if (start + 1 == N) start = N;
		if (khouot[start * offset + j] == 0 && start + 1 < N){
			dauj[j * segment_limit + number_of_seg] = start;
			end = start;
			while ( (khouot[end * offset + j] == 0) && (end < N) ) {end++;}
			
			if ((khouot[end * offset + j] != 0) && ( end <= N)){
				
				// if (threadIdx.x == 0 && j == 3)
				// 	printf(" khouot[%d %d], %d\n", end, j, khouot[end * offset + j]);
				cuoij[j * segment_limit + number_of_seg] = end - 1;
				number_of_seg++;
				// if (j == 3 && threadIdx.x == 0) 
				// 	printf("%d %d\n",end, number_of_seg );
				start = end;
			} else{
				cuoij[j * segment_limit + number_of_seg] = N;
				start = N;
				// if (j == 3 && threadIdx.x == 0) 
				// 	printf("%d %d\n",end, number_of_seg );
				number_of_seg++;
			}
		}
	}
	// if (j == 3 && threadIdx.x == 0)
	// 	printf(" khouot[%d %d], %d\n", end, j, khouot[end * offset + j]);
	mocj[j] = number_of_seg;

}


__global__ void Htuongdoi(Argument_Pointers* arg){

	int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int M = arg->M;
	int N = arg->N; 
	DOUBLE* Htdu = arg->Htdu;
	DOUBLE* Htdv = arg->Htdv;
	DOUBLE* h = arg->h;
	DOUBLE* z = arg->z;

	if ((i > N) || (j > M)) return;
	int width = M + 3;
    Htdu[i * width + j] = (h[i * width + j - 1] + h[i * width + j] + z[(i + 1) * width + j] + z[i * width + j]) * 0.5;
    Htdv[i * width + j] = (h[(i - 1) * width + j] + h[i * width + j] + z[i * width + j + 1] + z[i * width + j]) * 0.5;
    
}

__global__ void boundary_up(DOUBLE t, int segment_limit, int M, int N, bool* bienQ, DOUBLE* t_z, int* daui, int* cuoii, 
	int* dauj, int* cuoij, DOUBLE* vbt, DOUBLE* vbd, DOUBLE* ubt, DOUBLE* ubp ){
	
	int thrx = blockIdx.x * blockDim.x + threadIdx.x;
    int thry = blockIdx.y * blockDim.y + threadIdx.y;
    int i = thrx * (blockDim.y * gridDim.y) + thry + dauj[M * segment_limit];
	int offset = M + 3;
	//printf("i = %d, cuoij: %d\n", i, cuoij[M * offset] );
	if (i > cuoij[M * segment_limit]) return;
	// if (bienQ[0])
	// 	{vbt[i] = 0;
	// 			printf("here\n");
	// 	}
	else{
		t_z[i * offset + M] = 0.01 * cos(2 * PI / 27.750 * t) * cos(2 * (PI / 100) * (100 - dY / 2));
		t_z[i * offset + M + 1] = 0.01 * cos(2 * PI / 27.75 * t)  * cos(2 * (PI / 100) *  (100 + dY / 2));
		//printf("tz[%d, %d] = %.15f\n",i, M, t_z[i * offset + M]);
	}
}

__global__ void boundary_down(DOUBLE t, int segment_limit, int M, int N, bool* bienQ, DOUBLE* t_z, int* daui, int* cuoii, 
	int* dauj, int* cuoij, DOUBLE* vbt, DOUBLE* vbd, DOUBLE* ubt, DOUBLE* ubp ){
	
	int thrx = blockIdx.x * blockDim.x + threadIdx.x;
    int thry = blockIdx.y * blockDim.y + threadIdx.y;
    int i = thrx * (blockDim.y * gridDim.y) + thry + dauj[2 * segment_limit];
	int offset = M + 3;
	if (i > cuoij[2 * segment_limit]) return;
	if (bienQ[1])
		vbd[i] = 0;
	else{
		t_z[i * offset + 2] = 0.01 * cos(2 * PI / 27.75 * t ) * cos(2 * (PI / 100) * dY / 2);
        t_z[i * offset + 1] = 0.01 * cos(2 * PI / 27.75 * t ) * cos(2 * (PI / 100) * (-dY) / 2);
        //if (t >= 6.75) printf(" tz[%d, %d] = %.15f\n",i, 2, t_z[i * offset + 2]);
	}
}

__global__ void boundary_left(DOUBLE t, int segment_limit, int M, int N, bool* bienQ, DOUBLE* t_z, int* daui, int* cuoii, 
	int* dauj, int* cuoij, DOUBLE* vbt, DOUBLE* vbd, DOUBLE* ubt, DOUBLE* ubp){

	
	int thrx = blockIdx.x * blockDim.x + threadIdx.x;
    int thry = blockIdx.y * blockDim.y + threadIdx.y;
    int i = thrx * (blockDim.y * gridDim.y) + thry + daui[2 * segment_limit];
	int offset = M + 3;
	//printf("i: %d\n", cuoii[2 * offset]);
	if (i > cuoii[2 * segment_limit]) return;
	if (bienQ[2])
		ubt[i] = 0;
	else{
		t_z[2 * offset + i] = 0.01 * cos(2 * PI / 27.75 * t ) * cos(2 * (PI / 100) * dX / 2);
        t_z[1 * offset + i] = 0.01 * cos(2 * PI / 27.75 * t ) * cos(2 * (PI / 100) * (- dX) / 2);
        //if (t >= 6.75) printf("tz[%d, %d] = %.15f\n",2, i, t_z[2 * offset + i]);

	}

}

__global__ void boundary_right(DOUBLE t, int segment_limit, int M, int N, bool* bienQ, DOUBLE* t_z, int* daui, int* cuoii, 
	int* dauj, int* cuoij, DOUBLE* vbt, DOUBLE* vbd, DOUBLE* ubt, DOUBLE* ubp){
	
	int thrx = blockIdx.x * blockDim.x + threadIdx.x;
    int thry = blockIdx.y * blockDim.y + threadIdx.y;
    int i = thrx * (blockDim.y * gridDim.y) + thry + daui[2 * segment_limit];
	int offset = M + 3;
	if (i > cuoii[N * segment_limit]) return;
	if (bienQ[3])
		ubp[i] = 0;
	else{
		t_z[N * offset + i] = 0.01 * cos(2 * PI / 27.75 * t)  * cos(2 * (PI / 100) *  (100 - dX / 2));
        t_z[(N + 1) * offset + i] = 0.01 * cos(2 * PI / 27.75 * t)  * cos(2 * (PI / 100) *  (100 + dX / 2));
        //if (t >= 6.75) printf("tz[%d, %d] = %.15f\n",N, i, t_z[N * offset + i]);
	}
}



__global__ void preprocess_data(Argument_Pointers* arg){

	DOUBLE* hi = arg->hi;
	DOUBLE* h = arg->h;
	// DOUBLE* hsnham = arg->hsnham;
	int* daui = arg->daui;
	int* cuoii = arg->cuoii;
	int* dauj = arg->dauj;
	int* cuoij = arg->cuoij;
	int* moci = arg->moci;
	int* mocj = arg->mocj;
	int N = arg->N; int M = arg->M;

	// int* daui, *cuoii, *dauj, *cuoij, *moci, *mocj;
	// h[i, M]
	for (int k = 0; k < mocj[M]; k++){
		DOUBLE sum  = 0;
		for (int i = dauj[M * 5 + k]; i <= cuoij[M * 5 + k]; i++){
			hi[i] = (h[i * (M + 3) + M] + h[(i - 1) * (M + 3) + M]) / 2.0 - NANGDAY;
			// printf("h[%d], %f\n", i, h[i] );
			sum += powf(hi[i], 5.0/3.0);// / hsnham[i * (M + 3) + M];
		}

		for (int i = dauj[M * 5 + k]; i <= cuoij[M * 5 + k]; i++){
			hi[i] = powf(hi[i], 2.0 / 3.0) / (sum * dX ) ;//* hsnham[i * (M + 3) + M]);
		}

	}

	// h[i, 2]

	hi += N + 3;

	for (int k = 0; k < mocj[2]; k++){
		DOUBLE sum  = 0;
		// int sum = 0;
		for (int i = dauj[2 * 5 + k]; i <= cuoij[2 * 5 + k]; i++){
			hi[i] = (h[i * (M + 3) + 1] + h[(i - 1) * (M + 3) + 1]) / 2.0 - NANGDAY;
			sum += powf(hi[i], 5.0/3.0);// / hsnham[i * (M + 3) + 2];
		}

		for (int i = dauj[2 * 5 + k]; i <= cuoij[2 * 5 + k]; i++){
			hi[i] = powf(hi[i], 2.0 / 3.0) / (sum * dX) ; // * hsnham[i * (M + 3) + 2]);
		}

	}

	// // h[2, i]
	hi += N + 3;
	for (int k = 0; k < moci[2]; k++){
		DOUBLE sum = 0;
		// DOUBLE sum  = 0;
		for (int i = daui[2 * 5 + k]; i <= cuoii[2 * 5 + k]; i++){
			hi[i] = (h[1 * (M + 3) + i] + h[1 * (M + 3) + i - 1]) / 2.0 - NANGDAY;
			sum += powf(hi[i], 5.0/3.0);// / hsnham[2 * (M + 3) + i];
		}

		for (int i = daui[2 * 5 + k]; i <= cuoii[2 * 5 + k]; i++){
			hi[i] = powf(hi[i], 2.0 / 3.0) / (sum * dX); // * hsnham[2 * (M + 3) + i]);
		}

	}
	// h[N, i]

	hi += M + 3;
	for (int k = 0; k < moci[N]; k++){
		DOUBLE sum  = 0;
		for (int i = daui[N * 5 + k]; i <= cuoii[N * 5 + k]; i++){
			hi[i] = (h[N * (M + 3) + i] + h[N * (M + 3) + i - 1]) / 2.0 - NANGDAY;
			sum += powf(hi[i], 5.0/3.0);// / hsnham[N * (M + 3) + i];
		}
		for (int i = daui[N * 5 + k]; i <= cuoii[N * 5 + k]; i++){
			hi[i] = powf(hi[i], 2.0 / 3.0) / (sum * dX);// * hsnham[N * (M + 3) + i]);
		}

	}

}



__device__ void Boundary_value(bool isU, DOUBLE t, int location, int location_extension, int width, int total_time,
	int* boundary_type, DOUBLE* hi, DOUBLE* boundary_array, DOUBLE* t_z, DOUBLE* boundary_condition, int* moc, int* dau, int* cuoi){

	int t1 = t / 3600;
	// DOUBLE t2 = (t - (3600.0 * (DOUBLE) t1) ) / 3600.0 ;
	DOUBLE t2 = (t - (3600.0f * t1) ) / 3600.0f ;
	
	// locate which segment the thread is in charge of
	int i = blockIdx.y * blockDim.y + threadIdx.y + 2;

	int seg_no = - 1;
	for (int k = 0; k < moc[location]; k++){
		if ((dau[location * 5 +  k] <= i) && (i <= cuoi[location * 5 + k])) 
			seg_no = k;
		break;
	}

	// if there is no boundary in a certain edge
	if (seg_no == -1) return;
	// if (seg_no > 0) 
	// {
	// 	printf("seg_no is not 0\n");
	// 	// return;
	// }

	DOUBLE boundary_value = boundary_condition[seg_no * total_time + t1] * (1.0f - t2) + boundary_condition[seg_no * total_time + t1 + 1] * t2;
	// if (i == 158)
	// 		printf("111 tz[%d, %d], %llx %llx\n", i, location, boundary_value , boundary_condition[seg_no * total_time + t1 + 1]);
	// if boundary condition is given in Q	
	if (boundary_type[seg_no] == 2){
		boundary_array[i] = boundary_value * hi[i];
		if (t == 3.0 && threadIdx.x == 0) {
			// printf ("t:%f \t i:%d \t hi%.8f \t boundary_array:%.8f\n", t, i, hi[i], boundary_array[i]);
		}
	} else{
	// if boudary condition is given in Z
		if (isU){
		t_z[location * width + i] = boundary_value;
		t_z[location_extension * width  + i] = boundary_value;
		} else{
		// if (i == 158)
		// 	printf("222 tz[%d, %d], %llx\n", location, i, boundary_value );
		 t_z[i * width + location] = boundary_value;
		 t_z[i * width + location_extension] = boundary_value;
		}
	}
	
}

__device__ void FS_boundary(bool isU, DOUBLE t, int width, int total_time, int location, DOUBLE hmax,  DOUBLE* htaiz, DOUBLE* FS, DOUBLE* CC, int* moc, int* dau, int*cuoi ){
	int t1 = t / 3600;
	// DOUBLE t2 = (t - (3600.0 * (DOUBLE) t1) ) / 3600.0 ;
	DOUBLE t2 = (t - (3600.0f * t1) ) / 3600.0f ;
	// locate segment
	int i, j;
	i = blockIdx.y * blockDim.y + threadIdx.y + 2;

	int seg_no = - 1;
	for (int k = 0; k < moc[location]; k++){
		if ((dau[location * 5 +  k] <= i) && (i <= cuoi[location * 5 + k])) 
			seg_no = k;
		break;
	}
	if (seg_no == - 1 ) return;

	if (isU){
		j = location;
		i = blockIdx.y * blockDim.y + threadIdx.y + 2;
	} else{
		i = location;
		j = blockIdx.y * blockDim.y + threadIdx.y + 2;

	}

	DOUBLE boundary_value = CC[seg_no * total_time + t1] * (1.0f - t2)  + CC[seg_no * total_time + t1 + 1] * t2; 
	FS[i * width + j] = boundary_value * htaiz[i * width + j] * 1 / (ros  * hmax);
}

// block= (1, 1024, 1), grid= (1, max(M, N) // 1024 + 1, 1)

__device__ void find_hmax(int* hmax1, int* hmax2, int* hmax3, int* hmax4, Argument_Pointers* arg){
	// hmax1 
	DOUBLE * htaiz = arg->htaiz;
	int M = arg->M;
	int N = arg->N;
	int width = M + 3;
	*hmax1 = htaiz[2 * width + M];
	*hmax2 = htaiz[2 * width + 2];
	for (int i = 3; i < N; i ++){
		if (htaiz[i * width + M] > *hmax1)
			*hmax1 = htaiz[i * width + M];
		if (htaiz[i * width + 2] > *hmax2)
			*hmax2 = htaiz[i * width + 2];
	}

	*hmax3 = htaiz[2 * width + 2];
	*hmax4 = htaiz[N * width + 2];
	for (int i = 3; i < M; i ++){
		if (htaiz[2 * width + i] > *hmax3)
			*hmax3 = htaiz[i * width + M];
		if (htaiz[N * width + i] > *hmax4)
			*hmax4 = htaiz[i * width + 2];
	}
}

__global__ void Update_Boundary_Value(DOUBLE t, int total_time, Argument_Pointers* arg ){
	
	DOUBLE* hi_up, *hi_down, *hi_left, *hi_right;
	__shared__ int M, N;
	int* boundary_type;

	M = arg->M; N = arg->N;
	// if (threadIdx.y == 0) printf("add0 %ld\n", arg->Htdu);

	if ( (blockIdx.y * blockDim.y + threadIdx.y + 2 > M) && (blockDim.y * blockIdx.y + threadIdx.y + 2  > N)) return;
	// up

	// printf("%d %d %d\n", arg->mocj[M], arg->dauj[M * 5], arg->cuoij[M * 5]);	
	hi_up = arg->hi;
	boundary_type = arg->boundary_type;

	// printf("%d %d\n", arg->dauj[120 * 5], arg->cuoij[120 * 5] );
	Boundary_value(false, t, M, M + 1, M + 3, total_time, boundary_type, hi_up, arg->vbt, arg->t_z, arg->bc_up, arg->mocj, arg->dauj,  arg->cuoij);

	// down
	boundary_type += 5;
	hi_down = hi_up + (N + 3);
	Boundary_value(false, t, 2, 1, M + 3, total_time, boundary_type, hi_down, arg->vbd,arg->t_z, arg->bc_right, arg->mocj, arg->dauj,  arg->cuoij);

	// left
	boundary_type += 5;
	hi_left = hi_down + (N + 3);

	Boundary_value(true, t, 2, 1, M + 3, total_time, boundary_type, hi_left, arg->ubt, arg->t_z, arg->bc_left, arg->moci, arg->daui, arg->cuoii);

	// right

	boundary_type += 5;
	hi_right = hi_left + (M + 3);
	Boundary_value(true, t, N, N + 1, M + 3, total_time, boundary_type, hi_right, arg->ubp, arg->t_z, arg->bc_down, arg->moci, arg->daui, arg->cuoii);
	
	// if (threadIdx.y == 0) printf("add1 %ld\n", arg->Htdu);
	int hmax_u, hmax_d, hmax_l, hmax_r;
	find_hmax(&hmax_u, &hmax_d, &hmax_l, &hmax_r, arg);
	FS_boundary(false, t, M + 3, total_time, M, hmax_u, arg->htaiz, arg->FS, arg->CC_u, arg->mocj, arg->dauj,  arg->cuoij);
	FS_boundary(false, t, M + 3, total_time, 2, hmax_d, arg->htaiz, arg->FS, arg->CC_d, arg->mocj, arg->dauj,  arg->cuoij);
	FS_boundary(true, t, M + 3, total_time, 2, hmax_l, arg->htaiz, arg->FS, arg->CC_l, arg->moci, arg->daui,  arg->cuoii);
	FS_boundary(true, t, M + 3, total_time, N, hmax_r, arg->htaiz, arg->FS, arg->CC_r, arg->moci, arg->daui,  arg->cuoii);
}


__global__ void update_uvz(int M, int N, DOUBLE* u, DOUBLE* v, DOUBLE* z,  DOUBLE* t_u, DOUBLE* t_v, DOUBLE* t_z, DOUBLE* tmp_u, DOUBLE* tmp_v, int kenhhepd=0, int kenhhepng=0){
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if ((i >= N + 3) || ( j >= M + 3)) return;
	int offset = M + 3;
	// when updating u, v, z after solving vz, t_u and tmp_u are the same
	// when updating u, v, z after solving uz, t_v, and tmp_v are the same
	// t_u[i * offset + j] = tmp_u[i * offset + j];
	// t_v[i * offset + j] = tmp_v[i * offset + j];
	z[i * offset + j] = t_z[i * offset + j];
	u[i * offset + j] = t_u[i * offset + j] * (1 - kenhhepd);
	v[i * offset + j] = t_v[i * offset + j] * (1 - kenhhepng);
}

__device__ void _normalize (DOUBLE coeff, int N, int M, int closenb_dist, int farnb_dist, DOUBLE* tmp_buffer, DOUBLE* val_buff, int* khouot){
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int width = M + 3;
	int grid_pos = i * width + j;
	tmp_buffer[grid_pos] = val_buff[grid_pos];

	if (i > N || j > M || i < 2 || j < 2) return;
	DOUBLE neigbor = 0;
	if (val_buff[grid_pos] != 0){
		int count = 2;
		neigbor = val_buff[grid_pos - closenb_dist] + val_buff[grid_pos + closenb_dist];

		if (khouot[grid_pos - farnb_dist] == 0){ neigbor += val_buff[grid_pos - farnb_dist]; count++;}

		if (khouot[grid_pos + farnb_dist] == 0){ neigbor += val_buff[grid_pos + farnb_dist]; count++;}

		tmp_buffer[grid_pos] = tmp_buffer[grid_pos] * coeff + (1 - coeff) * neigbor / count;
	}

	// this one is for debugging only. need to change afterward.
}

__global__ void update_buffer(bool updateU, Argument_Pointers* arg, Array_Pointers* arr){
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int width = arg->M + 3;
	int grid_pos = i * width + j;
	if (i > arg->N || j > arg->M || i < 2 || j < 2) return;
	DOUBLE* tmp_buffer;
	DOUBLE* val_buffer;
	if (updateU){
		val_buffer = arg->t_u;
		tmp_buffer = arr->AA;
	} else{
		val_buffer = arg->t_v;
		tmp_buffer = arr->BB;
	}
	val_buffer[grid_pos] = tmp_buffer[grid_pos];
}

__global__ void Normalize(DOUBLE isU, Argument_Pointers* arg, Array_Pointers* arr){
	if (isU)
		_normalize(heso, arg->N, arg->M, arg->M + 3, 1, arr->AA, arg->t_u, arg->khouot);
	else
		_normalize(heso, arg->N, arg->M, 1, arg->M + 3, arr->BB, arg->t_v, arg->khouot);

}





/*** ATTENTION: this function assumes that values of h[i, 1] and h[i, 2] do not differ much, since with higher computation power
     we can do fine-grain-ly divide calculation mesh and so that h[i, 1] - h[i, 2] is negligible
***/
// __global__ void gpu_GiatriHtaiZ(Argument_Pointers *arg){
// 	int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
// 	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
// 	if ((i > N) || (j > M)) return;
// 	if (i == 1) 
// 		htaiz[width + j]  = (h[width + j - 1] + h[width + j]) * 0.5;
// 	else 
// 		htaiz[i * width + j] = (h[(i - 1) * width + j - 1] + h[(i - 1) * width + j] + h[i *width + j - 1] + h[i * width + j]) * 0.25;

// }

