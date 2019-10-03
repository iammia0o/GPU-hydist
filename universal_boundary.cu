#include "constant.cuh"
#include "cuda_runtime.h"

#define DOUBLE float


__global__ void preprocess_data(Argument_Pointers* arg){

	DOUBLE* hi = arg->hi;
	DOUBLE* h = arg->h;
	// DOUBLE* hsnham = arg->hsnham;
	int* daui = arg->daui;
	int* dauj = arg->dauj;
	int* cuoij = arg->cuoij;
	int* cuoii = arg->cuoii;
	int* moci = arg->moci;
	int* mocj = arg->mocj;
	int N = arg->N; int M = arg->M;

	// int* daui, *cuoii, *dauj, *cuoij, *moci, *mocj;
	// h[i, M]
	for (int k = 0; k < mocj[M]; k++){
		int sum  = 0;
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
		int sum  = 0;
		for (int i = dauj[2 * 5 + k]; i <= cuoij[2 * 5 + k]; i++){
			hi[i] = (h[i * (M + 3) + 2] + h[(i - 1) * (M + 3) + 2]) / 2.0 - NANGDAY;
			sum += powf(hi[i], 5.0/3.0);// / hsnham[i * (M + 3) + 2];
		}
		for (int i = dauj[2 * 5 + k]; i <= cuoij[2 * 5 + k]; i++){
			hi[i] = powf(hi[i], 2.0 / 3.0) / (sum * dX) ; // * hsnham[i * (M + 3) + 2]);
		}

	}

	// // h[2, i]
	hi += N + 3;
	for (int k = 0; k < moci[2]; k++){
		int sum  = 0;
		for (int i = daui[2 * 5 + k]; i <= cuoii[2 * 5 + k]; i++){
			hi[i] = (h[2 * (M + 3) + i] + h[2 * (M + 3) + i - 1]) / 2.0 - NANGDAY;
			sum += powf(hi[i], 5.0/3.0);// / hsnham[2 * (M + 3) + i];
		}
		for (int i = daui[2 * 5 + k]; i <= cuoii[2 * 5 + k]; i++){
			hi[i] = powf(hi[i], 2.0 / 3.0) / (sum * dX); // * hsnham[2 * (M + 3) + i]);
		}

	}
	// h[N, i]

	hi += M + 3;
	for (int k = 0; k < moci[N]; k++){
		int sum  = 0;
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
	DOUBLE t2 = t / 3600.0 - t1;

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
	// printf("%d %f\n", t1, t2);
	// printf("i = %d seg_no = %d dau = % d cuoi = %d location = %d \n", i, seg_no, dau[location * 5], cuoi[location * 5], location);


	DOUBLE boundary_value = boundary_condition[seg_no * total_time + t1] * (1 - t2) + boundary_condition[seg_no * total_time + t1 + 1];
	// printf("%d %f, %f\n",i, hi[i], boundary_condition[seg_no * total_time + t1]);

	// if boundary condition is given in Q	
	if (boundary_type[seg_no] == 2){
		boundary_array[i] = boundary_value * hi[i];
		// printf("%f, %f\n", hi[i], boundary_condition[seg_no * total_time + t1]);
	} else{
	// if boudary condition is given in Z
		if (isU){
		t_z[i * width + location] = boundary_value;
		t_z[i * width  + location_extension] = boundary_value;
		} else{
		 t_z[location * width + i] = boundary_value;
		 t_z[location_extension * width + i] = boundary_value;
		}
	}
	
}


__global__ void Update_Boundary_Value(DOUBLE t, int total_time, Argument_Pointers* arg ){
	
	__shared__ DOUBLE* hi_up, *hi_down, *hi_left, *hi_right;
	__shared__ int M, N;
	int* boundary_type;

	M = arg->M; N = arg->N;

	if ( (blockIdx.y * blockDim.y + threadIdx.y + 2 > M) || (blockDim.y * blockIdx.y + threadIdx.y + 2  > N)) return;
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
	hi_right = hi_left + (N + 3);
	Boundary_value(true, t, N, N + 1, M + 3, total_time, boundary_type, hi_right, arg->ubp, arg->t_z, arg->bc_down, arg->moci, arg->daui, arg->cuoii);

}



