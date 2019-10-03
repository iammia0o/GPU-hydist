
#define DOUBLE double
#include "spike_kernel.hxx"
// m is size of the matrix, which is SN
__device__ void findBestGrid( int m, int tile_marshal, int *p_m_pad, int *p_b_dim, int *p_s, int *p_stride)
{
    int b_dim, m_pad, s, stride;
    int B_DIM_MAX, S_MAX;
    
    if ( sizeof(DOUBLE) == 4) {
        B_DIM_MAX = 256;
        S_MAX     = 512;    
    }
    else if (sizeof(DOUBLE) == 8){ /* double and complex */
        B_DIM_MAX = 128;
        S_MAX     = 256;     
    }
    
    /* b_dim must be multiple of 32 */
    if ( m < B_DIM_MAX * tile_marshal ) {
        b_dim = max( 32, (m/(32*tile_marshal))*32);
        s = 1;
        m_pad = ((m + b_dim * tile_marshal -1)/(b_dim * tile_marshal)) * (b_dim * tile_marshal);
        stride = m_pad/(s*b_dim);    
    }
    else {
        b_dim = B_DIM_MAX;
        
        s = 1;
        do {       
            int s_tmp = s * 2;
            int m_pad_tmp = ((m + s_tmp*b_dim*tile_marshal -1)/(s_tmp*b_dim*tile_marshal)) * (s_tmp*b_dim*tile_marshal);           
            float diff = (float)(m_pad_tmp - m)/float(m);
            /* We do not want to have more than 20% oversize */
            if ( diff < .2 ) {
                s = s_tmp;      
            }
            else {
                break;
            }
        } while (s < S_MAX);
                       
        m_pad = ((m + s*b_dim*tile_marshal -1)/(s*b_dim*tile_marshal)) * (s*b_dim*tile_marshal);        
        stride = m_pad/(s*b_dim);
    }
      
    *p_stride = stride;
    *p_m_pad  = m_pad;
    *p_s      = s;
    *p_b_dim  = b_dim;        
}

void dtsvb_spike_v1(const DOUBLE* dl, const DOUBLE* d, const DOUBLE* du, DOUBLE* b,const int m)
{



	// cudaFuncSetCacheConfig(thomas_v1<DOUBLE>,cudaFuncCachePreferL1);
	// cudaFuncSetCacheConfig(spike_GPU_back_sub_x1<DOUBLE>,cudaFuncCachePreferL1);
	//parameter declaration
	int s; //griddim.x
	int stride;
	int b_dim;
    int m_pad;

	int tile_marshal = 16;
	int T_size = sizeof(DOUBLE);
	
    findBestGrid<DOUBLE>( m, tile_marshal, &m_pad, &b_dim, &s, &stride);
   
    printf("m=%d m_pad=%d s=%d b_dim=%d stride=%d\n", m, m_pad, s, b_dim, stride);

	
	int local_reduction_share_size = 2*b_dim*3*T_size;
	int global_share_size = 2*s*3*T_size;
	int local_solving_share_size = (2*b_dim*2+2*b_dim+2)*T_size;
	int marshaling_share_size = tile_marshal*(tile_marshal+1)*T_size;
	
	
	dim3 g_data(b_dim/tile_marshal,s);
	dim3 b_data(tile_marshal,tile_marshal);
	

    DOUBLE* dl_buffer;   //dl buffer
	DOUBLE* d_buffer;    //b
	DOUBLE* du_buffer; 
	DOUBLE* b_buffer;
	DOUBLE* w_buffer;
	DOUBLE* v_buffer;
	DOUBLE* c2_buffer;
	
	DOUBLE* x_level_2;
	DOUBLE* w_level_2;
	DOUBLE* v_level_2;
	
	
	//buffer allocation
	cudaMalloc((void **)&dl_buffer, T_size*m_pad); 
	cudaMalloc((void **)&d_buffer, T_size*m_pad); 
	cudaMalloc((void **)&du_buffer, T_size*m_pad); 
	cudaMalloc((void **)&b_buffer, T_size*m_pad); 
	cudaMalloc((void **)&w_buffer, T_size*m_pad); 
	cudaMalloc((void **)&v_buffer, T_size*m_pad); 
	cudaMalloc((void **)&c2_buffer, T_size*m_pad); 
	
	cudaMalloc((void **)&x_level_2, T_size*s*2); 
	cudaMalloc((void **)&w_level_2, T_size*s*2); 
	cudaMalloc((void **)&v_level_2, T_size*s*2); 
	
	
	
	//kernels 
	
	//data layout transformation
	foward_marshaling_bxb<DOUBLE><<<g_data ,b_data, marshaling_share_size >>>(dl_buffer, dl, stride, b_dim, m, cuGet<DOUBLE>(0));
	foward_marshaling_bxb<DOUBLE><<<g_data ,b_data, marshaling_share_size >>>(d_buffer,  d,  stride, b_dim, m, cuGet<DOUBLE>(1));
	foward_marshaling_bxb<DOUBLE><<<g_data ,b_data, marshaling_share_size >>>(du_buffer, du, stride, b_dim, m, cuGet<DOUBLE>(0));
	foward_marshaling_bxb<DOUBLE><<<g_data ,b_data, marshaling_share_size >>>(b_buffer,  b,  stride, b_dim, m, cuGet<DOUBLE>(0));
	 
	//partitioned solver
	thomas_v1<DOUBLE><<<s,b_dim>>>(b_buffer, w_buffer, v_buffer, c2_buffer, dl_buffer, d_buffer, du_buffer, stride);
	
	
	//SPIKE solver
	spike_local_reduction_x1<DOUBLE><<<s,b_dim,local_reduction_share_size>>>(b_buffer,w_buffer,v_buffer,x_level_2, w_level_2, v_level_2,stride);
	spike_GPU_global_solving_x1<<<1,32,global_share_size>>>(x_level_2,w_level_2,v_level_2,s);
	spike_GPU_local_solving_x1<DOUBLE><<<s,b_dim,local_solving_share_size>>>(b_buffer,w_buffer,v_buffer,x_level_2,stride);
	spike_GPU_back_sub_x1<DOUBLE><<<s,b_dim>>>(b_buffer,w_buffer,v_buffer, x_level_2,stride);

	back_marshaling_bxb<DOUBLE><<<g_data ,b_data, marshaling_share_size >>>(b,b_buffer,stride,b_dim, m);
	
	//free
	
	cudaFree(dl_buffer);
	cudaFree(d_buffer);
	cudaFree(du_buffer);
	cudaFree(b_buffer);
	cudaFree(w_buffer);
	cudaFree(v_buffer);
	cudaFree(c2_buffer);
	cudaFree(x_level_2);
	cudaFree(w_level_2);
	cudaFree(v_level_2);
				
}




/* explicit instanciation */
template void dtsvb_spike_v1<float>(const float* dl, const float* d, const float* du, float* b,const int m);
template void dtsvb_spike_v1<double>(const double* dl, const double* d, const double* du, double* b,const int m);
template void dtsvb_spike_v1<cuComplex>(const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* b,const int m);
template void dtsvb_spike_v1<cuDoubleComplex>(const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* b,const int m);