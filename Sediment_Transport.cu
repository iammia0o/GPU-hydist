#include "constant.cuh"
#define DOUBLE double


// Source function for cohesive case
__device__ DOUBLE source_chs(DOUBLE wsm , DOUBLE Zf, DOUBLE fs, DOUBLE fw, DOUBLE vth, DOUBLE h_moi){
	DOUBLE toee = Toe;
	DOUBLE todd = Tod;
	DOUBLE S = 0;
	DOUBLE Pd, beta, Cb;
	// for Tan Chau
	DOUBLE tob = ro * fw * (vth * vth);

	if (h_moi > ghtoe)
	    toee = Toe * (1 + hstoe * ((h_moi - ghtoe)));
	if (tob > toee)
		S = Mbochat * (tob - toee) / toee;
	else{
		if (tob < todd){
			// this can be optimized
			Pd = 1 - (tob / todd);
	        beta = 1 + (Zf / (1.25 + 4.75 * pow(Pd , 2.5)));
	        Cb = beta * fs;
	        S = - wsm * Cb * Pd;
		}
	}
	return S;
}

// Source function for non cohesive case
__device__ DOUBLE source_nchs(DOUBLE wsm, DOUBLE Zf, DOUBLE Dxr, DOUBLE Ufr, DOUBLE Uf, DOUBLE h_moi, DOUBLE fs){
	DOUBLE S = 0;
	DOUBLE gamac = 0.434 + 5.975 * Zf + 2.888 * Zf * Zf;
    DOUBLE Tx = max(0.0, (Uf * Uf - Ufr * Ufr) / (Ufr * Ufr));

	DOUBLE Cac = 0.015 * dm * pow(Tx , 1.5) * pow(Dxr , (-0.3)) / (0.05 * h_moi);
    S = wsm * (Cac - gamac * fs);
	return S;
}

__device__ void _FSi_calculate__mactrix_coeff(bool ketdinh, int i, int j, int first, int last, int seg_no, bool bienran1, bool bienran2, Argument_Pointers* arg, Array_Pointers* arr){

	__shared__ DOUBLE* FS, *H_moi, *AA, *BB, *CC, *DD, *t_u, *t_v, *VTH, *Htdu, *Htdv, *Fw, *Kx, *Ky;
	__shared__ int width;
	// int sn = last - first - 2;
	width = arg-> M + 3;
	FS = arg->FS;
	H_moi = arg->H_moi;
	t_v = arg->t_v;
	t_u = arg->t_u;
	Htdv = arg->Htdv;
	Htdu = arg->Htdu;
	VTH = arg->VTH;
	Kx = arg->Kx;
	Ky = arg->Ky;
	Fw = arg->Fw;
	AA = arr->AA;
	BB = arr->BB;
	CC = arr->CC;
	DD = arr->DD;

	if (last < first + 1 || j > last - 1 || j < first + 1) return;
	int pos = i * width + j;
	int offset = first + 1;

	DOUBLE c = 18 * log(12 * H_moi[pos] / Ks);
	DOUBLE Uf = sqrt(g) * abs(VTH[pos]) / c;
	DOUBLE wsm = wss() *  pow((1 - FS[pos]), 4);
	DOUBLE Zf = 0;
	
	if (FS[pos] > 0.00001)
		Zf = wsm / (0.4 * (Uf + 2 * wsm));

	DOUBLE S;
	if (ketdinh)
		S = source_chs(wsm, Zf, FS[pos], Fw[pos], VTH[pos], H_moi[pos] );
	else
		S = source_nchs(wsm, Zf, Dxr(), Ufr(), Uf, H_moi[pos], FS[pos]);

	DOUBLE gamav = 0.98 - 0.198 * Zf + 0.032 * Zf * Zf;

    

	AA[offset + j] = -gamav * 0.5 * (t_v[pos - 1] + t_v[pos]) / dY2 - Htdv[pos - 1] * Ky[pos - 1] / (H_moi[pos] * dYbp);
	CC[offset + j] = gamav * 0.5 * (t_v[pos - 1] + t_v[pos]) / dY2 - Htdv[pos] * Ky[pos] / (H_moi[pos] * dYbp);
	BB[offset + j] = HaiChiadT + (Htdv[pos] * Ky[pos] + Htdv[pos - 1] * Ky[pos - 1]) / (H_moi[pos] * dYbp);

	DOUBLE p, q;
	p = q = 0;
	if ((H_moi[pos + 2 * width] > H_TINH) &&  (H_moi[pos - width] > H_TINH))
	    p = (FS[pos + width] - FS[pos]);

	if  (H_moi[pos- 2 * width] > H_TINH && H_moi[pos + width] > H_TINH)
	    q = (FS[pos] - FS[pos - width]);

	p = (1 / (H_moi[pos] * dXbp)) * (Htdu[pos] * Kx[pos] * p - Htdu[pos - width] * Kx[pos - width] * q);

	if (H_moi[pos + 2 * width] > H_TINH && H_moi[pos - 2 * width] > H_TINH)
	    q = (FS[pos + width] - FS[pos - width]) / dX2;
	q = (t_u[pos] + t_u[pos - width]) * 0.5  * q * gamav;


	DD[offset + j] = FS[pos] / (dT * 0.5) - q + p + (S / H_moi[pos]);

	if (j == first + 1){
		if ((bienran1) || (t_v[i * width + first] == 0) ){
			BB[offset] = BB[offset] + AA[offset];
		} else{
			DD[offset] = DD[offset] - AA[offset] * FS[i * width + first];
		}
	}
		// this is likely to be changed
	if (j == last - 1){
		int sn = last - first - 1;

		if ((bienran2) || (t_v[i * width + last] == 0) ){
			// For Tan Chau
			BB[sn + offset] = BB[sn + offset] + CC[sn + offset];
		} else{
			DD[sn + offset] = DD[sn + offset] - CC[sn + offset] * FS[i * width + last];
		}
		arr->SN[i * 5 + seg_no] = sn;
	}
}

__device__ void _FSi_extract_solution(int i, int j, int first, int last, bool bienran1, bool bienran2, Argument_Pointers* arg, Array_Pointers* arr){

	__shared__ DOUBLE* FS, *x, *t_v;
	__shared__ int width;
	width = arg->M + 3;
	x = arr->x;
	FS = arg->FS;
	t_v = arg->t_v;
	int pos = i * width + j;
	if (j > last - 1 || j < first + 1 || last < first + 1)
		return;
	int offset = first + 1;

	if (x[j + offset] < 0) 
		x[j + offset] = NDnen;
	FS[pos] = x[j + offset];
	if (j == first + 1){

		if ((bienran1) || (t_v[i * width + first] == 0))
			FS[i * width + first] = FS[i * width + first + 1];
		else 
			FS[i * width + first - 1] = FS[i * width + first];
	}
	if (j == last - 1){
		if ((bienran2) || (t_v[i * width + last] == 0))
			FS[i * width + last] = FS[i * width + last - 1];
		else 
			FS[i * width + last - 1] = FS[i * width + last];
	}
}

__device__ void _FSj_calculate__mactrix_coeff(bool ketdinh, int i, int j, int first, int last, int seg_no, bool bienran1, bool bienran2, Argument_Pointers* arg, Array_Pointers* arr){
	if (i > last - 1 || i < first + 1 || last < first + 1)
		return;
	__shared__ DOUBLE* FS, *H_moi, *AA, *BB, *CC, *DD, *t_u, *t_v, *VTH, *Htdu, *Htdv, *Fw, *Kx, *Ky;
	__shared__ int width;
	// int sn = last - first - 2;
	width = arg-> M + 3;
	FS = arg->FS;
	H_moi = arg->H_moi;
	t_v = arg->t_v;
	t_u = arg->t_u;
	Htdv = arg->Htdv;
	Htdu = arg->Htdu;
	VTH = arg->VTH;
	Kx = arg->Kx;
	Ky = arg->Ky;
	Fw = arg->Fw;
	AA = arr->AA;
	BB = arr->BB;
	CC = arr->CC;
	DD = arr->DD;
	if (last < first + 1 || i > last - 1 || i < first + 1) return;
	int pos = i * width + j;

	DOUBLE c = 18 * log(12 * H_moi[pos] / Ks);
	DOUBLE Uf = sqrt(g) * abs(VTH[pos]) / c;
	DOUBLE wsm = wss() *  pow((1 - FS[pos]), 4);
	DOUBLE Zf = 0;
	DOUBLE S = 0;
	int offset = first + 1;

	DOUBLE gamav = 0.98 - 0.198 * Zf + 0.032 * Zf * Zf;

	if (FS[pos] > 0.00001)
		Zf = wsm / (0.4 * (Uf + 2 * wsm));
	
	if (ketdinh)
		S = source_chs(wsm, Zf, FS[pos], Fw[pos], VTH[pos], H_moi[pos] );
	else
		S = source_nchs(wsm, Zf, Dxr(), Ufr(), Uf, H_moi[pos], FS[pos]);        
        
    AA[offset + i] = -gamav * 0.5 * (t_u[pos] + t_u[pos - width]) / dX2 - Htdu[pos - width] * Kx[pos - width] / (H_moi[pos] * (dXbp));
    CC[offset + i] = gamav * 0.5 * (t_u[pos] + t_u[pos - width]) / dX2 - Htdu[pos] * Kx[pos] / (H_moi[pos] * (dXbp));
    BB[offset + i] = HaiChiadT + (Htdu[pos] * Kx[pos] + Htdu[pos - width] * Kx[pos - width]) / (H_moi[pos] * (dXbp));
    DOUBLE p, q;
    p = q = 0;
    if ((H_moi[pos - 1] > H_TINH) && (H_moi[pos + 2] > H_TINH))
        p = (FS[pos + 1] - FS[pos]);
    if ((H_moi[pos - 2] > H_TINH) && (H_moi[pos + 1] > H_TINH))
        q = (FS[pos] - FS[pos - 1]);

    p = (1 / (H_moi[pos] * dYbp)) * (Htdv[pos] * Ky[pos] * p - Htdv[pos - 1] * Ky[pos - 1] * q);

    if ((H_moi[pos - 2] > H_TINH) && (H_moi[pos + 2] > H_TINH))
        q = (FS[pos + 1] - FS[pos - 1]) / dY2;

    q = (t_v[pos] + t_v[pos - 1]) * 0.5 * q * gamav;

    DD[offset + i] = FS[pos] / (dT * 0.5) - q + p + (S / H_moi[pos]);

    if (i == first + 1){
	    if ((bienran1) || (t_u[first * width + j] == 0))
	    	// this is for song luy only
	    	BB[offset] += AA[offset];
	    else
	    	DD[offset] -= AA[offset] * FS[first * width + j];
	}    
	if (i == last - 1){
		int sn = last - first - 1;
		if ((bienran2) && (t_u[last * width + j] == 0)){
	    	BB[sn + offset] += CC[sn + offset];
	    } else
	    	DD[sn + offset] -= CC[sn + offset] * FS[last * width + j];
		arr->SN[j * 5 + seg_no] = sn;

	}

}

__device__ void _FSj_extract_solution(int i, int j, int first, int last, bool bienran1, bool bienran2, Argument_Pointers* arg, Array_Pointers* arr){

	__shared__ DOUBLE* FS, *x, *t_u;
	__shared__ int width;
	width = arg->M + 3;
	x = arr->x;
	FS = arg->FS;
	t_u = arg->t_u;
	int pos = i * width + j;
	if (i > last - 1 || i < first + 1 || last < first + 1)
		return;
	int offset = first + 1;
	if (x[i + offset] < 0) 
		x[i + offset] = NDnen;
	FS[pos] = x[i + offset];

	if (i == first + 1){
	    if ((bienran1) || (t_u[pos - width] == 0))
	    	// this is for song luy only
	    	FS[pos - width] = FS[pos];
	    else
	    	FS[pos - 2 * width] = FS[pos - width];
	} 

	if (i == last - 1){
		if ((bienran2) || (t_u[pos + width] == 0)){
			FS[pos + width] = FS[pos];
	    } else
	    	FS[pos + 2 * width] = FS[pos + width];
	}
}

__device__ void _calculate_Qb(bool ketdinh, int i, int j, int first, int last, bool bienran1, bool bienran2, Argument_Pointers* arg, Array_Pointers* arr){
	DOUBLE Tob;
	DOUBLE Toee = Toe;
	DOUBLE Tx = 0;
	__shared__ int* khouot;
	__shared__ DOUBLE *H_moi, *t_u, *t_v, *VTH, *Qbx, *Qby, *Fw;
	__shared__ int width, M, N;
	M = arg->M;
	N = arg->N;
	width = M + 3;
	H_moi = arg->H_moi;
	Fw = arg->Fw;
	Qbx = arg->Qbx;
	Qby = arg->Qby;
	t_v = arg->t_v;
	t_u = arg->t_u;
	VTH = arg->VTH;	
	khouot = arg->khouot;

	if (i > N || j > M || i < 2 || j < 2)
		return;
	int pos = i * width + j;
	if ((!VTH[pos]) && (khouot[pos] == 0)){
      
	    Tob = ro * Fw[pos] * (VTH[pos] * VTH[pos]);
	    if (H_moi[pos] > ghtoe)
	        Toee = Toe * (1 + hstoe * ((H_moi[pos] - ghtoe)));
	    
	    if (Tob > Toee)
	        Tx = Tob / Toee - 1;
    	Qbx[pos] = 0.053 * pow( (Sx - 1) * g , 0.5) * pow(dm , 1.5) * pow(Tx , 2.1) * pow(Dxr() , -0.3) * (t_u[pos] + t_u[pos - width]) * 0.5 / VTH[pos];
    	Qby[pos] = 0.053 * pow( (Sx - 1) * g , 0.5) * pow(dm , 1.5) * pow(Tx , 2.1) * pow(Dxr() , -0.3) * (t_v[pos] + t_v[pos - 1]) * 0.5 / VTH[pos];
    }
    else {
    	Qbx[pos] = 0;
    	Qby[pos] = 0;
    }
    if (i == 2){
    	Qbx[pos - width] = Qbx[pos];
    	Qby[pos - width] = Qby[pos];
    	
    }
    if (i == N){
    	Qbx[pos + width] = Qbx[pos];
    	Qby[pos + width] = Qby[pos];
    	
    }
    if (j == 2){
    	Qbx[pos - 1] = Qbx[pos];
    	Qby[pos - 1] = Qby[pos];
    	if (i == 2)
    		Qbx[1 * width + 1] = Qbx[pos];
    		Qby[1 * width + 1] = Qby[pos];
    	if (i == N)
    		Qbx[(N + 1 ) * width + 1] = Qbx[pos];
    		Qby[(N + 1 ) * width + 1] = Qby[pos];
    }
    if (j == M){
    	Qbx[pos + 1] = Qbx[pos];
    	Qby[pos + 1] = Qby[pos];
    	if (i == 2){
    		Qbx[1 * width + M + 1] = Qbx[pos];
    		Qby[1 * width + M + 1] = Qby[pos];
    	}
    	if (i == N)
    	{
    		Qbx[(N + 1) * width + M + 1] = Qbx[pos];
    		Qby[(N + 1) * width + M + 1] = Qby[pos];
    	}
    }

}

__device__ void _bed_load(DOUBLE t, bool ketdinh, int i, int j, int first, int last, bool bienran1, bool bienran2, Argument_Pointers* arg, Array_Pointers* arr){
	if (last - first < 2 || j < first || j > last)
		return;
	__shared__ DOUBLE* FS, *H_moi, *t_u, *t_v, *VTH, *Htdu, *Htdv, *Fw, *Kx, *Ky, *Qbx, *Qby, *htaiz, *dH ;
	__shared__ int width, *khouot, M, N;
	M = arg->M;
	N = arg->N;
	width = arg-> M + 3;
	FS = arg->FS;
	H_moi = arg->H_moi;
	t_u = arg->t_u;
	t_v = arg->t_v;
	Htdv = arg->Htdv;
	Htdu = arg->Htdu;
	VTH = arg->VTH;
	Kx = arg->Kx;
	Ky = arg->Ky;
	Fw = arg->Fw;
	Qbx = arg->Qbx;
	Qby = arg->Qby;
	htaiz = arg->htaiz;
	khouot = arg->khouot;
	dH = arg->dH;
	int pos = i * width + j;
	DOUBLE p, q;
	p = q = 0;
	if (khouot[pos - width] == 1){
		if (t_u[pos] < 0)
            p = (-3 * Qbx[pos] + 4 * Qbx[pos + width] - Qbx[pos + 2 * width]) / dX2 ;

	} else {
		if (khouot[pos + width] == 1){
			if (t_u[pos > 0])
                p = (3 * Qbx[pos] - 4 * Qbx[pos - width] + Qbx[pos - 2 * width]) / dX2;
		} else 
            p = (Qbx[pos + width] - Qbx[pos - width]) / dX2;
	}
    
    if (khouot[pos - 1] == 1) {
        if (t_v[pos] < 0) 
            q = (-3 * Qby[pos] + 4 * Qby[pos + 1] - Qby[pos + 2]) / dY2;
    } else {
        if (khouot[pos + 1] == 1) {
            if (t_v[pos] > 0) 
                q = (3 * Qby[pos] - 4 * Qby[pos - 1] + Qby[pos - 2]) / dY2;
        }  else
            q = (Qby[pos + 1] - Qby[pos - 1]) / dY2;
    }
        
    DOUBLE tH = p + q;
    p = FS[pos + width] - FS[pos];

    if ((khouot[pos + 2 * width] == 1 || khouot[pos - width] == 1) && (i < N)) 
        p = 0;
    q = FS[pos] - FS[pos - width];
    if ((i > 2) && (khouot[pos - 2 * width] == 1 || khouot[pos + width] == 1)) 
        q = 0;
        
    p = 1 / dXbp * (Htdu[pos] * Kx[pos] * p - Htdu[pos - width] * Kx[pos - width] * q);
        
    tH = tH + p;
	p = FS[pos + 1] - FS[pos];

	if (((last < M) && (khouot[pos - 1] == 1 || khouot[pos + 2] == 1)) ||
		(last == M && (khouot[pos - 1] == 1)))
			p = 0;
    q = FS[pos] - FS[pos - 1];
    if (((first > 2) && (khouot[pos - 2] == 1 || khouot[pos + 1] == 1)) || (first == 2 && khouot[pos + 1] == 1))
    	q = 0;
                 
    p = 1 / (dYbp) * (Htdv[pos] * Ky[pos] * p - Htdv[pos - 1] * Ky[pos - 1] * q);
    
    tH = tH + p;

    
    DOUBLE c = 18 * log(12 * H_moi[pos] / Ks);
	DOUBLE Uf = sqrt(g) * abs(VTH[pos]) / c;
	DOUBLE wsm = wss() * pow((1 - FS[pos]), 4);
	DOUBLE Zf = 0;
	DOUBLE S = 0;
	if (FS[pos] > 0.0)
		Zf = wsm / (0.4 * (Uf + 2 * wsm));
	
	if (ketdinh)
		S = source_chs(wsm, Zf, FS[pos], Fw[pos], VTH[pos], H_moi[pos] );
	else
		S = source_nchs(wsm, Zf, Dxr(), Ufr(), Uf, H_moi[pos], FS[pos]); 

    dH[pos] = dT / (1 - Dorong) * (S + tH) + dH[pos];   
    
}

__global__ void hesoK(Argument_Pointers* arg){
	
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x  + threadIdx.x;
	int M, N;
	M = arg->M;
	N = arg->N;
	if (i > N || j > M)
		return;
	DOUBLE Cz = 0;
	__shared__ DOUBLE *Kx, *Ky, *Htdu, *Htdv, *t_u, *t_v, *h;
	Kx = arg->Kx;
	Ky = arg->Ky;
	Htdu = arg->Htdu;
	Htdv = arg->Htdv;
	t_u = arg->t_u;
	t_v = arg->t_v;
	h = arg->h;
	__shared__ int width;
	width = M + 3;
	int pos = i * width + j;
	if (Htdu[pos] > 0){
		Cz = 7.8 * log(12 * Htdu[pos] / Ks);
		Kx[pos] = 5.93 * sqrt(g) * (h[pos - 1] + h[pos])* 0.25 * abs(t_u[pos]) / Cz;
		Kx[pos] = min(100.0, max(5.0, Kx[pos]));
	}
	if (Htdv[pos] > 0){
		Cz = 7.8 * log(12 * Htdv[pos] / Ks);
		Ky[pos] = 5.93 * sqrt(g) * (h[pos - width] + h[pos])* 0.25 * abs(t_v[pos]) / Cz;
		Ky[pos] = min(100.0, max(5.0, Ky[pos]));
	}
}

__global__ void VTH(Argument_Pointers* arg){
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x  + threadIdx.x;
	int M, N;
	M = arg->M;
	N = arg->N;
	if ( i < 2 || j < 2 || i > M || j > N)
		return;
	__shared__ DOUBLE * H_moi, *htaiz, *VTH, *t_u, *t_v;
	__shared__ int width;
	width = M + 3;
	H_moi = arg->H_moi;
	htaiz = arg->htaiz;
	t_u = arg->t_u;
	t_v = arg->t_v;
	VTH = arg->VTH;
	int pos = i * width + j;
	if (htaiz[pos] > NANGDAY && H_moi[pos] > H_TINH){
		DOUBLE ut = (t_u[pos] + t_u[pos - width]) * 0.5;
		DOUBLE vt = (t_v[pos] + t_v[pos - 1]) * 0.5;
		VTH[pos] = sqrt(ut * ut + vt * vt);
	} else VTH[pos] = 0;
}

__global__ void Scan_FSi(bool ketdinh, int startidx, int endidx, Argument_Pointers* arg, Array_Pointers* arr){
// find first, last, bienran1, bienran2, i, j, pass argument
	int i = blockIdx.y * blockDim.y + threadIdx.y + startidx;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 2;
    if (i >= endidx ) return;
    bool bienran1 = false;
    bool bienran2 = false;
    int first = 0; int last = 0;
    int seg_no = locate_segment_v(arg->N, arg->M, &bienran1, &bienran2, &first, &last, i, j, arg->daui, arg->cuoii, arg->moci, arg->h);
    _FSi_calculate__mactrix_coeff(ketdinh, i, j, first, last, seg_no, bienran1, bienran2, arg, arr);
}

__global__ void Scan_FSj(bool ketdinh, int startidx, int endidx, Argument_Pointers* arg, Array_Pointers* arr){
// find first, last, bienran1, bienran2, i, j, pass argument
	int i = blockIdx.y * blockDim.y + threadIdx.y + 2;
    int j = blockIdx.x * blockDim.x + threadIdx.x + startidx;
    // if (i == 2 && j == startidx)
    //     printf("add0 %x\n", arg->Htdu);
    if (j >= endidx ) return;
    bool bienran1 = false;
    bool bienran2 = false;
    int first = 0; int last = 0;
    int seg_no = locate_segment_u(arg->N, arg->M, &bienran1, &bienran2, &first, &last, i, j, arg->dauj, arg->cuoij, arg->mocj, arg->h);
    _FSj_calculate__mactrix_coeff(ketdinh, i, j, first, last, seg_no, bienran1, bienran2, arg, arr);
}

__global__ void FSj_extract_solution(bool ketdinh, int startidx, int endidx, Argument_Pointers* arg, Array_Pointers* arr){
	int i = blockIdx.y * blockDim.y + threadIdx.y + startidx;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 2;
    if (i >= endidx ) return;
    bool bienran1 = false;
    bool bienran2 = false;
    int first = 0; int last = 0;
    locate_segment_u(arg->N, arg->M, &bienran1, &bienran2, &first, &last, i, j, arg->daui, arg->cuoii, arg->moci, arg->h);
    _FSj_extract_solution(i, j, first, last, bienran1, bienran2, arg, arr);
}

__global__ void FSi_extract_solution( bool ketdinh, int startidx, int endidx, Argument_Pointers* arg, Array_Pointers* arr){
	int i = blockIdx.y * blockDim.y + threadIdx.y + startidx;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 2;
    if (i >= endidx ) return;
    bool bienran1 = false;
    bool bienran2 = false;
    int first = 0; int last = 0;
    locate_segment_v(arg->N, arg->M, &bienran1, &bienran2, &first, &last, i, j, arg->daui, arg->cuoii, arg->moci, arg->h);
    _FSi_extract_solution(i, j, first, last, bienran1, bienran2, arg, arr);

}

__global__ void BedLoad(DOUBLE t, bool ketdinh, int startidx, int endidx, Argument_Pointers* arg, Array_Pointers* arr){
	int i = blockIdx.y * blockDim.y + threadIdx.y + startidx;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 2;
    if (i >= endidx ) return;
    bool bienran1 = false;
    bool bienran2 = false;
    int first = 0; int last = 0;
    locate_segment_v(arg->N, arg->M, &bienran1, &bienran2, &first, &last, i, j, arg->daui, arg->cuoii, arg->moci, arg->h);
    _bed_load(t, ketdinh, i, j, first, last, bienran1, bienran2, arg, arr);
}