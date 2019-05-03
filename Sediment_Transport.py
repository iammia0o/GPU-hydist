import numpy as np 
import math
from Reynold_Equation_Solver import truyduoi

# Solve FS in i or j direction. isI indicate the direction
# If solving in i direction, i is fixed and itr iterate along j direction
# If solcing in j direction, j is fixed and itr iterate along i direction
def FS_Solver(isI, pos, l, r, somecoefficient):

    i = pos
    if isI: 
        uvt = ut 
        tuv = tv
        Htd = Htdu
        Kxy = Kx    
        dXYbp = dYbp
        dXY = dX
        Hnew = H_moiP
    else: 
        uvt = np.transpose(vt)
        tuv = np.transpose(tu)
        Hdt = np.transpose(Htdv)
        Kxy = np.transpose(Ky)
        dXYbp = dXbp
        dXY = dY
        Hnew =  np.transpose(H_moi)
        # tFS
        #FS
        #VTH
        #Fw

    if r - l > 1:
        for itr in range (l+1, r):

            c = somecoefficient * log(12* Hnew[i, itr] / Ks)

            Uf = sqrt(g) * abs(VTH[i, itr]) / c

            wsm = wss * pow(1 - fs / ros, 4) 

            Zf = 0
            if fs > 1e-5:
                Zf = wsm / (0.4 * (Uf + 2 * wsm))
            gamav = 0.98 - 0.198 * Zf + 0.032 * pow( Zf, 2)

            Tob = ro * Fw[i, itr] * pow(VTH[i, itr], 2)
            if Hnew[i, itr] > ghtoe:
                Toee = Toe * (1 + hstoe * (Hnew[i, itr] - ghtoe))
            else:
                Toee = toe

            Todd = Tod 

            if Tob > Toe:
                S = Mbochat * (Tob - Toee) / Toee
            elif Tob < Todd:
                Pd = 1 - Tob / Todd
                beta = 1 + (Zf / (1.25 + 4.75 * pow(Pd, 2.5)))
                Cb = beta * FS[i, itr]

                S = -wsm * Cb * Pd
            else:
                S = 0

            AA[itr - l] = -gamav * uvt[i, itr] / dY2 - Htd[i, itr - 1] * Kxy[i, itr - 1] / (Hnew[i, itr] * dXYbp)
            BB[itr - l] = HaiChiadT + (Htd[i, itr] * Kxy[i, itr] + Htd[i, itr - 1] * Kxy[i, itr - 1]) / (Hnew[i, itr] * dXYbp)
            CC[itr - l] = gamav * uvt[i, itr] / dY2 - Htd[i, itr] * Kxy[i, itr] / (Hnew[i, itr] * dXYbp)

            if Hnew[i + 2, itr] <= H_TINH or Hnew[i - 1, itr] <= H_TINH:
                tmp1 = 0
            else:
                tmp1 = (FS[i + 1, itr] - FS[i, itr]) / dXY
            if Hnew[i - 2, itr] <= H_TINH or Hnew[i + 1, itr] <= H_TINH:
                tmp2 = 0
            else:
                tmp2 = (FS[i, itr] - FS[i - 1, itr]) / dXY
            tmp1 = (1 / (Hnew[i, j] * dXYbp)) * (Htd[i, j] * Kxy[i, j] * tmp1 - Htd[i - 1, j] * Kxy[i - 1, j] * tmp2)
            tmp2 = 0
            if (Hnew[i + 2, itr] >= H_TINH and Hnew[i - 2, itr] >= H_TINH):
                tmp2 =  (FS[i + 1, itr] - FS[i -1 , itr]) / (2*dXY)
            tmp2 *= uvt[i, itr] * gamav
            DD[itr - l] = FS[i, itr] / (dT * 0.5) - tmp2 + tmp1 + (S / Hnew[i, itr])

        if (bienran1 is False) and tuv[i, l] > 0:
            DD[1] -= - AA[1] * tFS[i, l]
        else:
            BB[1] += AA[1]

        if bienran2 is False and tuv[i, r] <0:
            DD[r - l - 1] -= CC[r - l - 1] * tFS[i, r]
        else:
            BB[r - l - 1] += CC[r - l -1]

        truyduoi(r - l -1)

        for itr in range(l+1, r):
            if x[itr - l] < 0:
                x[itr - l] = NDnen
                if isI: 
                    FS[i, itr] = x[itr - l]
                else:
                    FS[itr, i] = x[itr - l]

    if bienran1 is False and tuv[i, l] < 0:
        FS[i, l] = tFS[i, l]
    else:
        FS[i, l] = tFS[i, l + 1]

    if bienran2 is False and tuv[i, r] <0:
        FS[i, r] = tFS[i, r]
    else:
        FS[i, r] = FS[i, r - 1]

    if isI is False:
        vt = np.transpose(uvt)
        tu = np.transpose(tuv)
        Htdv = np.transpose(Htd)
        Ky = np.transpose(Kxy)
        H_moi = np.transpose(Hnew)

    return(Toee, toe)


    

    
    
    
    
    
    
    
   



        

