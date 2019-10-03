
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

def visualize(U, V, savefile=False, stt=0):
    # im = plt.imshow(U)
    # plt.show()
    # im = plt.imshow(V)
    # plt.show()
    n, m = U.shape
    fig = plt.figure(num=None, figsize=(10, 10), facecolor='w', edgecolor='k')
    X, Y = np.mgrid[0:n, 0:m]
    EE = np.sqrt(U * U + V * V)
    cax = plt.axis('equal')
    plt.quiver(X, Y, U, V, alpha=.5)
    normalize = Normalize ()
    cmap=normalize (EE.flatten ())
    normalize.autoscale (EE.flatten ())
    im = plt.quiver(X[::4,::4], Y[::4,::4], U[::4,::4], V[::4,::4], EE[::4,::4], scale=20, headwidth=4, pivot='tail', angles='uv')
    if savefile == True:
        fig.colorbar(im)
        plt.savefig('Outputs/TanChau/' + 'vector_field_' + str(stt) + '.png')
    plt.show()

def draw_contour(FS):
    X, Y = np.mgrid[2:N+1, 2:M+1]
    fig, ax = plt.subplots(figsize=(10, 10))
    CS = ax.contour(X, Y, FS)
    fig.savefig('fs' + str(i) + '.png')

def saveFS(value, filename, mode='DSAA', val_range=(0, 10)):


    file = open(filename, 'w')

    file.write(mode + '\n')
    file.write(str(N - 1) + ' ' + str(M - 1) + '\n')
    file.write(str(2) + ' ' + str(N) + '\n')
    file.write(str(2) + ' ' + str(M) + '\n')
    file.write(str(val_range[0]) + ' ' + str(val_range[1]) + '\n')
    for i in range(2, N):
        for j in range(2, M):
            # print value[i, j]
            file.write(str(value[i, j] * ros) + ' ')
        file.write('\n')
    file.close()