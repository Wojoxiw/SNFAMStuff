import matplotlib.pyplot as plt
import numpy as np

def jFromsmn(s,m,n): # from [3] appendix A1
    return (s+2*( n*(n+1) + m-1 )) - 1 ## -1 to go from j to python index

def smnFromj(j): # from [3] appendix A1
    j = j+1 ##to translate from Python starting at index 0
    if(j%2 == 0):
        s = 2
    else:
        s = 1
    n = int(np.floor(np.sqrt((j - s)/2 + 1)))
    m = int((j-s)/2 + 1 -n*(n+1))
    return s, m, n

def plotQs(Qs, justTriang = False):
    s, m, nmax = smnFromj(len(Qs)-1)
    mmax = nmax*2+1
    Q_mn = np.zeros((mmax,nmax))
    ns = np.arange(nmax)+1
    ms = np.arange(-nmax,nmax+1)
    Q_n = np.zeros(nmax)
    for n in ns:
        for s in [1,2]:
            for m in np.arange(-n,n+1):
                Q_n[n-1] += np.abs(Qs[jFromsmn(s, m, n)])
                Q_mn[int((mmax-1)/2+m)][n-1] += np.abs(Qs[jFromsmn(s, m, n)])
                
    ##normalize + log-scale:
    Q_n = 10*np.log10(Q_n/np.max(Q_n))
    Q_mn = 10*np.log10(Q_mn/np.max(Q_mn))
    
    if(not justTriang):
        fig = plt.figure(figsize = (9,6))
        plt.xlabel('Mode number, n')
        plt.ylabel('Normalized Amplitude [dB]')
        plt.grid(True)
        plt.title(r'Spherical mode $Q_n$ amplitude, summed over m')
        plt.ylim(-20,0)
        
        plt.plot(ns, Q_n)
        plt.tight_layout()
    
    
    fig = plt.figure(figsize = (6,9))
    plt.imshow(Q_mn, cmap=plt.cm.jet, extent=[1,nmax,-nmax,nmax])
    #X, Y = np.meshgrid(ns,ms)
    #plt.contourf(ns, ms, Q_mn, 100, cmap=plt.cm.nipy_spectral)
    plt.colorbar()
    plt.clim(-20,0)
    plt.xlabel('N')
    plt.ylabel('M')
    plt.title(r'$|Q_{nm}|$ [dB] ')
    plt.tight_layout()
    plt.show()