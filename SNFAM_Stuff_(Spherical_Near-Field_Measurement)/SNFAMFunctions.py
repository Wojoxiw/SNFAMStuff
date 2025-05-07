from scipy.constants import c, epsilon_0, mu_0, k, elementary_charge, m_p
import numpy as np
import scipy
from math import pi, factorial
from scipy.special import spherical_jn, spherical_yn, lpmv, binom
from scipy.integrate import dblquad
from wigners import wigner_3j
import matplotlib.pyplot as plt
from numpy.random._examples.cffi.extending import vals
import os

eta_0 = np.sqrt(mu_0/epsilon_0) ##impedance of free space
eta_0_h = np.sqrt(epsilon_0/mu_0) ## specific admittance, definition from Hansen [3] and EuCAP course slides
eps = 1e-6 ## adjust angles with this to avoid singularities with trig functions.

## make sure angles are in radians here


#####################################################
# Mathematical functions to use
#####################################################
## following Hansen's single index convention [3], but equations from 'Theory and Practice' [1]. B1 has s=1, B2 has s=2
##

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

def JfromN(N):
    return 2*N*(N+2)

## going between transmission and receiving, see (3.22), or (2.107) in Hansen [3], also (A2.16, 3.21)... Daniel's notes since probe used as AUT is rotated pi around y-axis to point to probe
def RfromT(T): ## assuming here that the antenna is reciprocal:
    R = np.zeros(len(T), dtype=complex)
    
    for j in range(len(T)): ## how Daniel did it
        s, m, n = smnFromj(j)
        R[j] = (-1)**(n) * T[j]
    
    #===========================================================================
    # for j in range(len(T)): ## (2.107) - will not work/give bad results
    #     s, m, n = smnFromj(j)
    #     R[j] = (-1)**(m) * T[jFromsmn(s, -m, n)]
    #===========================================================================
        
    return R

def ElectricDipole(p, r, k0):
    #"""Compute the field from an electric dipole with dipole moment p at (0,0,0) evaluated at positions (r_r,r_theta,r_phi), with angular wavevnumber k_0. Takes an array of positions, returns [E_r, E_theta, E_phi]"""'
    ## p is made into a vector here for convenience, not sure what was supposed to happen...
    ps = np.zeros(np.shape(r))
    ps[:,0] = p[0]
    ps[:,1] = p[1]
    ps[:,2] = p[2]
    rabs0 = myabs(r)
    rabs = np.array([rabs0, rabs0, rabs0]).transpose() # Needs to repeat the rabs0 in order to make element-wise multiplication work
    ruv = r/rabs
    k0r = k0*rabs
    E = k0**3/(4*np.pi*epsilon_0)*(mycross(mycross(ruv, ps), ruv)/k0r + (1/k0r**3 - 1j/k0r**2)*myprojection(ruv, ps))*np.exp(1j*k0r)
    return(E)


def delta(a,b): ## kronecker delta
    if(a==b):
        return 1
    else:
        return 0

def SMCsHertzianDipole(J, rotatedForMeas = False): ##based on section 2.3.4 in [2], calculates the SMCs with N modes or J coefficients. They are all zero except when s=2,m=0,n=1 (j = 4)
                                    ## since this is for a z-directed dipole, should rotate these coefficients for i.e. the actual probe, which is in the x-y plane

    T = np.zeros((J), dtype = complex) ## the SMCs
    if(rotatedForMeas): ## Hertzian dipole R^p, rotated to take place of the probe. Seems to then have strange polarizations...
        T[jFromsmn(2, 1, 1)] = -1/np.sqrt(2)
        T[jFromsmn(2, -1, 1)] = 1/np.sqrt(2)
    else:
        for j in range(np.size(T)):
            if(j == 4): 
                T[j-1] = 1 ## j-1 to go from j to python index
    return T

def calcTsProbeCorr(data, freq): ## calculates Tsmn as in section (4.3.2) with self.ProbeRs probe receiving coefficients (Rs), takes the data and the frequency to calc for
    print('Calculating Tsmn with probe correction...')
    Tsmn = np.zeros(data.J, dtype=complex)
    kA = data.A*2*pi*data.fs[freq]/c
    Cs = getCs(data.J, kA) ## translation coefficients
    
    Rs = data.ProbeRs
    if(np.size(Rs) < data.J): ## since the probe is smaller than the AUT - set extra coefficients to zero
        Rs = np.zeros(data.J, dtype=complex)
        Rs[0:np.size(data.ProbeRs)] = data.ProbeRs
    
    Ps = np.dot(Cs, Rs)/2 ## probe response constants, (4.39)
    Ss = data.S21_sphere[freq] ## probe pol (theta, then phi), theta, and phi angle
    thetavec = data.thetavec_sphere*pi/180
    phivec = data.phivec_sphere*pi/180
    dPhi = data.phiSpacing*pi/180
    dTheta = data.thetaSpacing*pi/180
    
    ### remove any near-zero values to avoid dividing by zero
    thetavec[np.abs(thetavec) < eps] = eps
    thetavec[np.abs(thetavec - pi) < eps] = pi - eps
    
    ## the phi integral
    wm_theta = np.fft.fft(Ss[0], axis=1)*dPhi ## chi = 0, theta-hat part
    wm_phi = np.fft.fft(Ss[1], axis=1)*dPhi ## chi = pi/2, phi-hat part
    ### 4.61 and on, the chi integral (4.65, 4.66)
    wmum_p1 = (wm_theta - 1j*wm_phi)/2 ## w_mum, +1 part
    wmum_n1 = (wm_theta + 1j*wm_phi)/2 ## w_mum, -1 part
    
    ## and solve (4.133, 4.134, or 4.53, 4.54) (spacing of 2 since s=1 and s=2 solved together)
    for j in range(0, data.J, 2):
        s, m, n = smnFromj(j)
        P = np.array([ [Ps[jFromsmn(1, 1, n)], Ps[jFromsmn(2, 1, n)]], [Ps[jFromsmn(1, -1, n)], Ps[jFromsmn(2, -1, n)]] ]) ## each value of P
        ### theta integral, (4.55)
        wnmum_p1 = (2*n + 1)/2*np.sum(wmum_p1[:,m]*dnmum(n, 1, m, thetavec)*np.sin(thetavec)*dTheta)
        wnmum_n1 = (2*n + 1)/2*np.sum(wmum_n1[:,m]*dnmum(n, -1, m, thetavec)*np.sin(thetavec)*dTheta)
        
        W = np.array([wnmum_p1, wnmum_n1])
        T = np.linalg.solve(P, W) ## this gets T_1mn and T_2mn
        Tsmn[jFromsmn(1, m, n)] = T[0]
        Tsmn[jFromsmn(2, m, n)] = T[1]
        
    return Tsmn

def calcTsNoProbeCorr(data, freq): ## calculates Tsmn as in (4.30) assuming no probe correction (electric dipole probe), takes the data and the frequency
                                ## data as in SNFAM Stuff
    J = data.J
    A = data.A
    f = data.fs[freq]
    k = 2*pi*f / c

    ## method, using NFTs and np summing to be much faster:
    Q = np.zeros(J, dtype=complex)
    Es = data.S21_sphere[freq]
    thetavec = data.thetavec_sphere*pi/180
    phivec = data.phivec_sphere*pi/180
    Nphi = len(phivec)
    dphi = data.phiSpacing*pi/180
    dtheta = data.thetaSpacing*pi/180
           
    EDFT = np.fft.fft(Es, axis = 2)*dphi ## compute phi integral as a DFT
    #EDFT = Nphi*np.fft.ifft(Es, axis = 2)*dphi ## compute phi integral as a DFT
        
    ### remove any near-zero values to avoid dividing by zero
    if np.isscalar(thetavec):
        if np.abs(thetavec) < eps:
            thetavec = eps
        elif np.abs(thetavec - np.pi) < eps:
            thetavec = np.pi - eps
    else:
        thetavec[np.abs(thetavec) < eps] = eps
        thetavec[np.abs(thetavec - pi) < eps] = pi - eps
        
    for j in range(J):
        if (j%int(J/12.67))== 0:
            print(f'Computing SMCs, j = {j+1} / {J}')
        s, m, n = smnFromj(j)
        F_r, F_theta, F_phi = Fcsmn(3, s, -m, n, A, k, thetavec, 0)
        prefactor = 2/(np.sqrt(6*pi))* (-1)**m / (Rc_sn(k*A, 3, s, n)**2)
        Q[j] = prefactor * np.sum( F_theta*EDFT[0][:,m]*np.sin(thetavec)*dtheta + F_phi*EDFT[1][:,m]*np.sin(thetavec)*dtheta ) ### the theta integration
    return Q

def spherical_h1(n, z, derivative=False):
    return(spherical_jn(n, z, derivative) + 1j*spherical_yn(n, z, derivative))

def spherical_h2(n, z, derivative=False):
    return(spherical_jn(n, z, derivative) - 1j*spherical_yn(n, z, derivative))

# Functions of z = kr
def zc_n(kr, c, n, derivative=False): ## from (2.10)
    if c == 1:
        z_func = spherical_jn(n, kr, derivative)
    elif c == 2:
        z_func = spherical_yn(n, kr, derivative)
    elif c == 3:
        z_func = spherical_h1(n, kr, derivative)
    else:
        z_func = spherical_h2(n, kr, derivative)
    return(z_func)

def Rtildecgamma_sn(kr, c, gamma, s, n): ## from (4.9) (for 4.17)
    if(s==1):
        return Rc_sn(kr, c, s, n)*Rc_sn(kr, gamma, s, n)
    else:
        return Rc_sn(kr, c, s, n)*Rc_sn(kr, gamma, s, n) + n*(n+1)* (zc_n(kr, c, n)/kr) * (zc_n(kr, c, gamma)/kr)

def Rc_sn(kr, c, s, n): ## from (A1.6)
    if(s==1):
        return zc_n(kr, c, n)
    else:
        return oneoverkrdkrzdkr(kr, c, n)

def normlpmv(m, n, x): ## normalized assocaited legendre function, as in (A1.25)
    return lpmv(m,n,x) * np.sqrt( (2*n+1)/2 * factorial(n-m) / factorial(n+m) ) *(-1)**m ## just includes the normalization factor, and ## lpmv is the associated legendre function, defined in Hansen without the -1**m factor, which is included in the scipy implementation
    
def oneoverkrdkrzdkr(kr, c, n):
    return zc_n(kr, c, n, derivative = True) + zc_n(kr, c, n)/kr ## using derivatives instead
    #return (n+1) * zc_n(kr, n, n)/(kr) - zc_n(kr, c, n+1) ## from (A1.9)
    #return zc_n(kr, c, n-1) + n*zc_n(kr, c, n)/kr ## from (A1.8)
    
def Pbarm_n(m, n, costheta): ## normalized associated legendre function as in (A1.25)
    return normlpmv(m, n, costheta) ## lpmv is the associated legendre function, defined in Hansen without the -1**m factor, which is included in the scipy implementation

def dPbar(m, n, theta): ## dPbar(costheta)/dtheta, as in (A1.34b). ## including factor to go from P to Pbar
    costheta = np.cos(theta)
    normfactor = np.sqrt( (2*n+1)/2 * factorial(n-m) / factorial(n+m) ) ## to transport P to Pbar
    if(m==0):
        return 1*lpmv(1, n, costheta) * normfactor
    else:
        sintheta = np.sin(theta)
        return  -( (n-m+1)*(n+m)*lpmv(m-1,n,costheta) + m*costheta/sintheta*lpmv(m,n,costheta) ) * normfactor *(-1)**m ##P(m+1) term changed out using recurrance relation A1.32, since m+1 can be greater than n
    
def Fcsmn(c, s, m, n, A, k, theta, phi): ## spherical wave function, from (A1.45)
    if(m==0):
        mpart = 1
    else:
        mpart = (-1*m/np.abs(m))**m
    prefactor = 1/( np.sqrt(2*pi*n*(n+1)) ) * mpart * np.exp(1j*m*phi)
    costheta = np.cos(theta)
    if(s==1):
        thetaPart = prefactor* zc_n(k*A, c, n) * 1j*m*Pbarm_n(np.abs(m), n, costheta) / (np.sin(theta))  ## theta-hat part
        phiPart = prefactor* -1* zc_n(k*A, c, n) * dPbar(np.abs(m), n, theta) ##phi-hat part
        return [thetaPart*0, thetaPart, phiPart]
    else:
        rPart = prefactor* n*(n+1)/(k*A) * zc_n(k*A, c, n) * Pbarm_n(np.abs(m), n, costheta) ## r-hat part
        thetaPart = prefactor* oneoverkrdkrzdkr(k*A, c, n) * dPbar(np.abs(m), n, theta) ## theta-hat part
        phiPart = prefactor* oneoverkrdkrzdkr(k*A, c, n) * 1j*m*Pbarm_n(np.abs(m), n, costheta) / (np.sin(theta)) ##phi-hat part
        return [rPart, thetaPart, phiPart]


def dnmum(n, mu, m, theta): ## rotation coefficients, from Hansen (A2.5) ##for the jacobi, possibly the top two arguments have to be greater than -1, so use if statements to convert them if not
    ## check arguments for Jacobi polynomial - if bad, switch using symmetries (A2.8, A2.9)
    if(mu - m > -1 and mu + m > -1): ## args are good
        d = find_dnmum(n, mu, m, theta)
    elif(mu - m <= -1 and mu + m > -1): ## swap m and mu
        d = find_dnmum(n, m, mu, theta)*(-1)**(mu + m)
    elif(mu - m <= -1 and mu + m <= -1): ## make m and mu negative
        d = find_dnmum(n, -mu, -m, theta)*(-1)**(mu + m)
    elif(mu - m > -1 and mu + m <= -1): ## swap m and mu + make them negative
        d = find_dnmum(n, -m, -mu, -theta)*(-1)**(mu + m) ## Daniel has this factor, but it seems like it should be cancelled out - maybe has no effect?
    
    return d
def find_dnmum(n, mu, m, theta): ## to allow changing mu, m, etc
    prefactor = np.sqrt( (factorial(n+mu)*factorial(n-mu)) / (factorial(n+m)*factorial(n-m)) ) * np.cos(theta/2)**(mu+m)*np.sin(theta/2)**(mu-m)
    return prefactor * scipy.special.eval_jacobi(n-mu, mu-m, mu+m, np.cos(theta))

def Csn3sigmunu(s,n,sigma,nu,mu,kA): ## from Hansen (A3.3), assuming positive kA
    prefactor = np.sqrt( (2*n+1)*(2*nu+1) / (n*(n+1)*nu*(nu+1)) ) * np.sqrt(factorial(nu+mu)*factorial(n-mu) / (factorial(nu-mu)*factorial(n+mu)) ) * (-1)**(mu)*1/2*(1j)**(n-nu)
    sum = 0
    for p in range(np.abs(n-nu), n+nu+1):
        sum += (1j)**(-p) * (delta(s,sigma)*( n*(n+1) + nu*(nu+1) - p*(p+1) ) + delta(3-s,sigma)*2j*mu*kA ) * a(mu,n,nu,p)*zc_n(kA, 3, p)
    return prefactor*sum

def a(mu,n,nu,p): ## 'linearization coefficients', from (A3.6)
    return (2*p+1)*np.sqrt( factorial(n+mu)*factorial(nu-mu) / (factorial(n-mu)*factorial(nu+mu)))*wigner_3j(n, nu, p, 0, 0, 0)*wigner_3j(n, nu, p, mu, -mu, 0)

def getCs(J, kA, calcNew = False): ## computes the translation coefficients above, or loads them if already calculated for a given J, kA
    fF = 'C:/Users/al8032pa/Work Folders/Documents/antenna measurements/SNFAM/savedCs/'
    file = fF+f'J{J}kA{kA:.5f}.npz'
    if(os.path.isfile(file) and not calcNew):
        print('Importing previous translation coefficents...')
        C = np.load(file)['Cs']
    else:
        C = np.zeros((J, J), dtype=complex)
        for j in range(J):
            if np.mod(j, int(J/9.2)) == 0:
                print(f'Computing Csn3sigmunu, j = {j} / {J}')
            s, m, n = smnFromj(j)
            for j2 in range(J):
                sigma, mu, nu = smnFromj(j2)
                if mu == m:
                    C[j,j2] = Csn3sigmunu(s,n,sigma,nu,mu,kA)
        np.savez(file,Cs = C)
    return C

def Ksmn(s,m,n,theta,phi): ## Ksmn, from Hansen (A1.59,A1.60)
    if(len(theta)==len(phi)): ## to get data for cut plotting
        K = np.zeros((2, len(theta)), dtype=complex) ##pols, theta/phi vals
    else:
        K = np.zeros((2, len(theta), len(phi)), dtype=complex) ##pols, theta vals, phi vals
    if(m==0):
        mPart = 1
    else:
        mPart = (-1*np.abs(m)/m)**m
    if(s==1):
        prefactor = np.sqrt(2/(n*(n+1))) * mPart * (-1j)**(n+1) * np.exp(-1j*m*phi) ## either +j or -j, causes a rotation
        Kthet = 1j*m/np.sin(theta) * Pbarm_n(np.abs(m), n, np.cos(theta)) ## theta part K
        Kphi = -1 * dPbar(np.abs(m), n, theta) ## phi part K
    else:
        prefactor = np.sqrt(2/(n*(n+1))) * mPart * (-1j)**(n) * np.exp(-1j*m*phi) ## either +j or -j, causes a rotation
        Kthet = dPbar(np.abs(m), n, theta) ## theta part K
        Kphi = 1j*m/np.sin(theta) * Pbarm_n(np.abs(m), n, np.cos(theta)) ## phi part K
        
    if(len(theta)==len(phi)): ## to get data for cut plotting
        K[0] = Kthet*prefactor
        K[1] = Kphi*prefactor
    else:
        K[0] = (np.outer(Kthet, prefactor))
        K[1] = (np.outer(Kphi, prefactor))
    return K

def findFarField(Ts, thetas, phis):
    '''
    Finds far-field at some theta, phi using transmission coefficients, as in (2.182, 2.180), Ks as in (A1.59, A1.60)
    Returns array of theta- and phi- pol Es
    If theta and phi vectors are same length, calculate 1-D answer rather than use outer products (returns array of [2, len(angles)]). Otherwise, [2, len(thetas), len(phis)]    
    :param Ts: The coefficients
    :param thetas: Vector of theta angles
    :param phis: Vector of phi angles
    '''
    J = len(Ts)
    ### remove any near-zero values to avoid dividing by zero
    if np.isscalar(thetas):
        if np.abs(thetas) < eps:
            thetas = eps
        elif np.abs(thetas - np.pi) < eps:
            thetas = np.pi - eps
    else:
        thetas[np.abs(thetas) < eps] = eps
        thetas[np.abs(thetas - pi) < eps] = pi - eps
       
    if(len(thetas)==len(phis)): ## to get data for cut plotting
        bigK = np.zeros((2, len(thetas)), dtype=complex) ## pol (theta, then phi), theta, and phi angle
        for j in range(J):
            if np.mod(j, int(J/3.2)) == 0:
                print(f'Computing farfield cut, j = {j} / {J}')
            s, m, n = smnFromj(j)
            bigK += Ts[j]*Ksmn(s, m, n, thetas, phis)
        return bigK/np.sqrt(4*pi*eta_0_h)
    else: ## to get data for sphere plotting
        bigK = np.zeros((2, len(thetas), len(phis)), dtype=complex) ## pol (theta, then phi), theta, and phi angle
        for j in range(J):
            if np.mod(j, int(J/4.2)) == 0:
                print(f'Computing farfield top sphere, j = {j} / {J}')
            s, m, n = smnFromj(j)
               
            bigK += Ts[j]*Ksmn(s, m, n, thetas, phis)
        return bigK/np.sqrt(4*pi*eta_0_h)

def gravityDroopCompensation(data, alphas, droopDist, measurementDist, doPlots = 1):
    '''
    Returns compensated values at the same positions, determined as in 'Compensation of Gravity Bending of MetOp-SG on-ground Calibration Measurements', from DTU
    Works as follows:
    The measurement coordinate system (MCS) has axes x (pointing toward the door), y (pointing upward), and z (pointing to probe), originating where the roll and azimuthal axes meet (center of the antenna)
    The antenna coordinate system (ACS) has corresponding axes u, v, and w which are parallel to x, y, z when the antenna points toward the probe. It originates in the flange where bending occurs (-droopDist in z). This gives the 'actual' pointing of the antenna.
    The 'real' measurement angle is calculated from the facing of the flange: first a phi rotation around the z-axis, then an alpha-rotation around the bending axis (x-axis translated to point through the ACS origin), and finally a theta-rotation around the y-axis.
    
    Compensated data is then not on the regular sampling points, so 2d interpolation needs to be used later to generate values at the original angles.
    
    :param data: The array of data points
    :param alphas: Array of the droop angles, in degrees, for each theta angle (np.sort, np.unique, thetas)
    :param droopDist: Distance between antenna and drooping connection
    :param measurementDist: Distance between antenna and probe
    :param doPlots: If True, plots some vectors to visualize the situation
    :param dist: Distance between the intersection of rotation axes and the connection to the rotator (where the droop is)
    '''
    rotZ = lambda phi: np.array([[np.cos(phi), -1*np.sin(phi), 0],
                                 [np.sin(phi), np.cos(phi), 0],
                                 [0, 0, 1]]) ## rotation matrix around z-axis (phi rotation)
    rotY = lambda theta: np.array([[np.cos(theta), 0, np.sin(theta)],
                                   [0, 1, 0],
                                   [-1*np.sin(theta), 0, np.cos(theta)]]) ## rotation matrix around y-axis (theta rotation)
    rotX = lambda alpha: np.array([[1, 0, 0],
                                   [0, np.cos(alpha), -1*np.sin(alpha)],
                                   [0, np.sin(alpha), np.cos(alpha)]]) ## rotation matrix around x-axis (gravity droop rotation)
    rotArb = lambda ang, ax: np.array([[0, -ax[2], ax[1]],
                                       [ax[2], 0, -ax[0]],
                                       [-ax[1], ax[0], 0]])*np.sin(ang) + np.identity(3)*np.cos(ang) + (1-np.cos(ang))*np.outer(ax, ax) ## rotation matrix around some given axis, according to Wikipedia
    
    def rotatePlusTranslate(vectors, rMat, transVec): ## rotate around a list of axes that may be distant, but applying a translation matrix before and after. then normalize
        ## using the 4-D matrix method (accepts 3D vec/mats)
        transMat = lambda vec: np.array([[1, 0, 0, -vec[0]],
                                         [0, 1, 0, -vec[1]],
                                         [0, 0, 1, -vec[2]],
                                         [0, 0, 0, 1]])
        rotMat = np.array([[rMat[0, 0], rMat[0, 1], rMat[0, 2], 0],
                           [rMat[1, 0], rMat[1, 1], rMat[1, 2], 0],
                           [rMat[2, 0], rMat[2, 1], rMat[2, 2], 0],
                           [0, 0, 0, 1]])
        total = np.matmul(transMat(-1*transVec), np.matmul(rotMat, transMat(transVec))) ## translate, rotate, then translate back
        for v in range(len(vectors)):
            vector = vectors[v]
            vector = np.array([vector[0], vector[1], vector[2], 1])
            vector = np.matmul(total, vector)
            vector = np.array([vector[0], vector[1], vector[2]]) ## convert back to 3-vector
            vectors[v] = vector/np.linalg.norm(vector)
            if(len(vectors) == 1): ## just the one
                vectors = vector
        return vectors ## normalize
    origin = np.array([0, 0, 0]) ## antenna/MCS origin
    probePos = np.array([0, 0, measurementDist]) ## translation from MCS to ACS
    ACSorigin = np.array([0, 0, -droopDist]) ## origin of ACS (flange position)
    thets = np.round(np.real(data[:,3]), 3)
    thetas = np.sort(np.unique(thets))
    phs = np.round(np.real(data[:,4]), 3)
    phis = np.sort(np.unique(phs))
    
    xax, yax, zax = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    
    def adjustPoint(theta, phi): ## takes the theta and phi angle of a measurement point, and returns the adjuted theta' and phi'
        alphaidx = np.argmin(np.abs(phis*pi/180-phi))## index of the correct alpha for this theta angle
        alpha = alphas[alphaidx]
        thphp = ACSorigin # the origin of the ACS, as it rotates? must apply rotations on itself, since this is defined from the MCS
        antx, anty, antz = np.matmul(rotZ(phi), xax), np.matmul(rotZ(phi), yax), np.matmul(rotZ(phi), zax) ## apply phi rotation around z-axis of MCS
        antx, anty, antz = rotatePlusTranslate([antx, anty, antz], rotX(alpha), transVec=origin) ## apply angular droop (alpha rotation around bending axis (MCS x-axis at flange position)). should be at the origin of the ACS
        thphp = rotatePlusTranslate([thphp], rotArb(theta, yax), transVec=ACSorigin)
        antx, anty, antz = rotatePlusTranslate([antx, anty, antz], rotArb(theta, yax), transVec=-origin) # theta rotation around the y-axis, which is in front of the ACS origin now
        p = probePos - thphp ## vector from ACS to probe
        p = p/np.linalg.norm(p) # normalize
        thetap = np.arccos(np.clip(np.dot(p, antz), -1, 1))
        phip = np.arctan2(np.dot(p, antx), np.dot(p, anty)) + pi/2
        
        if(theta < 0): ## move quadrants if theta < 0
            thetap = -thetap
            phip = phip + pi
            if(phip > 2*pi): ## this can go over 360 degrees
                phip -= 2*pi
        return thetap, phip
    
    
    if(doPlots > 0): ## to visualize the situation
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        #ax.quiver(0, 0, 0, 1, 0, 0, linewidth = 1.5, color = 'gray')#, label = 'x')
        #ax.quiver(0, 0, 0, 0, 1, 0, linewidth = 1.5, color = 'gray')#, label = 'y')
        ax.scatter(-measurementDist, 0, 0, s = 100, color = 'gray', label = 'Door') ## If the AUT is pointing at (0, 0) - this is the part of the meas. sphere that hits the door. (theta = -90, phi=0)
        #ax.quiver(0, 0, 0, 0, 0, 1, linewidth = 1.5, color = 'gray')#, label = 'z')
        ax.scatter(0, 0, 0, label = 'origin/AUT', s = 33)
        ax.text(0, 0, -measurementDist, "Back of room")
        ax.scatter(0, 0, -droopDist, label = 'flange', s = 33)
        ax.scatter(0, 0, measurementDist, label = 'probe', s = 100)
        ax.set_xlim([-measurementDist, measurementDist])
        ax.set_ylim([-measurementDist, measurementDist])
        ax.set_zlim([-measurementDist, measurementDist])
        ax.set_aspect("equal")
        ax.set_xlabel('X-Axis', fontsize = 12)
        ax.set_ylabel('Y-Axis', fontsize = 12)
        ax.set_zlabel('Z-Axis', fontsize = 12)
        plt.title('Pointing of AUT on Meas. Sphere', fontsize = 13)
        
        thph00 = [0, 0, 1] ## vector for theta=phi=0 degrees pointing direction
        theta = -8*pi/180 ## theta, phi for intended measurement coordinate, to be plotted without bending
        phi = 90*pi/180
        
        thetap = theta ## theta, phi for intended measurement coordinate, to be bent with a droop of alpha
        phip = phi
        alphaidx = np.argmin(np.abs(phis*pi/180-phip))## index of the correct alpha for this theta angle
        alpha = alphas[alphaidx]
        
        ## try using axes of antenna
        antx, anty, antz, thph = np.matmul(rotZ(phi), xax), np.matmul(rotZ(phi), yax), np.matmul(rotZ(phi), zax), np.matmul(rotZ(phi), thph00) ## apply phi rotation around z-axis of MCS
        ## apply angular droop
        thph = np.matmul(rotArb(theta, yax), thph) ## apply theta rotation around y-axis of MCS
        thphp = ACSorigin # the origin of the ACS, as it rotates? must apply rotations on itself, since this is defined from the MCS
        antx, anty, antz = np.matmul(rotZ(phip), xax), np.matmul(rotZ(phip), yax), np.matmul(rotZ(phip), zax) ## apply phi rotation around z-axis of MCS
        antx, anty, antz = rotatePlusTranslate([antx, anty, antz], rotX(alpha), transVec=origin) ## apply angular droop (alpha rotation around bending axis (MCS x-axis at flange position)). should be at the origin of the ACS
        thphp = rotatePlusTranslate([thphp], rotArb(thetap, yax), transVec=ACSorigin)
        antx, anty, antz = rotatePlusTranslate([antx, anty, antz], rotArb(thetap, yax), transVec=-origin) # theta rotation around the y-axis, which is in front of the ACS origin now
        ax.quiver(thphp[0], thphp[1], thphp[2], antx[0], antx[1], antx[2], linewidth = 1.5, color = 'gray', label = 'u', alpha = 0.66)
        ax.quiver(thphp[0], thphp[1], thphp[2], anty[0], anty[1], anty[2], linewidth = 1.5, color = 'black', label = 'v', alpha = 0.66)
        ax.quiver(thphp[0], thphp[1], thphp[2], antz[0], antz[1], antz[2], linewidth = 1.5, color = 'green', label = 'w', alpha = 0.66)
        
        ## calculate theta and phi:
        p = probePos - thphp ## vector from ACS to probe
        ax.quiver(thphp[0], thphp[1], thphp[2], p[0], p[1], p[2], linewidth = 0.5, color = 'purple', label = r'$\boldsymbol{p}$')
        p = p/np.linalg.norm(p) # normalize
        thetap, phip = adjustPoint(thetap, phip)
        
        # thphp is now the pointing direction of the axis
        antx, anty, antz, thphp= np.matmul(rotZ(phip), xax), np.matmul(rotZ(phip), yax), np.matmul(rotZ(phip), zax), np.matmul(rotZ(phip), thph00)
        thphp = np.matmul(rotArb(thetap, yax), thphp) ## apply theta rotation around y-axis of MCS
        ax.quiver(0, 0, 0, thph[0], thph[1], thph[2], linewidth = 1.5, color = 'red', label = r'$\theta, \phi =$'+f'({theta*180/pi:.1f}$^\circ ,${phi*180/pi:.1f}$^\circ$)', length = measurementDist)
        ax.quiver(0, 0, 0, thphp[0], thphp[1], thphp[2], linewidth = 1.5, color = 'blue', label = r'$\theta^\prime, \phi^\prime$='+f'({thetap*180/pi:.1f}$^\circ ,${phip*180/pi:.1f}'+r'$^\circ ,\alpha=$'+f'{alpha*180/pi:.1f}'+r'$^\circ$)', length = measurementDist)
        ## draw the intended measurement partial-sphere in gray, actual in red
        nth = 16
        nph = 20
        th, ph = np.mgrid[-100*pi/180:100*pi/180:nth*1j, 0:240*pi/180:nph*1j]
        x = np.cos(ph)*np.sin(th)
        y = np.sin(ph)*np.sin(th)
        z = np.cos(th)
        ax.plot_wireframe(x*measurementDist, y*measurementDist, z*measurementDist, color="gray", linewidth = 0.5)
        ## now the rotated one
        nth = 20
        nph = 15
        th, ph = np.mgrid[-100*pi/180:100*pi/180:nth*1j, 0:240*pi/180:nph*1j]
        for j in range(nph): ## for each point on the meshgrid, do the calculation for angles
            for i in range(nth):
                thetap, phip = adjustPoint(th[i, j], ph[i, j])
                th[i, j] = thetap
                ph[i, j] = phip
            x = np.cos(ph)*np.sin(th)
            y = np.sin(ph)*np.sin(th)
            z = np.cos(th)
            ax.plot(x[:, j]*measurementDist, y[:, j]*measurementDist, z[:, j]*measurementDist, color="blue", linewidth = 0.66)
        plt.legend()
        fig.tight_layout()
        plt.show()
    
    print('Applying gravity droop compensation...')
    
    ## First convert to cartesian coordinates, then apply roll rotation (around z), gravitational droop (around x but with an origin at the bending place), then azimuthual rotation (around y).
    # Take the y-axis to be vertical, pointed upward (theta-angle is then negative rotation around this axis)
    # Take the z-axis to point toward the probe antenna
    # Take the x-axis to point away from the door (to the left from the x-axis)
    
    for i in range(np.shape(data)[0]): # just iterate over each data point to rotate it (slow, but trying to vectorize is annoying)
        theta = np.real(data[i, 3]) * pi/180 ## convert to radians
        phi = np.real(data[i, 4]) * pi/180 ## convert to radians
        if(theta == 0):
            theta += 1e-12 ## to avoid singularities/etc
        
        rotatedTheta, rotatedPhi = adjustPoint(theta, phi)
        #if(np.abs(theta) < 10*pi/180):
        #    rotatedPhi = phi
        
        
        #print(rotatedTheta, rotatedPhi, ' :: ', theta, phi)
        data[i, 3] = rotatedTheta * 180/pi ## convert to degrees
        data[i, 4] = rotatedPhi * 180/pi ## convert to degrees
    return data

def interpToGrid(data, oThetas, oPhis): 
    '''
    Interpolates the data back onto the measurement grid of the original points. 
    :param data: The data that has been moved off the grid
    :param oThetas: Original thetas
    :param oPhis: Original phis
    '''
    gridPoints = np.transpose(np.vstack((oThetas, oPhis)))
    thetas = np.round(np.real(data[:,3]), 3)
    phis = np.round(np.real(data[:,4]), 3)
    points = np.transpose(np.vstack((thetas, phis)))
    NFs = int((np.shape(data)[1]-5)/2)
    print('Using RBF interpolator to calculate values at grid points...')
    for pp in [0, 90]: ## for each probe pol
        ppindices = np.argwhere(np.abs(data[:, 2]-pp) < 1e-1)[:, 0] ## indices of data points at this probe pol
        if(len(ppindices != 0)): ## don't do this if nothing to interpolate
            for h in range(NFs): ## each frequency point
                i = 6+2*h ## the index
                Ssreal = np.real(data[ppindices, i])
                Ssimag = np.imag(data[ppindices, i])
                
                ## RBF interpolator is slower, but with Nneighbours > 25 gives reasonable results even near edges
                newSsreal = scipy.interpolate.RBFInterpolator(points[ppindices, :], Ssreal, neighbors = 50)(gridPoints[ppindices, :])
                newSsimag = scipy.interpolate.RBFInterpolator(points[ppindices, :], Ssimag, neighbors = 50)(gridPoints[ppindices, :])
                
                #===============================================================
                # ## grid interpolator is fast, but gives strange behaviour near edges. I will cut these edges off anyway, though...
                # newSsreal = scipy.interpolate.griddata(points[ppindices, :], Ssreal, gridPoints[ppindices, :], method = 'linear', fill_value = 0)
                # newSsimag = scipy.interpolate.griddata(points[ppindices, :], Ssimag, gridPoints[ppindices, :], method = 'linear', fill_value = 0)
                #===============================================================
                
                data[ppindices, i] = newSsreal + 1j*newSsimag
    data[:, 3] = oThetas
    data[:, 4] = oPhis
    return data

### REFERENCES

#[1]: Theory and Practice of Modern Antenna Range Measurements 2nd Expanded Edition, Volume 2 (2020)

#[2] https://mathworld.wolfram.com/SphericalHankelFunctionoftheSecondKind.html

#[3]: Spherical near-field antenna measurements, Hansen (1988)

#[4]: Fully Probe-Corrected Near-Field Far-Field Transformations With Unknown Probe Antennas, Paulus (2023)