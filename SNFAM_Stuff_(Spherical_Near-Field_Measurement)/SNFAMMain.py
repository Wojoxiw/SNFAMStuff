from scipy.constants import c, epsilon_0, mu_0, k, elementary_charge, m_p
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import os
from scipy.special import jn_zeros
from scipy.special import jnp_zeros
import scipy.signal
import miepython
from timeit import default_timer as timer
from SNFAMFunctions import ElectricDipole
import SNFAMFunctions
from QPlotter import plotQs
import plotCSTData
import statsmodels.api as sm
from ParaviewVisualization import ExportDataForParaview

lowess = sm.nonparametric.lowess
eps = 1e-20 ## to avoid singularities
eta_0 = np.sqrt(mu_0/epsilon_0)
linestyles = ['-',':','-.','--']
colors = ('tab:blue','tab:orange','tab:red','tab:purple','tab:green','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan')

#####################################################
# This should eventually take in spherical near-field antenna measurements, and output far-field results. Hopefully it does that now.
#####################################################

class SNFData():
    """Class to hold the data."""
    def __init__(self, Nmodes, measurement_radius, folder, files, name, fake = False, duplicates = 'overwrite', phiCut = (6.2,238.5), phiAdjust = []): ## use fake=True if generating data
                                ### duplicates can be 'average', 'overwrite', or 'ignore', phiCut removes phi data outside that range
                                ### phiAdjust is list of two arrays of (theta angle, phi angle adjustment) for use aligning phi angles, one list for each probe pol. [[pp=0], [pp=90]]
        self.name = name
        self.A = measurement_radius
        self.J = Nmodes
        
        if(fake): ## if fake, pass in the fake data as 'files'. Currently the code has no fake data to try, so this is unnecessary
            dats = files
        else: ##                                                         [0]           [1]            [2]           [3]                 [4]      [5]      [6]    [7]   [8]     [9]   [10]
            for i in range(len(files)): ## data is in the Format: # of triggers, time since start, probe pol, theta angle [degrees], phi angle, f1 [Hz], S21(f1), f2, S21(f2), f3, S21(f3)
                                        ## probe pol = 0 should be horizontal (theta) pol, pp = 90 should be phi (vertical) pol
                file = files[i]
                load = np.loadtxt(folder+file, dtype = complex, delimiter = ',', skiprows = 3)
                
                #===============================================================
                # ## cut out the first bunch and final few phi values here (worse data there, especially at start since positioner has just switched rotation direction):
                # load = load[(load[:,4]>phiCut[0])]
                # load = load[(load[:,4]<phiCut[1])]
                # #
                #===============================================================
                 
                if(i==0): ## first file, setup the array
                    dats = load
                else:
                    dats = np.vstack((dats,load))
                    
            dats[:,2:5] = np.round(np.real(dats[:,2:5]), 3) ## round and real all angles before continuing
                    
            self.plotContoursPreProcessing(dats, pol=1) ### this line makes a contour plot of the raw data before any processing
            self.plotContoursPreProcessing(dats, pol=0) ### this line makes a contour plot of the raw data before any processing
                 
            if(len(phiAdjust) > 0): ## try this adjustment
                #self.plotContoursPreProcessing(dats, pol=0) ### this line makes a contour plot of the raw data before any processing
                #self.plotContoursPreProcessing(dats, pol=1) ### this line makes a contour plot of the raw data before any processing
                self.plotContoursPreProcessing(dats, pol=1) ### this line makes a contour plot of the raw data before any processing
                 
                ########### aligning phi angles - done at each theta angle
                # first manually find phi adjustments at different theta angles where side lobes/zero angles allow easy alignment (phiAdjust list)
                # then interpolate, smooth, and apply to each theta angle (below), for each probe pol
                plt.title('Phi Adjustments by Theta Angle')
                plt.xlabel('Theta [degrees]')
                plt.ylabel('Phi [degrees]')
                uniquethetas = np.sort(np.real(np.unique(dats[:,3])))
                smoothingFrac = 1/8 ## more for more smoothness
                ## pp0
                plt.plot(phiAdjust[0][0], phiAdjust[0][1], label='pp=0', linestyle = '--')
                y_makimapp0 = scipy.interpolate.Akima1DInterpolator(phiAdjust[0][0], phiAdjust[0][1], method = 'makima') ## ultimate values to adjust
                adjusterpp0 = lowess(y_makimapp0(uniquethetas), uniquethetas, frac = smoothingFrac, return_sorted = False) ## smoothing
                plt.plot(uniquethetas, adjusterpp0, label='pp=0 Interpolation', linewidth = 1.9)
                roundadjustpp0 = np.round(adjusterpp0*1/0.8)*0.8/1
                plt.plot(uniquethetas, roundadjustpp0, label='pp=0 Interpolation, rounded')
                ## pp90
                plt.plot(phiAdjust[1][0], phiAdjust[1][1], label='pp=90', linestyle = '--')
                y_makimapp90 = scipy.interpolate.Akima1DInterpolator(phiAdjust[1][0], phiAdjust[1][1], method = 'makima') ## ultimate values to adjust
                adjusterpp90 = lowess(y_makimapp90(uniquethetas), uniquethetas, frac = smoothingFrac, return_sorted = False)
                plt.plot(uniquethetas, adjusterpp90, label='pp=90 Interpolation', linewidth = 1.9)
                roundadjustpp90 = np.round(adjusterpp90*1/0.8)*0.8/1
                plt.plot(uniquethetas, roundadjustpp90, label='pp=90 Interpolation, rounded')
                
                ## then plot it
                plt.legend()
                plt.tight_layout()
                plt.show()
                ### now adjust phi angles
                thetas = np.real(dats[:,3])
                pps = np.real(dats[:,2])
                Nfs = int((np.shape(dats)[1]-5)/2) ## number of freq pts
                
                for i in range(len(uniquethetas)): ## for each theta angle, adjust the phi values
                    angle = uniquethetas[i]
                    atangle = np.isclose(thetas, angle)
                    idxpp0 = np.isclose(pps, 0) & atangle
                    idxpp90 = np.isclose(pps, 90) & atangle
                    
                    #===============================================================
                    # ## if I just increase to the nearest rounded angle (no interpolation): (this is bugged currently)
                    # dats[:,4][idxpp0] = dats[:,4][idxpp0] + roundadjustpp0[i] ## pp0
                    # dats[:,4][idxpp90] = dats[:,4][idxpp90] + roundadjustpp90[i] ## pp90
                    #===============================================================
                    
                    ## if I instead set the S21-data to its interpolated equivalent at the phi-adjusted angles: (need to cut out values where no interpolation is possible)
                    for j in range(NFs): ## adjust S21 for each frequency measured
                        Sspp0 = dats[idxpp0, 6+2*j] ## the S21 at frequency j, for this theta angle and probe polarization
                        phispp0 = dats[idxpp0,4] ## the phi values
                        phiAdj = adjusterpp0[i] ## how much to adjust phi values by
                        
                        ### set new phis to be the 
                        
                        interpValues = [] ## values I want to find the S21 at
                        
                        realInterppp0 = lowess(scipy.interpolate.Akima1DInterpolator(phispp0, np.real(S21spp0), method = 'makima')(interpValues), uniquethetas, frac = smoothingFrac, return_sorted = False) ## interpolating and smoothing, real part
                    
                
                
                self.plotContoursPreProcessing(dats, pol=1) ###contour plot of the data again after adjusting phi angles
                exit()
            
             
            #=======================================================================
            # # Rotate each data point to compensate for gravitational droop
            # #dats = SNFAMFunctions.gravityDroopCompensation(dats, .45) ## second argument is gravity droop angle in degrees. This has not been implemented.
            #=======================================================================
            
                    
            ## Fill in unmeasured values at theta = 0, phi from 240 to 360 (also any cut-out values)
            newDats = []
            unMeasuredRange = [phiCut[1],360] ## values in this range were not measured (or are cut away)... uninclusive of 240 and 0 degrees
            unMeasuredRange2 = [0,phiCut[0]] ## values in this range were cut away (inclusive of first boundary)
            for dat in dats: ## each data point, add mirrored values at theta=0 to non-measured phi angles
                theta = dat[3]
                phi = dat[4]
                if(theta == 0): ## we are only doing this for theta = 0
                    if((phi > (unMeasuredRange[0]-180) and phi < (unMeasuredRange[1]-180))): ## phi is within the range
                        dat2 = np.copy(dat) ## new 'fake data point' to fill the gap. Must copy or it will simply use it
                        dat2[4] = phi+180
                        newDats.append(dat2)
                    elif(phi >= (unMeasuredRange2[0]+180) and phi < (unMeasuredRange2[1]+180)): ## phi is within the second range
                        dat2 = np.copy(dat) ## new 'fake data point' to fill the gap. Must copy or it will simply use it
                        dat2[4] = phi-180
                        newDats.append(dat2)
            dats = np.vstack((dats,np.array(newDats)))
            ##
            
            ###### iterate through sorted thetas and phis to determine the ranges and spacings.
            thetasSorted = np.real(dats[(np.abs(dats[:,2])*1e3 + (dats[:,3])*1e-3 + np.abs(dats[:,4]-6.4)*1).argsort(), 3]) ## sorted in pol, phi, then theta
            i = 2 
            fVal = thetasSorted[0] ## first value
            while(True):
                if(thetasSorted[i] == fVal):
                    self.thetaRange = np.unique(np.sort(thetasSorted[0:i]))
                    self.thetaSpacing = self.thetaRange[1]-self.thetaRange[0]
                    break
                i=i+1
            phisSorted = np.real(dats[(np.abs(dats[:,2])*1e3 + dats[:,3]*1 + np.abs(dats[:,4])*1e-3).argsort(), 4])
            i = 2
            fVal = phisSorted[0]
            while(True):
                ## first value
                if(phisSorted[i] == fVal):
                    self.phiRange = np.sort(phisSorted[0:i])
                    self.phiSpacing = self.phiRange[1]-self.phiRange[0]
                    break
                i=i+1
            ######
            
            #===================================================================
            # ## probe pol 0 has a max at 0, min at 90, probe pol 90 vice versa (but look for min at 180). look at theta = 0. Sort using freq index 1
            # for pp in (0, 90): ## for each probe pol (handle each separately)
            #      
            #     ## calculate the phi adjustment needed, based on the theta = 0 data
            #     phiIndices = np.intersect1d(np.argwhere(self.phiRange>55), np.argwhere(self.phiRange<215)) ## just look between 55 and 215 degrees... presumably this is well within the range of 'good values'
            #     thetaSort = (np.abs(dats[:,2]-pp)*1e3 + np.abs(dats[:,3]-0)*1 + np.abs(dats[:,4])*1e-3).argsort() ## probe pol = pp, theta = 0, sorted for phi
            #     thetaSortedPhis = dats[thetaSort, 4][phiIndices[0]:phiIndices[-1]]
            #     #print(thetaSortedPhis)
            #     thetaSortedSs = dats[thetaSort, 6][phiIndices[0]:phiIndices[-1]]
            #     phiVal = thetaSortedPhis[np.argmin(np.abs(thetaSortedSs))]
            #     zeroAngle = self.phiRange[np.abs(self.phiRange - (90+pp)).argmin()] ## sadly if we measure in 0.8 degree increments we don't measure 90 degrees, need same angle for both probe pols -> find closest value to phiSpacing
            #          
            #     phiAdjust = zeroAngle-phiVal ## expected min. at phi=90, add this to every phi value. Then add 360 if phi is below 0?
            #     print(f'Adjusting phi values for probe pols. pp = {pp}, 0 -> {phiAdjust}', end =" ")
            #     dats[np.argwhere(dats[:,2] == pp),4] += phiAdjust ## adjust phi values
            #                   
            # dats[:,4][dats[:,4] < 0] += 360 ## to wrap negative values around 360 degrees
            # 
            # ############
            #===================================================================
            
        
        self.len = np.shape(dats)[0] ## number of data points
        Nfs = int((np.shape(dats)[1]-5)/2) ## number of freq pts
        fs = []
        for i in range(Nfs):
            fs.append(np.real(dats[0,5+2*i]))
            if(i==0): ## first point, create the array
                self.S21s = dats[0:self.len, 6] ## the S21s for the first freq
            else: ## stack the other data
                self.S21s = np.vstack((self.S21s, dats[0:self.len, 6+2*i])) ## the S21s for each other freq
        self.S21s = np.conjugate(self.S21s) ## if this is needed, to match the VNA's time convention
        self.fs = np.array(fs)
        self.pos= np.real(np.array([dats[0:self.len, 2], dats[0:self.len, 3], dats[0:self.len, 4]])) ##position/angle data for each freq. [probe pol, theta, phi]
        
        
        print(self.name+' loaded, with '+str(self.len)+' measurements')
        
        ##make a meshgrid of phi, theta, and S points.
        self.thetavec_sphere = np.arange(0, 180, self.thetaSpacing) # Don't include theta = 180, spacing, taken from the meas.
        self.phivec_sphere = np.arange(0, 360, self.phiSpacing) # Do not include phi = 360
        self.theta_sphere, self.phi_sphere = np.meshgrid(self.thetavec_sphere, self.phivec_sphere, indexing='ij')
        
        self.S21_sphere = np.zeros((Nfs, 2, np.size(self.thetavec_sphere), np.size(self.phivec_sphere)), dtype=complex) ## freq, probe pol (theta, then phi), theta, and phi angle
        
        # Go through the data vectors to find which entry in the sphere grid that should be populated.
        print('Attaching S-parameters to meshgrid...')
        for n in range(self.len): ## each data point
            if(n%int(self.len/10) == 1 and n>0):
                print(str(int(n/self.len*100))+'% complete')
                
            if self.pos[1][n] < 0: # Transform negative theta angles to positive. This also gives a 180 degree phase shift - flipping things around - so a negative sign to the data
                theta = np.abs(self.pos[1][n])
                phi = (self.pos[2][n] + 180)%360
                pS = -1 ## pi phase shift
            else:
                theta = self.pos[1][n]
                phi = self.pos[2][n]
                pS = 1 ## 0 phase shift
            ntheta = np.argmin(np.abs(self.thetavec_sphere - theta))
            nphi = np.argmin(np.abs(self.phivec_sphere - phi))
            
            if( self.pos[0][n] == 90 ): ## probePol = 90, or phi
                pp = 1 ## probe pol index
            elif( self.pos[0][n] == 0 ): ## probePol = 0, or theta
                pp = 0
            else:
                print('Probe pol not 0 or 90?')
                exit()
            if(self.S21_sphere[0,pp,ntheta,nphi] == 0 or duplicates == 'overwrite'): ## value not yet assigned, or duplicate data overwriting
                self.S21_sphere[:,pp,ntheta,nphi] = self.S21s[:,n]*pS
            elif(duplicates == 'average'): ## if duplicate value, try taking the average
                self.S21_sphere[:,pp,ntheta,nphi] = (self.S21s[:,n]+self.S21_sphere[:,pp,ntheta,nphi]*pS)/2 ## subtract second measurement because pi phase difference for some reason?
            elif(duplicates == 'ignore'): ## just drop duplicate data
                pass
            else:
                print('You\'ve made a typo in duplicates: '+duplicates)
                exit()
                
        #self.S21_sphere = np.roll(self.S21_sphere, 100, axis=-1)
        
        #self.plotPlanes()
        #self.plot()
        
    def plot(self, freq = 1, pol = 2, phase = 0): ## plots 3-D countour, and sets up spherical data - plots near-field data
        
        self.plotContours(self.S21_sphere[freq], pol = pol, freq = freq, phase = phase)
        self.plotContoursProjected(self.S21_sphere[freq], pol = pol, freq = freq)
        
        #=======================================================================
        # ##imshow part
        # plt.figure()
        # plt.title(f'{self.name}, f={self.fs[freq]*1e-9:.3} GHz NF imshow Plot')
        # plt.imshow(20*np.log10(np.sqrt(np.abs(self.S21_sphere[freq][0])**2+np.abs(self.S21_sphere[freq][1])**2)))
        # plt.colorbar()
        # ##
        #=======================================================================
        
        plotPlanes([self],[freq])
            
    def calcTsmnNoProbeCorrection(self, freq = 1, calcNew = True, fakeVals = None): ## calculates the transmission coefficients, from Hansen's (4.30)
        fF = 'C:/Users/al8032pa/Work Folders/Documents/antenna measurements/SNFAM/savedTs/'
        file = fF+self.name+'TsNPC.npz'
        if(os.path.isfile(file) and not calcNew):
            print('Importing previous SMCs...')
            self.Ts = np.load(file)['TsNPC']
        else:
            self.Ts = SNFAMFunctions.calcTsNoProbeCorr(self, freq)
            np.savez(file,TsNPC = self.Ts)
    
    def calcTsmn(self, calcNew = True, freq = 1, fakeVals = None): ## calculates the transmission coefficients, from Hansen's section (4.3.2) - using probe coefficients. currently assumes probe only has mu +-1 (see (3.10) maybe)
        fF = 'C:/Users/al8032pa/Work Folders/Documents/antenna measurements/SNFAM/savedTs/'
        file = fF+self.name+'TsPC.npz'
        if(os.path.isfile(file) and not calcNew):
            print('Importing previous SMCs...')
            self.Ts = np.load(file)['TsPC']
        else:
            self.ProbeRs = np.load(fF+'ProbeCoefficientsMeasurementReceivingCoefficients'+f'DRH50{self.fs[freq]:4.5e}'+'.npz')['Rs'] #self.ProbeRs = np.load(fF+'ProbeCoefficientsMeasurementReceivingCoefficients'+f'DRH50{self.fs[freq]:4.5e}'+f'J{self.J}.npz')['Rs']
            self.Ts = SNFAMFunctions.calcTsProbeCorr(self, freq)
            np.savez(file,TsPC = self.Ts)
            
    def calcProbeCoefficients(self, freq = 1, calcNew = True, fakeVals = None): ## calculates the probe's receiving coefficients, following Hansen [3] Section 3.2.5 on interative calibration.
    ## I assume here that the measurement used was with the two identical DRH50 horn probe antennas as transmitter/receiver. All other measurements were done with one of these as the probe, so just load the coeffs if not calculating them anew
        fF = 'C:/Users/al8032pa/Work Folders/Documents/antenna measurements/SNFAM/savedTs/'
        file = fF+'ProbeCoefficientsMeasurementReceivingCoefficients'+f'DRH50{self.fs[freq]:4.5e}'+f'.npz'
        if(os.path.isfile(file) and not calcNew): ## file exists and not calculating anew --> just load it
            print('Importing previous probe SMCs...')
            self.ProbeRs = np.load(file)['ProbeRs']
        else: ## calculate the Rs:
            print('Iteratively calculating probe SMCs...')
            self.ProbeRs = SNFAMFunctions.SMCsHertzianDipole(self.J, rotatedForMeas = True) ## initial guess - Hertzian dipole
            n = 0 ## step counter
            while True: ## iterating
                n+=1
                self.Ts = SNFAMFunctions.calcTsProbeCorr(self, freq) ## calculates Ts, using current Rs
                Rsold = self.ProbeRs ## old Rs, for convergence/comparison
                self.ProbeRs = (SNFAMFunctions.RfromT(self.Ts) + self.ProbeRs)/2 ## sets new Rs from current Ts. Possibly take half the old, half the new to help convergence
                convergence = np.linalg.norm((self.ProbeRs-Rsold)) / np.linalg.norm(self.ProbeRs) ##calculate difference between current and previous step - using 2-norm of difference divided by 2-norm
                print('Step '+str(n)+', convergence (relative difference): '+str(convergence)+f', norm(old) = {np.linalg.norm(Rsold)}, norm(new) = {np.linalg.norm(self.ProbeRs)}')
                if(convergence < 1e-5): ### if it has sufficiently converged, finish
                    np.savez(file, Rs = self.ProbeRs)
                    print('Sufficient convergence achieved: probe coefficients calculated and saved.')
                    #plotQs(self.ProbeRs)
                    break
    
    def plotFarField(self, normalize = True, levels = None, Nlevels = 45, pol = 2, spacing = 1, Nplot = 777, freq = 1, plotCuts = False, plotContours = True, colourbar = False): ## uses calculated transmission coefficients to calculate the far-field. Creates data on the sphere as in plot(), then plots planes (TODO)
        
        ##make a meshgrid of phi, theta, and S points for the top half of a sphere. Lower spacing calculates more points, is thus slower
        self.FFthetavec_sphere = np.arange(0, 90+spacing, spacing) # Include theta = 180
        self.FFphivec_sphere = np.arange(0, 360+spacing, spacing) # Do include phi = 360 - otherwise we get a chunk of white in the missing angle
        self.FFtheta_sphere, self.FFphi_sphere = np.meshgrid(self.FFthetavec_sphere, self.FFphivec_sphere, indexing='ij')
        
        print('Attaching FF S-parameters to meshgrid...')
        self.FFS21_sphere = np.zeros((2, np.size(self.FFthetavec_sphere), np.size(self.FFphivec_sphere)), dtype=complex) ## pol (theta, then phi), theta, and phi angle
        self.FFS21_sphere = SNFAMFunctions.findFarField(self.Ts, self.FFthetavec_sphere*pi/180, self.FFphivec_sphere*pi/180) ## pol (theta, then phi), theta, and phi angle
        FFS = self.FFS21_sphere
        ## top half sphere contour plot
        if(pol==0):
            lvls = 20*np.log10(np.abs(FFS[0])+1e-12)
            titleAdd = ', Theta-pol (Vert.)'
        elif(pol==1):
            lvls = 20*np.log10(np.abs(FFS[1])+1e-12)
            titleAdd = ', Phi-pol (Horiz.)'
        elif(pol==2):
            lvls = 20*np.log10(np.sqrt(np.abs(FFS[0])**2+np.abs(FFS[1])**2)+1e-12)
            titleAdd = ' Magnitude'
            
        if(normalize):
            lvls = lvls - lvls.max()
            
        if(levels == None):
            levels = lvls.max() + np.linspace(-50, 0, Nlevels)
            
        figsizex = 7.35
        if(colourbar): ### more x-space for colorbar to fit
            figsizex+=1.15
        plt.figure(figsize = (figsizex,7))
        cnt = plt.contourf(np.sin(pi/180*self.FFtheta_sphere)*np.cos(pi/180*self.FFphi_sphere), np.sin(pi/180*self.FFtheta_sphere)*np.sin(pi/180*self.FFphi_sphere), lvls, levels=levels, extend='both')
        ## plotting it twice makes the contour borderlines harder to see - also takes double filesize
        #cnt.set_edgecolor("face") ## so the contour borderlines cannot be seen
        #cnt.set_linewidth(0.000000000001) ## try savefig(dpi=400), or other dpi amount - may make image better, should be equivalent to just changing fig size
        #plt.plot(np.sin(10*pi/180)*np.cos(self.FFphivec_sphere), np.sin(10*pi/180)*np.sin(self.FFphivec_sphere), '-w', linewidth = 0.1) ## 10 degree circle
        v = np.linspace(levels.min(), levels.max(), 10, endpoint=True)
        if(colourbar):
            plt.colorbar(ticks=v)
        #plt.xlabel('Horizontal')
        #plt.ylabel('Vertical')
        plt.title(f'{self.name} Far-field'+titleAdd)
        plt.axis('square')
        #plt.axis('off') ## removes labels, border, and ticks
        plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        plt.tight_layout()
        
        ## also plot some planes - 2 theta cuts at phi angles
        val1 = 0 ## compare between axis 2 = val1 and axis 2 = phi2
        val2 = 90 ##
        
        if(plotCuts):
            Nplot = 1000 ## number of pts to calculate
            ###
            ### THETA CUTS
            ###
            print(f'Computing farfield theta sweep')
            patts = []
            thetaplot = np.linspace(-180, 180, Nplot)
            phivec1 = np.ones(Nplot)*val1*pi/180
            phivec2 = np.ones(Nplot)*val2*pi/180
            phivec1[thetaplot<0] = val1 + pi
            phivec2[thetaplot<0] = val2 + pi
            thetavec = np.abs(thetaplot)*pi/180
            FF1 = SNFAMFunctions.findFarField(self.Ts, thetavec, phivec1)
            FF2 = SNFAMFunctions.findFarField(self.Ts, thetavec, phivec2)
            F_abs1 = np.sqrt(np.abs(FF1[0])**2 + np.abs(FF1[1])**2)
            F_abs2 = np.sqrt(np.abs(FF2[0])**2 + np.abs(FF2[1])**2)
            patts.append([[np.abs(FF1[0]), thetavec], [np.abs(FF1[1]), thetavec], self.name+'{:.2f}'.format(self.fs[freq]/1e9)+'GHz, theta pol'])
            patts.append([[np.abs(FF2[0]), thetavec], [np.abs(FF2[1]), thetavec], self.name+'{:.2f}'.format(self.fs[freq]/1e9)+'GHz, phi pol'])
            patts.append([[F_abs1, thetavec], [F_abs2, thetavec], self.name+'{:.2f}'.format(self.fs[freq]/1e9)+'GHz, abs'])
            plotPatterns(patts, normalize, cuts=[val1,val2])
            
            
            ###
            ### PHI CUTS
            ###
            print(f'Computing farfield phi sweep')
            patts = []
            phiplot = np.linspace(-180, 180, Nplot)*pi/180
            thetavec1 = np.ones(Nplot)*val1*pi/180 ## assume these are positive, so no need to wrap to positive
            thetavec2 = np.ones(Nplot)*val2*pi/180
            FF1 = SNFAMFunctions.findFarField(self.Ts, thetavec1, phiplot)
            FF2 = SNFAMFunctions.findFarField(self.Ts, thetavec2, phiplot)
            F_abs1 = np.sqrt(np.abs(FF1[0])**2 + np.abs(FF1[1])**2)
            F_abs2 = np.sqrt(np.abs(FF2[0])**2 + np.abs(FF2[1])**2)
            patts.append([[np.abs(FF1[0]), phiplot], [np.abs(FF1[1]), phiplot], self.name+'{:.2f}'.format(self.fs[freq]/1e9)+'GHz, theta pol'])
            patts.append([[np.abs(FF2[0]), phiplot], [np.abs(FF2[1]), phiplot], self.name+'{:.2f}'.format(self.fs[freq]/1e9)+'GHz, phi pol'])
            patts.append([[F_abs1, phiplot], [F_abs2, phiplot], self.name+'{:.2f}'.format(self.fs[freq]/1e9)+'GHz, abs'])
            plotPatterns(patts, normalize, cuts=[val1, val2], axes = [r'$\phi$', r'$\theta$'])
            
        if(plotContours):
            self.plotContours(self.FFS21_sphere, FF = True)
        
    def plotContoursPreProcessing(self, data, levels = None, Nlevels = 22, pol = 2, phase = 0, FF = False, normalize=False): ### plots the unprocessed data
        ## data is in the Format: # of triggers, time since start, probe pol, theta angle [degrees], phi angle, f1 [Hz], S21(f1), f2, S21(f2), f3, S21(f3)

        Ss = data[:,8] ## S21
        pols = np.real(data[:, 2]) ## pp
        uniquethetas = np.real(np.unique(data[:,3]))
        uniquephis= np.real(np.unique(data[:,4]))
        thetas, phis = np.meshgrid(uniquethetas, uniquephis, indexing='ij')
        S21s = np.zeros((np.size(uniquethetas), np.size(uniquephis)))
        #### iterate through the data points, adding each one to the appropriate spot in S21s
        for i in range(len(data)): ## iterate over all data
            t, p = data[i, 3:5] ## theta, phi
            n_t = np.argmin(np.abs(uniquethetas-t))
            n_p = np.argmin(np.abs(uniquephis-p))
            
            if(pol==0): ## pp=0
                if(pols[i] == 0): ## every duplicate-angled point should just be a different polarization
                    S21s[n_t,n_p] = 20*np.log10(np.abs(Ss[i])) ## magnitudes, in dB
            elif(pol==1): ## pp=90
                if(pols[i] == 90): ## every duplicate-angled point should just be a different polarization
                    S21s[n_t,n_p] = 20*np.log10(np.abs(Ss[i])) ## magnitudes, in dB
            elif(pol==2): ## magnitude
                if(S21s[n_t,n_p] == 0): ## every duplicate-angled point should just be a different polarization
                    S21s[n_t,n_p] = np.abs(Ss[i]) ## magnitudes, in dB
                else: ## dupe
                    S21s[n_t,n_p] = 20*np.log10(np.sqrt(np.abs(S21s[n_t,n_p])**2+np.abs(Ss[i])**2)) ## magnitudes, in dB
                
        idx0 = np.isclose(S21s, 0, atol=1e-4) ## atol chosen by necessity... presumably any real measurement has much lower levels than ~~0
        S21s[idx0] = 20*np.log10(S21s[idx0]+1e-12) ## to take care of unmeasured data points
        
        if(levels == None):
            levels = S21s.max() + np.linspace(-50, 0, Nlevels)
            
        if(normalize):
            lvls = lvls - lvls.max()
            
        plt.figure(figsize = (10,7))
        cnt = plt.contourf(thetas, phis, S21s, levels=levels, extend='both')
        cnt.set_edgecolor("face") ## so the contour borderlines cannot be seen
        cnt.set_linewidth(0.000000000001)
        
        plt.colorbar()
        plt.xlabel(r'$\theta$-angle (degrees)')
        plt.ylabel(r'$\phi$-angle (degrees)')
        plt.title(f'{self.name} Near-field Magnitude, pol'+str(pol))
        plt.tight_layout()
        plt.plot()
        plt.show()
        
    def plotContours(self, Ss, freq = 1, levels = None, Nlevels = 22, pol = 2, phase = 0, FF = False): ### plots the data
        ### choose frequency, possibly send in levels for consistent plotting, and which pol to plot (0=theta, 1=phi, 2=magnitude). phase = 1 to plot phase instead. FF to use farfield theta/phi vecs
        if(pol==0):
            lvls = 20*np.log10(np.abs(Ss[0])+1e-12)*(1-phase) + phase*np.angle(Ss[0])
            titleAdd = ', Theta-pol (Vert.)'
        elif(pol==1):
            lvls = 20*np.log10(np.abs(Ss[1])+1e-12)*(1-phase) + phase*np.angle(Ss[1])
            titleAdd = ', Phi-pol (Horiz.)'
        elif(pol==2):
            lvls = 20*np.log10(np.sqrt(np.abs(Ss[0])**2+np.abs(Ss[1])**2)+1e-12)*(1-phase) + phase*np.angle(Ss[0]+Ss[1])
            titleAdd = ', Abs. Value.'
        
        if(levels == None and phase == 0):
            levels = lvls.max() + np.linspace(-50, 0, Nlevels)
        elif(levels == None and phase == 1):
            levels = lvls.max() + np.linspace(-2*pi, 0, Nlevels)
            
        plt.figure(figsize = (10,7))
        if(FF):
            plt.contourf(self.FFtheta_sphere, self.FFphi_sphere, lvls, levels=levels, extend='both')
        else:
            plt.contourf(self.theta_sphere, self.phi_sphere, lvls, levels=levels, extend='both')
        plt.colorbar()
        plt.xlabel('theta (degrees)')
        plt.ylabel('phi (degrees)')
        plt.title(f'{self.name} NF, frequency {self.fs[freq]*1e-9:.4} GHz')
        plt.tight_layout()
        
    def plotContoursProjected(self, Ss, freq = 1, levels = None, Nlevels = 45, pol = 2, extraText = ''): ### plots the top-half of the sphere (theta between 0,90 and phi from 0,360)
        ### choose frequency, possibly send in levels for consistent plotting, and which pol to plot (0=theta, 1=phi, 2=magnitude)
        if(pol==0):
            lvls = 20*np.log10(np.abs(Ss[0])+1e-12)
            titleAdd = ', Theta-pol (Vert.)'
        elif(pol==1):
            lvls = 20*np.log10(np.abs(Ss[1])+1e-12)
            titleAdd = ', Phi-pol (Horiz.)'
        elif(pol==2):
            lvls = 20*np.log10(np.sqrt(np.abs(Ss[0])**2+np.abs(Ss[1])**2)+1e-12)
            titleAdd = ', Abs. Value'
            
        if(levels == None):
            levels = lvls.max() + np.linspace(-50, 0, Nlevels)
            
        
        nThetas = int(np.size(self.thetavec_sphere)/2) ## to use half the theta angles (this should then go only from 0 to 90 degrees)
        plt.figure(figsize = (8.5,7))
        cnt = plt.contourf(np.sin(pi/180*self.theta_sphere[:nThetas,:])*np.cos(pi/180*self.phi_sphere[:nThetas,:]), np.sin(pi/180*self.theta_sphere[:nThetas,:])*np.sin(pi/180*self.phi_sphere[:nThetas,:]), lvls[:nThetas,:], levels=levels, extend='both')
        #cnt.set_edgecolor("face") ## so the contour borderlines cannot be seen
        #cnt.set_linewidth(0.000000000001)
        v = np.linspace(levels.min(), levels.max(), 10, endpoint=True)
        plt.colorbar(ticks = v)
        #plt.xlabel('Horizontal')
        #plt.ylabel('Vertical')
        plt.title(f'{self.name} NF, f={self.fs[freq]*1e-9:.4} GHz'+titleAdd+extraText)
        plt.axis('square')
        plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        plt.tight_layout()
        plt.gca().set_aspect(1)
            
def plotPlanes(datas, fs = [1], pPs = []): ## plots the phi = 0 and phi = 90 planes for both probe pols. Can send in phase factors for each dataset
        val1 = 0 ## compare between axis 2 = val1 and axis 2 = phi2
        val2 = 90 ## make sure this is actually a measured value, since the plot will say it's this value anyway
        ###
        ### PHI CUTS
        ###
        for i in fs: ## plot each freq, or probably just the center one
            patts = []
            for data in datas:
                for pp in (0,90):
                    thetas = data.pos[1][(np.abs(data.pos[0]-pp)*1e3 + data.pos[2]*1 + data.pos[1]*1e-3).argsort()]
                    SPhi1 = data.S21s[i][(np.abs(data.pos[0]-pp)*1e3 + np.abs(data.pos[2]-val1)*1 + data.pos[1]*1e-3).argsort()] ## add the different arguments to sort by with large magnitudes so they dont get mixed up
                    SPhi2 = data.S21s[i][(np.abs(data.pos[0]-pp)*1e3 + np.abs(data.pos[2]-val2)*1 + data.pos[1]*1e-3).argsort()]
                    
                    Nthetas = np.size(data.thetaRange)
                    patts.append([[np.abs(SPhi1[0:Nthetas]), thetas[0:Nthetas]*pi/180], [np.abs(SPhi2[0:Nthetas]), thetas[0:Nthetas]*pi/180], data.name+'{:.2f}'.format(data.fs[i]/1e9)+'GHz,Pp='+str(pp)])
            plotPatterns(patts, normalize = False, cuts=[val1,val2])
            
        ###
        ### THETA CUTS
        ###
        for i in fs: ## plot each freq
            patts = []
            for data in datas:
                for pp in (0,90):
                    phis = data.pos[2][(np.abs(data.pos[0]-pp)*1e3 + np.abs(data.pos[2])*1e-3 + data.pos[1]*1).argsort()]
                    SPhi1 = data.S21s[i][(np.abs(data.pos[0]-pp)*1e3 + np.abs(data.pos[2])*1e-3 + np.abs(data.pos[1]-val1)*1).argsort()] ## add the different arguments to sort by with large magnitudes so they dont get mixed up
                    SPhi2 = data.S21s[i][(np.abs(data.pos[0]-pp)*1e3 + np.abs(data.pos[2])*1e-3 + np.abs(data.pos[1]-val2)*1).argsort()]
                    
                    Nphis = np.size(data.phiRange)
                    patts.append([[np.abs(SPhi1[0:Nphis]), phis[0:Nphis]*pi/180], [np.abs(SPhi2[0:Nphis]), phis[0:Nphis]*pi/180], data.name+'{:.2f}'.format(data.fs[i]/1e9)+'GHz,Pp='+str(pp)])
            plotPatterns(patts, normalize = False, cuts=[val1, val2], axes = [r'$\phi$', r'$\theta$'])
            
        if(len(pPs) != 0): ## non-empty list, so plot sum of patterns with phase factors applied
            ### PHI CUTS
            for i in fs: ## plot each freq, or probably just the center one
                patts = []
                for pp in (0,90):
                    for d in range(len(datas)):
                        data = datas[d]
                        pP = np.exp(1j*pPs[d])
                        if(d==0): ## first one, initialize
                            thetas = data.pos[1][(np.abs(data.pos[0]-pp)*1e3 + data.pos[2]*1 + data.pos[1]*1e-3).argsort()]
                            SPhi1 = data.S21s[i][(np.abs(data.pos[0]-pp)*1e3 + np.abs(data.pos[2]-val1)*1 + data.pos[1]*1e-3).argsort()]*pP ## add the different arguments to sort by with large magnitudes so they dont get mixed up
                            SPhi2 = data.S21s[i][(np.abs(data.pos[0]-pp)*1e3 + np.abs(data.pos[2]-val2)*1 + data.pos[1]*1e-3).argsort()]*pP
                        else: ## add another pattern
                            SPhi1 += data.S21s[i][(np.abs(data.pos[0]-pp)*1e3 + np.abs(data.pos[2]-val1)*1 + data.pos[1]*1e-3).argsort()]*pP ## add the different arguments to sort by with large magnitudes so they dont get mixed up
                            SPhi2 += data.S21s[i][(np.abs(data.pos[0]-pp)*1e3 + np.abs(data.pos[2]-val2)*1 + data.pos[1]*1e-3).argsort()]*pP
                            
                    Nthetas = np.size(data.thetaRange)
                    patts.append([[np.abs(SPhi1[0:Nthetas]), thetas[0:Nthetas]*pi/180], [np.abs(SPhi2[0:Nthetas]), thetas[0:Nthetas]*pi/180], data.name+'{:.2f}'.format(data.fs[i]/1e9)+'GHz,Pp='+str(pp)])
                plotPatterns(patts, normalize = False, cuts=[val1,val2])
                
            #===================================================================
            # ### THETA CUTS
            # for i in fs: ## plot each freq
            #     patts = []
            #     for pp in (0,90):
            #         for d in range(len(datas)):
            #             data = datas[d]
            #             pP = pPs[d]
            #             if(d==0): ## first one, initialize
            #                 phis = data.pos[2][(np.abs(data.pos[0]-pp)*1e3 + np.abs(data.pos[2])*1e-3 + data.pos[1]*1).argsort()]
            #                 SPhi1 = data.S21s[i][(np.abs(data.pos[0]-pp)*1e3 + np.abs(data.pos[2])*1e-3 + np.abs(data.pos[1]-val1)*1).argsort()]*pP ## add the different arguments to sort by with large magnitudes so they dont get mixed up
            #                 SPhi2 = data.S21s[i][(np.abs(data.pos[0]-pp)*1e3 + np.abs(data.pos[2])*1e-3 + np.abs(data.pos[1]-val2)*1).argsort()]*pP
            #             else: ## add another pattern
            #                 SPhi1 += data.S21s[i][(np.abs(data.pos[0]-pp)*1e3 + np.abs(data.pos[2])*1e-3 + np.abs(data.pos[1]-val1)*1).argsort()]*pP ## add the different arguments to sort by with large magnitudes so they dont get mixed up
            #                 SPhi2 += data.S21s[i][(np.abs(data.pos[0]-pp)*1e3 + np.abs(data.pos[2])*1e-3 + np.abs(data.pos[1]-val2)*1).argsort()]*pP
            #                 
            #         Nphis = np.size(data.phiRange)
            #         patts.append([[np.abs(SPhi1[0:Nphis]), phis[0:Nphis]*pi/180], [np.abs(SPhi2[0:Nphis]), phis[0:Nphis]*pi/180], data.name+'{:.2f}'.format(data.fs[i]/1e9)+'GHz,Pp='+str(pp)])
            #     plotPatterns(patts, normalize = False, cuts=[val1, val2], axes = [r'$\phi$', r'$\theta$'])
            #===================================================================

def calcBroadside(phases, datas, angle): ## given some datas and relative phases, calculates the sum of broadsides at phi=0 and phi=90+. For optimiziation of phases.
    Ssum0 = 0
    Ssum90 = 0
    for i in range(len(datas)):
        data = datas[i]
        p = np.exp(1j*phases[i])
        Ssum0 += data.S21s[1][np.argmin(np.abs(data.pos[0]-0) + np.abs(data.pos[2]-angle[1]) + np.abs(data.pos[1]-angle[0]))]*p ## horiz. probe pol => phi = 0
        Ssum90 += data.S21s[1][np.argmin(np.abs(data.pos[0]-90) + np.abs(data.pos[2]-(90+angle[1])) + np.abs(data.pos[1]-angle[0]))]*p ## vert. probe pol => phi = 90
    #print(np.abs(Ssum0)+np.abs(Ssum90))
    return -(np.abs(Ssum0)+np.abs(Ssum90))
        
def plotCutsForSettingsTest(SNFDatas): ## takes a list of SNFDatas, plots cuts together similar to above
    Nthetas = 201 ## take the first 201 theta pts after sorting to get the full range.
    Nphis = 241
    val1 = 0 ## compare between axis 2 = val1 and axis 2 = phi2
    val2 = 90 ## make sure this is actually a measured value, since the plot will say it's this value anyway
    
    ###
    ### THETA CUTS
    ###
    pp = 90
    for i in [1]: ## plot each freq, or probably just the center one
        patts = []
        for SNFDat in SNFDatas:
            thetas = SNFDat.pos[1][(np.abs(SNFDat.pos[0]-pp)*1e3 + np.abs(SNFDat.pos[2]-val1)*1 + SNFDat.pos[1]*1e-3).argsort()]
            SPhi1 = SNFDat.S21s[i][(np.abs(SNFDat.pos[0]-pp)*1e3 + np.abs(SNFDat.pos[2]-val1)*1 + SNFDat.pos[1]*1e-3).argsort()] ## add the different arguments to sort by with large magnitudes so they dont get mixed up
            SPhi2 = SNFDat.S21s[i][(np.abs(SNFDat.pos[0]-pp)*1e3 + np.abs(SNFDat.pos[2]-val2)*1 + SNFDat.pos[1]*1e-3).argsort()]
             
             
             
            patts.append([[np.abs(SPhi1[0:Nthetas]), thetas[0:Nthetas]*pi/180], [np.abs(SPhi2[0:Nthetas]), thetas[0:Nthetas]*pi/180], SNFDat.name+'{:.2f}'.format(SNFDat.fs[i]/1e9)+'GHz,Pp='+str(pp)])
        plotPatterns(patts, normalize = False, cuts=[val1,val2])

def findFarFieldAtDistance(k, r, theta, phi, B1, B2, N): ## takes a wavenumber k, a radius r with angles theta and phi for the far-field position, and m*n spherical mode coefficients B1 and B2, N number of modes to use
                             ## returns electric field intensity on a sphere at distance r, along with a vector containing the phi and theta angles
                             ## follows eqn. 8.5a in [1]   
                             ## Should update this if going to use it.
    E = k/np.sqrt(eta_0)
    sum = 0
    c = 0
    for n in range(1,N+1):
        for m in range(-n,n+1):
            sum += B1[c]*M4_mn(k,r,theta,phi,m,n) + B2[c]*N4_mn(k,r,theta,phi,m,n)
            c+=1
    return E*sum

def findFarField(k, r, theta, phi, B1, B2, N):  ## Asymptotic far-field
                             ## takes a wavenumber k, a radius r with angles theta and phi for the far-field position, and m*n spherical mode coefficients B1 and B2, N number of modes to use
                             ## returns electric field intensity on a sphere at distance r, along with a vector containing the phi and theta angles
                             ## follows eqn. 8.5a in [1]   
    E = k/np.sqrt(eta_0)
    sum = 0
    c = 0
    for n in range(1,N+1):
        for m in range(-n,n+1):
            sum += B1[c]*M4_mn(k,r,theta,phi,m,n) + B2[c]*N4_mn(k,r,theta,phi,m,n)
            c+=1
    return E*sum
    
def plotPatterns(patterns, normalize = 'each', cuts=[0, 90], axes = [r'$\theta$', r'$\phi$']): ## takes a list of patterns given by an array of intensities at e.g. theta angles along phi cuts, the theta values, and a name [[[I_phi=0, thetas],[I_phi=90, thetas],'name'],...]
                             ## plots two theta-cuts of the far-field for each pattern side-by-side. Can plot normalized to the max value of all patterns (together), or each pattern individually (each)
    
    ymin = -40
    ymax = 1
    #yticks = [-30,-20,-15,-10,-5,0,5,10,15,20]
    fig = plt.figure(figsize = (12,9))
    ax1a = plt.subplot(2, 1, 1)
    plt.xlabel(axes[0]+' [degrees]')
    plt.ylabel(r'Gain [dB]')
    plt.grid(True)
    #plt.gca().set_xticks([-90, -60, -30, 0, 30, 60, 90])
    #plt.gca().set_yticks(yticks)
    if(normalize!=False):
        plt.ylim(ymin, ymax)
        plt.title(f'Normalized Gain - '+r'('+str(axes[1])+'='+str(cuts[0])+'$^\circ$ cut)')
    else:
        plt.title(f'Gain - '+r'('+str(axes[1])+'='+str(cuts[0])+'$^\circ$ cut)')
    ax1b = plt.subplot(2, 1, 2)
    
    c=0
    
    maxForNorm = 0
    for pattern in patterns: ## the normalization factor is the highest intensity from all patterns. So, similar quantities should be plotted
        if(normalize == 'together'):
            testVal = np.max(np.hstack((pattern[0][0],pattern[1][0]))) ## check both phi-angle bits
            if(testVal > maxForNorm):
                maxForNorm = testVal
                
    for pattern in patterns:
        if(normalize == 'together'):
            normFactor = 1 /(maxForNorm) ## 1/highest value
        elif(normalize == 'each'):
            normFactor = 1 /(np.max(np.hstack((pattern[0][0],pattern[1][0])))) ## 1/highest value
        else:
            normFactor = 1 ##does nothing
        ax1a.plot(pattern[0][1]*180/pi, 20*np.log10(pattern[0][0]*normFactor), linestyle = linestyles[c%len(linestyles)], color = colors[c%len(linestyles)], label = pattern[2]) #1-cut
        ax1b.plot(pattern[1][1]*180/pi, 20*np.log10(pattern[1][0]*normFactor), linestyle = linestyles[c%len(linestyles)], color = colors[c%len(linestyles)], label = pattern[2]) #2-cut
        c+=1
        
    plt.xlabel(axes[0]+' [degrees]')
    plt.ylabel(r'Gain [dB]')
    plt.grid(True)
    #plt.gca().set_xticks([-90, -60, -30, 0, 30, 60, 90])
    #plt.gca().set_yticks(yticks)
    if(normalize!=False):
        plt.ylim(ymin, ymax)
        plt.title(f'Normalized Gain - '+r'('+str(axes[1])+'='+str(cuts[1])+'$^\circ$ cut)')
    else:
        plt.title(f'Gain - '+r'('+str(axes[1])+'='+str(cuts[1])+'$^\circ$ cut)')
    #ax1a.set_yticklabels(ax1a.get_yticks(), rotation=0, weight='bold')
    #ax1a.set_xticklabels(ax1a.get_xticks(), rotation=0, weight='bold')
    #ax1b.set_yticklabels(ax1b.get_yticks(), rotation=0, weight='bold')
    #ax1b.set_xticklabels(ax1b.get_xticks(), rotation=0, weight='bold')
    handles, labels = ax1a.get_legend_handles_labels()
    ax1b.legend(handles, labels, loc = 'best', ncol=2, fontsize = 17, framealpha=0.45)
    #ax1a.set_ylim(ymin, ymax)
    #ax1b.set_ylim(ymin, ymax)
    fig.tight_layout()
    #plt.show()
    
def patternFromVals(pos, Es, name): ## makes a pattern for plotPatterns from data - phi = 0 and 45 degree cuts 
    ## takes a data vector and a name, each row is a data point, pattern is the E-data along cut 1 and 2, the theta angles, and a string for a name
    phi1 = np.transpose(pos)[2][np.argmin(np.abs(np.transpose(pos)[2]-0)+np.transpose(pos)[1]*1e-6)] ## add a small theta component to order in both phi and theta... even though this does nothing here.
    phi2 = np.transpose(pos)[2][np.argmin(np.abs(np.transpose(pos)[2]-pi/4)+np.transpose(pos)[1]*1e-6)]
    Es1 = []
    Es2 = []
    thetaVals1 = []
    thetaVals2 = []
    for i in range(len(pos)):
        if(pos[i][2] == phi1): ## if phi angle matches
            Es1.append(Es[i])
            thetaVals1.append(pos[i,1])
            
        if(pos[i][2] == phi2):
            Es2.append(Es[i])
            thetaVals2.append(pos[i,1])
    
    pattern = [[np.linalg.norm(np.array(Es1), axis=1)**2, np.array(thetaVals1)], [np.linalg.norm(np.array(Es2), axis=1)**2,np.array(thetaVals2)], name]
    return pattern
    
def EDip(p,pos,f): 
    ## gives the E-field of a z-directed Hertzian dipole at positions p (frequency f)
    # Returns the E-field at the position as an array of [E_r, E_theta, E_phi]
    k = 2*pi*f/c
    r = pos[:,0]
    theta = pos[:,1]
    phi = pos[:,2]
    
    E_r = 1j*f*p*eta_0*( 1/r - 1j/(k*r**2) )*np.exp(-1j*k*r)/r*np.cos(theta)
    E_theta = -1*p*mu_0*f**2/pi* ( 1 + 1/(1j*k*r) - 1/(k*r)**2 )*np.exp(-1j*k*r)/r*np.sin(theta)
    
    return np.transpose(np.array([E_r,E_theta,np.zeros(len(E_r))]))
                             
def ElectricDipolePattern(p, r, f): ##calculates the electric field intensity from a z-directed dipole
    ## takes z-directed dipole moment p, frequency f, and plots the pattern at distance r
    phi1 = 0 ## cut 1
    phi2 = 90*pi/180 ## cut 2
    thetas = np.linspace(-180,180,20000)*pi/180 ## array of theta vals to calculate the pattern at
    rs1 = np.zeros((len(thetas),3))
    rs1[:,1] = thetas[:]
    rs1[:,2] = phi1
    rs1[:,0] = r
    rs2 = np.zeros((len(thetas),3))
    rs2[:,1] = thetas[:]
    rs2[:,2] = phi2
    rs2[:,0] = r
    
    Es1 = EDip(p, rs1, f)
    Es2 = EDip(p, rs2, f)
    
    pattern = [[np.linalg.norm(Es1, axis=1)**2, thetas], [np.linalg.norm(Es2, axis=1)**2,thetas] , 'Dipole pattern at r='+str(r)+' m']
    return pattern

def plotFarFieldCut(data, phi, normalize = True, pol = 2, spacing = .1, freq = 1): ## uses calculated transmission coefficients to calculate the far-field. Only plots the curves, doesnt make a figure
    ## Plots two theta-cuts at given phi val1 and val2. Points spaced evenly as given spacing. H and V planes default. Plots in dB, normalize sets max magnitude to 0
    data.FFthetavec_sphere = np.arange(0, 90+spacing, spacing) # Include theta = 180
    data.FFphivec_sphere = np.arange(0, 360, spacing) # Do not include phi = 360
    data.FFtheta_sphere, data.FFphi_sphere = np.meshgrid(data.FFthetavec_sphere, data.FFphivec_sphere, indexing='ij')
    
    print(f'Computing farfield theta sweep at phi = {phi}')
    patts = []
    thetaplot = np.arange(-100, 100, spacing)
    phivec = np.ones(np.size(thetaplot))*phi*pi/180
    
    phivec[thetaplot<0] = phi*pi/180 + pi
    thetavec = np.abs(thetaplot)*pi/180
    
    FF1 = SNFAMFunctions.findFarField(data.Ts, thetavec, phivec)
    F_abs = 20*np.log10(np.sqrt(np.abs(FF1[0])**2 + np.abs(FF1[1])**2))
    
    if(normalize):
        max = np.max(F_abs)
        F_abs = F_abs-max
    plt.plot(thetaplot, F_abs, label = f'SNFFT $\phi = {phi}^\circ$')

def plotVsSlotPlanes(datas = [], alsoNF = True, freq = 1, normalize=True): ## takes data with Tsmn, plots far-field V and H planes against digitized values made from the plots from SAAB. Will also plot near-field measurements directly for comparison if true
    dataV = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/antenna measurements/SNFAM/Digitized Slot Array 9.35GHz/Vplane.txt', delimiter = ',', skiprows = 1))
    dataH = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/antenna measurements/SNFAM/Digitized Slot Array 9.35GHz/Hplane.txt', delimiter = ',', skiprows = 1))
    if(normalize):
        dataV[1] = dataV[1]-np.max(dataV[1])
        dataH[1] = dataH[1]-np.max(dataH[1])
    
    plt.figure() ## V-PLANE
    plt.plot(-dataV[0],dataV[1], label = 'V-Plane: Digitized Ref', linewidth = 2.5) ## -theta angles here since their measurement seems to have had it the other way (this makes our FF match theirs)
    phi1 = 90
    for data in datas:
        plotFarFieldCut(data, phi=phi1,normalize=normalize)
        if(alsoNF): ## plot the near-field
            Nthetas = np.size(data.thetaRange)
            SThetasPP0 = data.S21s[freq][(np.abs(data.pos[0]-0)*1e3 + np.abs(data.pos[2]-phi1+data.phiSpacing/5)*1 + (data.pos[1])*1e-3).argsort()][0:Nthetas] ## NF phi can't be 90/270 - must be multiple of 0.8
            SThetasPP90 = data.S21s[freq][(np.abs(data.pos[0]-90)*1e3 + np.abs(data.pos[2]-phi1+data.phiSpacing/5)*1 + (data.pos[1])*1e-3).argsort()][0:Nthetas] ## so I add part of phiSpacing
            NF_abs = 20*np.log10(np.sqrt(np.abs(SThetasPP0)**2 + np.abs(SThetasPP90)**2))
            if(normalize):
                NF_abs = NF_abs-np.max(NF_abs)
            plt.plot(data.thetaRange, NF_abs, label = f'NF $\phi = {phi1}^\circ$', linestyle = '--')
    plt.grid()
    plt.legend()
    plt.title('Normalized SNFM vs Given V Planes', fontsize=20)
    plt.xlabel('Elevation Angle [degrees]', fontsize=18)
    plt.ylabel('Amplitude [dB]', fontsize=18)
    plt.tight_layout()
    plt.ylim(-65,3)
    plt.xlim(-90,90)
    
    plt.figure() ## H-PLANE
    plt.plot(dataH[0],dataH[1], label = 'H-Plane: Digitized Ref', linewidth = 2.5)
    phi2 = 0
    for data in datas:
        plotFarFieldCut(data, phi=phi2,normalize=normalize)
        if(alsoNF):
            Nthetas = np.size(data.thetaRange)
            SThetasPP0 = data.S21s[freq][(np.abs(data.pos[0]-0)*1e3 + np.abs(data.pos[2]-phi2)*1 + (data.pos[1])*1e-3).argsort()][0:Nthetas]
            SThetasPP90 = data.S21s[freq][(np.abs(data.pos[0]-90)*1e3 + np.abs(data.pos[2]-phi2)*1 + (data.pos[1])*1e-3).argsort()][0:Nthetas]
            NF_abs = 20*np.log10(np.sqrt(np.abs(SThetasPP0)**2 + np.abs(SThetasPP90)**2))
            if(normalize):
                NF_abs = NF_abs-np.max(NF_abs)
            plt.plot(data.thetaRange, NF_abs, label = f'NF $\phi = {phi2}^\circ$', linestyle = '--')
    plt.grid()
    plt.legend()
    plt.title('Normalized SNFM vs Given H Planes', fontsize=20)
    plt.xlabel('Elevation Angle [degrees]', fontsize=18)
    plt.ylabel('Amplitude [dB]', fontsize=18)
    plt.tight_layout()
    plt.ylim(-65,3)
    plt.xlim(-90,90)
    plt.show()
    
def plotS11s(): ## plots the measured S11s for the cloaked and uncloaked antennas over the slot array
    plt.figure()
    fileLoc = 'C:/Users/al8032pa/Work Folders/Documents/antenna measurements/SNFAM/Measured S11s/'
    loadUnc = np.loadtxt(fileLoc+'S11forUncloakedAntenna.csv', delimiter = ',', skiprows = 3)
    fsUnc, SsUnc = loadUnc[:,0], loadUnc[:,1]
    loadCl = np.loadtxt(fileLoc+'S11forCloakedAntenna.csv', delimiter = ',', skiprows = 3)
    fsCl, SsCl = loadCl[:,0], loadCl[:,1]
    
    plt.plot(fsUnc/1e9, SsUnc, label = 'Uncloaked Dipole', linewidth = 2.5)
    plt.plot(fsCl/1e9, SsCl, label = 'Cloaked Dipole', linewidth = 2.5)
    plt.axvline(x=1.045, color = 'gray', linestyle = '--', alpha = 0.5, linewidth = 2)
    plt.grid()
    plt.legend()
    plt.title(r'S$_{11}$ of Dipoles over Slot Array', fontsize=20)
    plt.xlabel('Frequency [GHz]', fontsize=18)
    plt.xlim(.8, 1.2)
    plt.ylabel('Reflection Coefficient [dB]', fontsize=18)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    begin = True
startTime = timer()
print('Start:')

#plt.rcParams.update({'font.size': 22})
#plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=27)     # fontsize of the axes title
plt.rc('axes', labelsize=27)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=30)  # fontsize of the figure title
#plt.rcParams['figure.dpi'] = 72
fileLoc = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Scripted Farfields/'

#plotS11s() ## plots measured S11 for a dipole of each type, placed over the slot array
#plotCSTData.plotLinesPaper2() ## plots of CST sims for slot array + lined dipoles

#===============================================================================
# ##TESTING PBAR DERIVS
# theta = np.linspace(0,180,num=10000)*pi/180
# cth = np.cos(theta)
# m = 1
# n = 1
# pbar = SNFAMFunctions.Pbar(m, n, cth)
# 
# dthetas = theta[0:len(theta)-1]
# cdthetas = np.cos(dthetas)
# dPbar = np.zeros((9999))
# for i in range(len(dthetas)):
#     dPbar[i] = (SNFAMFunctions.Pbar(m, n, cth[i+1]) - SNFAMFunctions.Pbar(m, n, cth[i]))/(theta[i+1]-theta[i])
# 
# plt.plot(cth,pbar, label = 'func')
# plt.plot(cdthetas,dPbar, label = 'actual deriv')
# plt.plot(cth, SNFAMFunctions.diePdiethetaAlt(n, m, cth) ,label = 'type 1')
# plt.plot(cth, SNFAMFunctions.diePdietheta(n, m, cth),label = 'type alt')
# plt.legend()
# plt.show()
#===============================================================================

nu = 1.045e9

print('{:.3e}'.format(nu/1e9)+' GHz info:')
lamb = c/nu

D = 0.7

ff1G = 2*D**2/lamb
a = D/2 ## radius of array-enclosing sphere
k = 2*pi*nu/c ## wavenumber
print('far-field dist: {:.5e}'.format(ff1G))
N = int(np.ceil(k*a + 10)) ## number of modes to use
J = 2*N*(N+2) ## total number of spherical mode coefficients, s = 1 or 2, n up to N, then m from -n to n
print('N = '+str(N))
print('J = '+str(J))


nu = 9.35e9
k = 2*pi*nu/c ## wavenumber
print('{:.3e}'.format(nu/1e9)+' GHz info:')
lamb = c/nu
ff10G = 2*D**2/lamb


print('far-field dist: {:.5e}'.format(ff10G))

#plotPatterns([ElectricDipolePattern(1e-6, .02, nu), ElectricDipolePattern(1e-6, .2, nu), ElectricDipolePattern(1e-6, 2, nu), ElectricDipolePattern(1e-6, 20000, nu)], normalize = 'each') ## E-dipole patterns at various radii

N = int(np.ceil(k*a + 10))-1 ## number of modes to use
J = 2*N*(N+2) ## total number of spherical mode coefficients, s = 1 or 2, n up to N, then m from -n to n
print('N = '+str(N))
print('J = '+str(J))

## scan area truncation
d = 3.4 ## measurement distance (of the measurement sphere)
r0 = a ## MRS for the measured antenna

theta_v = 90*pi/180 ## retrieved pattern angle
theta_m = theta_v + np.arcsin(r0/d) ## measured pattern angle
print(f'To retrieve pattern up to +- {theta_v*180/pi} degrees, must measure +- {theta_m*180/pi} degrees')



#===============================================================================
# ####
# #### test dipole far-field pattern vs NFFFT transforming the near-field
# ####
#   
# a = lamb/4 ## half wavelength dipole, why not
# N = int(np.ceil(k*a + 10)) ## number of modes to use
#  
# thetas = np.linspace(0+eps,pi-eps, num = N+1) ## theta samples
# phis = np.linspace(0+eps,2*pi-eps, num = 2*N+1) ## phi samples
# r = lamb*2 # near field distance for calculating the 'measured' fields
# ps = np.zeros((len(thetas)*len(phis),3)) ## positions for the samples
#   
# for p in range(len(phis)):
#     for t in range(len(thetas)):
#         ps[p*len(thetas)+t,0] = r
#         ps[p*len(thetas)+t,1] = thetas[t]
#         ps[p*len(thetas)+t,2] = phis[p]
#   
# nfEs = EDip(1e-6, ps, nu)
# SMCs = SNFAMFunctions.findSMCsNoProbe(nfEs, ps, k, N)
#   
# rff = 100 ## far-field distance to calculate the far fields at. Should not affect the pattern, just intensity
#   
# thetasP = np.arange(-pi+eps,pi-eps,.03) ## theta samples
# phisP = np.arange(0+eps,2*pi-eps,.03) ## phi samples
# psP = np.zeros((len(thetasP)*len(phisP),3)) ## positions for the samples
# for p in range(len(phisP)):
#     for t in range(len(thetasP)):
#         psP[p*len(thetasP)+t,0] = rff
#         psP[p*len(thetasP)+t,1] = thetasP[t]
#         psP[p*len(thetasP)+t,2] = phisP[p]
#           
#   
# ffEs = SNFAMFunctions.EFFAsymptotic(k, psP, SMCs, N) ## far-field electric fields
#   
# NFFFTpat = patternFromVals(psP, ffEs, 'NFFFT')
# plotPatterns([ElectricDipolePattern(1e-6, rff, nu), ElectricDipolePattern(1e-6, r, nu), NFFFTpat]) ## E-dipole patterns at various radii
#   
# ####
# #### end dipole test case
# ####
#===============================================================================


####
#### test array approximately by array of dipoles far-field pattern vs NFFFT transforming the near-field
####


####
#### end array of dipoles test case
####

#===============================================================================
# ####
# #### test finding probe coefficients with Algorithm 5 from [4]
# ####
#  
# a = 0.1/2 ## radius of probe-enclosing sphere
# k = 2*pi*nu/c ## wavenumber
#  
# thetas = np.linspace(0+eps,pi-eps, num = 20) ## theta samples
# phis = np.linspace(0+eps,2*pi-eps, num = 40) ## phi samples
# r = 1 # near field distance at which fields are 'measured'
# pos = np.zeros((len(thetas)*len(phis),3))
# for p in range(len(phis)):
#     for t in range(len(thetas)):
#         pos[p*len(thetas)+t,0] = r
#         pos[p*len(thetas)+t,1] = thetas[t]
#         pos[p*len(thetas)+t,2] = phis[p]
#  
# Np = int(np.ceil(k*a + 10)) ## number of probe modes to use
# Ai = SNFAMFunctions.AiMatrix(Np, Np, pos, k*r)
# Rguess = np.zeros((2*Np*(Np+2)), dtype=complex)
# Rguess[0] = 1
# ws = EDip(1e-6, pos, nu) ## fake measurements
# R = SNFAMFunctions.Algorithm5(Ai, ws, Rguess)
#  
# ####
# #### end array of Algorithm 5
# ####
#===============================================================================


#===============================================================================
# ####
# #### test the radiation pattern of a Hertzian dipole based on the 1 SMC. Test successful (need to be careful with python indices --> j indices)
# ####
# a = lamb/8 ## MRS, say lambda/8 arbitrarily
# N = int(np.ceil(k*a + 10)) ## number of modes to use
#   
# SMCs = SNFAMFunctions.SMCsHertzianDipole(N)
#    
# thetasP = np.arange(-pi+eps,pi-eps,.03) ## theta samples
# phisP = np.arange(0+eps,2*pi-eps,.03) ## phi samples
# psP = np.zeros((len(thetasP)*len(phisP),3)) ## positions for the samples
# for p in range(len(phisP)):
#     for t in range(len(thetasP)):
#         psP[p*len(thetasP)+t,0] = 100 ## irrelevent
#         psP[p*len(thetasP)+t,1] = thetasP[t]
#         psP[p*len(thetasP)+t,2] = phisP[p]
#            
#    
# ffEs = SNFAMFunctions.EFFAsymptotic(k, psP, SMCs, N) ## far-field electric fields
#    
# NFFFTpat = patternFromVals(psP, ffEs, 'NFFFT')
# plotPatterns([NFFFTpat, ElectricDipolePattern(1e-6, 1e10, nu)]) ## E-dipole patterns at various radii
# ####
# #### hertzian dipole SMC test case
# ####
#===============================================================================

#===============================================================================
# ####
# #### new (using Hansen formulas) test the radiation pattern of a Hertzian dipole based on the 1 SMC. Test 
# ####
# a = lamb/8 ## MRS, say lambda/8 arbitrarily
# N = int(np.ceil(k*a + 10)) ## number of modes to use
# mDist = a ## 'measurement' distance
# f = 9.35e9
# ### generate fake data over a measurement sphere
# thetas = np.arange(0, 180, 1)
# phis = np.arange(0, 360, 1)
# pts = [] ## the measurement points
# print('Creating fake data...')
# for t in thetas:
#     #print(t)
#     for p in phis:
#         E = EDip(1e-6, np.array([[mDist, t*pi/180, p*pi/180]]), f)[0]
#         pts.append([-1, -1, 0, t, p, -1, -1, f, E[1], -1, -1]) ## #of triggers, time since start, probe pol, theta angle [degrees], phi angle, f1 [Hz], S21(f1), f2, S21(f2), f3, S21(f3)
#         pts.append([-1, -1, 90, t, p, -1, -1, f, E[2], -1, -1]) ## phi-hat component
#         
# print('Fake data created. Finding Tsmn...')
#         
# fakeDat = SNFData(2*N*(N+2), mDist, 'folder', np.array(pts), name = 'HertzianDipole', fake=True)
# #fakeDat.plot()
# #plt.show()
# fakeDat.calcTsmn(fakeVals='Dipole')
# plotQs(fakeDat.Ts)
# fakeDat.calcFarField()
# fakeDat.plotFarField()
# plt.show()
#             
# thetasP = np.arange(-pi+eps,pi-eps,.03) ## theta samples
# phisP = np.arange(0+eps,2*pi-eps,.03) ## phi samples
# psP = np.zeros((len(thetasP)*len(phisP),3)) ## positions for the samples
# for p in range(len(phisP)):
#     for t in range(len(thetasP)):
#         psP[p*len(thetasP)+t,0] = 100 ## irrelevent
#         psP[p*len(thetasP)+t,1] = thetasP[t]
#         psP[p*len(thetasP)+t,2] = phisP[p]
#                     
# ####
# #### hertzian dipole SMC test case
# ####
#===============================================================================


#### measurement stuff
folder = 'C:/Users/al8032pa/Work Folders/Documents/antenna measurements/SNFAM/data files/'## folder containing datafiles
measDist = 3.40 ##approx distance between MRS for AUT and probe. Measure this more accurately later.

#===============================================================================
# ###cut sweep to test settings stuff:
# CTs = []
# for i in range(1,5):
#     files = ['9.35GHZArraycutstest'+str(i)+'.csv']
#     CTs.append(SNFData(J, measDist, folder, files, name = 'SlotArrayBareCT'+str(i)))
# plotCutsForSettingsTest(CTs)
# ###
#===============================================================================


#===============================================================================
# ## more pts and slow
# files = ['9.35GHzArrayPhiSweepManyMorePtsAndSlow.csv']
# sAsl = SNFData(J, measDist, folder, files, name = 'SlotArraySlowRun')
# sAsl.plot(duplicates='average',pol=0, freq = 1)
#===============================================================================

#===============================================================================
# ## more points data
# files = ['9.35GHzArrayPhiSweepManyPts.csv','9.35GHzArrayPhiSweepManyPtsb.csv','9.35GHzArrayPhiSweepManyPts2.csv','9.35GHzArrayPhiSweepManyPts2b.csv']
# sAPm = SNFData(J, measDist, folder, files, name = 'SlotArrayMoreData')
# sAPm.plot(pol=2,freq = 1)
# plt.show()
# #sAPm.calcTsmn()
# #sAPm.plotFarField()
#===============================================================================
 
#===============================================================================
# ### bare array, looking at sweeping theta first vs phi first
# files = ['9.35GHZArrayPhiSweep1.csv','9.35GHZArrayPhiSweep1bcropped.csv','9.35GHZArrayPhiSweep2cropped.csv','9.35GHZArrayPhiSweep2b.csv']
# sAPS = SNFData(J, measDist, folder, files, name = 'SlotArrayPhiSweep')
# sAPS.plot(pol=2)
# plt.show()
# files = ['9.35GHZArrayThetaSweep.csv','9.35GHZArrayThetaSweep2.csv']
# sATS = SNFData(J, measDist, folder, files, name = 'SlotArrayThetaSweep')
#===============================================================================



#===============================================================================
# files = ['9.35GHZArray1.csv','9.35GHZArray2.csv'] ### earlier try 1
# ## data in the form of a list containing arrays of: [numMeas, time, probe pol, f1, S21_f1, f2, S21_f2, f3, S21_f3]
# ### probe pol is either 0 (co-pol, horizontal) or 90 (cross-pol, vertical)
#   
# ## theta is azimuthal angle, phi is rotation around axis between the two.
#   
# sAB = SNFData(J, measDist, folder, files, name = 'SlotArrayBare', duplicates = 'average')
# sAB.plot()
# plt.show()
# sAB.calcTsmn()
# sAB.calcFarField()
# sAB.plotFarField()
# plt.show()
#===============================================================================

#===============================================================================
# ### new stuff for conference
# #===============================================================================
# # files = ['9.35GHzArray+moreuncloakedinmidpol90slowmodetest.csv','9.35GHzArray+moreuncloakedinmidpol0slowmodetest.csv'] ### recent many uncloakeds run
# # unc = SNFData(J, measDist, folder, files, name = 'SlowManyUncloaked')
# # #unc.plot(pol=2)
# # #plt.show()
# # unc.calcTsmnNoProbeCorrection()
# # unc.plotFarField()
# # plt.show()
# #===============================================================================
# 
# files = ['9.35GHzArray+moreuncloakedinmidpol90.csv','9.35GHzArray+moreuncloakedinmidpol0.csv'] ### recent many uncloakeds run
# unc = SNFData(J, measDist, folder, files, name = 'ManyUncloaked')
# unc.plot(pol=2)
# plt.show()
# unc.calcTsmnNoProbeCorrection()
# #plotQs(unc.Ts)
# unc.plotFarField()
# plt.show()
#      
# files = ['9.35GHZArray+cloaked.csv','9.35GHZArray+cloaked2.csv'] ### earlier try 1
# unc = SNFData(J, measDist, folder, files, name = 'SlotArray+Cloaked')
# unc.plot(pol=2)
# plt.show()
#        
# files = ['9.35GHZArray+uncloaked.csv','9.35GHZArray+uncloaked2.csv'] ### earlier try 1
# unc = SNFData(J, measDist, folder, files, name = 'SlotArray+Uncloaked')
# unc.plot(pol=2)
# plt.show()
#   
#   
# files = ['9.35GHZHornProbe+AUTpol0.csv','9.35GHZHornProbe+AUTpol90.csv'] ### earlier try 1
# unc = SNFData(J, measDist, folder, files, name = 'ProbeCoefficientsMeasurement')
# unc.plot(pol=2)
# plt.show()
#===============================================================================

#===============================================================================
# ## more pts and slow
# files = ['9.35GHzArray+moreuncloakedinmid.csv']
# munc = SNFData(J, measDist, folder, files, name = 'More Uncloaked')
# munc.plot(pol=2, freq = 1)
# plt.show()
#===============================================================================

#===============================================================================
# files = ['9.35GHZArrayActual.csv','9.35GHZArrayActual2.csv'] ### earlier try 1
# bare = SNFData(J, measDist, folder, files, name = 'SlotArrayBare')
# bare.plot(pol=2)
# plt.show()
# bare.calcTsmn()
# bare.plotFarField()
# plt.show()
#===============================================================================

###
##newer runs, .8 degree spacing
###
#===============================================================================
# files = ['manyCloakeds.8spacingPol0.csv','manyCloakeds.8spacingPol90try2.csv']
# manyCl = SNFData(J, measDist, folder, files, name = 'Many Cloakeds')
# manyCl.plot(pol=2)
# plt.show()
# manyCl.calcTsmnNoProbeCorrection()
# manyCl.plotFarField()
# plt.show()
#   
# files = ['manyUncloakeds.8spacingPol0.csv','manyUncloakeds.8spacingPol90.csv'] ### recent many uncloakeds run
# manyUncl = SNFData(J, measDist, folder, files, name = 'Many Uncloakeds')
# #manyUncl.plot(pol=2)
# #plt.show()
# manyUncl.calcTsmnNoProbeCorrection()
# manyUncl.plotFarField()
# plt.show()
#===============================================================================
 
#===============================================================================
# files = ['slotArray.8spacingPol90.csv','slotArray.8spacingPol0.csv']
# bareSlotAdjust= []
# bareSlotAdjust.append(np.array([[-50.6, -34.4, -16, -10.6, 11.9, 16.8, 37.9, 55.5], [-1.5, -1.6, -1.5, -1.5, 5, 6.5, 4.4, 5]])) ## pp=0
# bareSlotAdjust.append(np.array([1])) ## pp=90
#  
# slotBare = SNFData(J, measDist, folder, files, name = 'Bare Slot Array', phiAdjust = bareSlotAdjust)
# slotBare.plot(pol=2, phase=0)
# plt.show()
# #slotBare.calcTsmnNoProbeCorrection()
# #plotQs(slotBare.Ts)
# #slotBare.plotFarField()
# #plt.show()
# slotBare.calcTsmn()
# #plotQs(slotBare.Ts)
# #plotQs(slotBare.ProbeRs)
# slotBare.plotFarField(colourbar=True)
# plt.show()
# #plotVsSlotPlanes([slotBare])
#===============================================================================

    
#===============================================================================
# files = ['9.35GHzArray+0.2lambdaspaceduncloakedsPol90.csv', '9.35GHzArray+0.2lambdaspaceduncloakedsPol0.csv'] ## slot array with evenly .2lambda at 1.045 GHz-spaced uncloaked antennas on top. first positioned in mid
# slotPlusUncloaked = SNFData(J, measDist, folder, files, name = r'Slot+Uncloakeds')
# slotPlusUncloaked.plot(pol=2)
# plt.show()
# slotPlusUncloaked.calcTsmn()
# slotPlusUncloaked.plotFarField()
# plt.show()
#===============================================================================
 
#===============================================================================
# files = ['9.35GHzArray+0.2lambdaspacedcloakedsLaterTryPol0.csv'] ## slot array with evenly .2lambda at 1.045 GHz-spaced uncloaked antennas on top. first positioned in mid
# slotPlusCloaked = SNFData(J, measDist, folder, files, name = r'$.2\lambda$-sp. Cloakeds')
# slotPlusCloaked.plot(pol=0)
# plt.show()
#===============================================================================
###

### Yet newer runs, after the motor was fixed.
### Unfortunately, the rotation seems to slowly drift through the measurement even more than before - ~30 degree rotation after theta=0? Seems fairly non-drifting after that, for some reason...
### Seems like no matter what I do, it ends up having drifted 50 degrees...

#===============================================================================
# files = ['2025/9.35GHzSlotArrayFixedMotorPol90.csv', '2025/9.35GHzSlotArrayFixedMotorPol0.csv'] # these are bad, use old ones instead
# #files = ['slotArray.8spacingPol90.csv','slotArray.8spacingPol0.csv'] ## old one for comparison
# slotBare = SNFData(J, measDist, folder, files, name = 'Bare Slot Array')
# #ExportDataForParaview(slotBare)
# slotBare.plot(pol=0, phase=0)
# plt.show()
#===============================================================================

#files = ['2025/9.35GHzSlotArray+ManyCloakedsFixedMotorAllAtOnceTryPol0.csv', '2025/9.35GHzSlotArray+ManyCloakedsFixedMotorAllAtOnceTryPol90.csv'] ## roll drifts 70 deg
#files = ['2025/9.35GHzSlotArray+ManyCloakedsFixedMotorTryReverseThetasPol90.csv'] ## roll drifts 65 deg
files = ['2025/9.35GHzSlotArray+ManyCloakedsFixedMotortheta-100to0phito240Pol0.csv', '2025/9.35GHzSlotArray+ManyCloakedsFixedMotortheta-100to0phi-180Pol0.csv', '2025/9.35GHzSlotArray+ManyCloakedsFixedMotortheta-100to0Pol90.csv', '2025/9.35GHzSlotArray+ManyCloakedsFixedMotortheta-100to0phi-180Pol90.csv'] ## ended 15/5 degrees drifted (pol90), 12 degrees drifted (pol0)
#files = ['2025/9.35GHzSlotArray+ManyCloakedsFixedMotorAllAtOnceTry2ndHalfPol0.csv']
#files = ['slotArray.8spacingPol90.csv','slotArray.8spacingPol0.csv'] ## old one for comparison
slotBare = SNFData(J, measDist, folder, files, name = 'Slot Array+Many Cloakeds')
slotBare.plot(pol=1, phase=0)
plt.show()


print('DRH50 Horn probe info:')
lamb = c/nu
D = 0.06 ## diameter of probe-enclosing sphere (high guess from datasheet)
ff10G = 2*D**2/lamb
print('far-field dist: {:.5e}'.format(ff10G))
a = D/2 ## radius of array-enclosing sphere
k = 2*pi*nu/c ## wavenumber
N = int(np.ceil(k*a + 10)) ## number of modes to use
J = 2*N*(N+2) ## total number of spherical mode coefficients, s = 1 or 2, n up to N, then m from -n to n
print('N = '+str(N))
print('J = '+str(J))

### probe measurements
files = ['probeToProbeSlowPol0try2.csv','probeToProbeSlowPol90.csv'] # ['probeToProbeSlowPol0.csv','probeToProbeSlowPol90.csv']
DRH50HornProbeAdjust= []
DRH50HornProbeAdjust.append(np.array([[-102, -90, -70, -40, -23, -10, 0, 5, 11, 20, 27, 41.8, 48, 58, 72, 90, 102], [-5.4, -5, -5.5, -6, -5, -4, -3.5, -3, -3.4, -3, -2, -1.8, -1.2, -.6, -.5, -1, 0]])) ## pp=0, adjust to have 0 occur at phi=90
DRH50HornProbeAdjust.append(np.array([[-102, -90, -75, -42, -35, -25, -12, 0, 16, 23, 35.5, 44.7, 72, 90, 102], [4, 5.2, 6, 6, 5.6, 5.6, 6, 7.3, 8.6, 8.2, 9, 10, 12.5, 12.5, 12.5]])) ## pp=90, adjust for 0 at phi = 180
p2p = SNFData(J, measDist, folder, files, name = 'ProbesMeas', phiAdjust = DRH50HornProbeAdjust)
p2p.plot(pol=2, phase=0)
plt.show()
p2p.calcProbeCoefficients()
p2p.calcTsmnNoProbeCorrection()
plotQs(p2p.ProbeRs)
p2p.plotFarField(pol=2)
plt.show()
p2p.calcTsmn()
#plotQs(p2p.Ts)
p2p.plotFarField(pol=2)
plt.show()



###
## 1.045 GHz antenna runs:
###
nu = 1.045e9
lamb = c/nu
k = 2*pi/lamb
d = lamb*0.4
a = 0.72/2 ## radius of array-enclosing sphere - using entire slot array... could probably go a bit smaller
N = int(np.ceil(k*a + 10)) ## number of modes to use
J = 2*N*(N+2)

#===============================================================================
# ### full more-than-half sphere measurements:
# files = ['1.045GHzUpperMiddleUncloaked.8spacingPol90.csv','1.045GHzUpperMiddleUncloaked.8spacingPol0.csv'] 
# upperMidUncl = SNFData(J, measDist, folder, files, name = 'UM Uncl Full')
# #upperMidUncl.plot(pol=2, freq = 0)
# #plt.show()
# upperMidUncl.calcTsmnNoProbeCorrection(freq = 0)
# upperMidUncl.plotFarField(freq = 0)
# plt.show()
#===============================================================================

#===============================================================================
# files = ['1.045GHzCloakedUMPol0.csv','1.045GHzCloakedUMPol90.csv'] ### need to redo these
# upperMidCl = SNFData(J, measDist, folder, files, name = 'UM Cloak Full')
# upperMidCl.plot(pol=1, freq = 0)
# plt.show()
# upperMidCl.calcTsmnNoProbeCorrection(freq = 0)
# upperMidCl.plotFarField(freq = 0)
# plt.show()
#===============================================================================

#===============================================================================
# ### cut measurements: (UNCLOAKED)
# phaseFactors = []
# files = ['1.045GHzCut1UMPol0.csv','1.045GHzCut2UMPol0.csv','1.045GHzCutsUMPol90.csv']  ## upper-side (the side where the two foam blocks are glued together), in-between
# UMUncl = SNFData(J, measDist, folder, files, name = 'UM Uncl')
# phaseFactors.append(-k*d)
# #UMUncl.plot(freq=0)
# #plt.show()
# 
# files = ['1.045GHzCutsUCPol0.csv','1.045GHzCutsUCPol90.csv']  ## upper-side, placed by the centre of the slot array
# UCUncl = SNFData(J, measDist, folder, files, name = 'UC Uncl')
# phaseFactors.append(0)
# 
# files = ['1.045GHzCutsUEPol0.csv','1.045GHzCutsUEPol90.csv']  ## upper-side, placed by the edge of the slot array
# UEUncl = SNFData(J, measDist, folder, files, name = 'UE Uncl')
# phaseFactors.append(-k*d*1.95)
# 
# files = ['1.045GHzCutsLMPol0.csv','1.045GHzCutsLMPol90.csv']  ## upper-side (the side where the two foam blocks are glued together), in-between
# LMUncl = SNFData(J, measDist, folder, files, name = 'LM Uncl')
# phaseFactors.append(-k*d)
# 
# files = ['1.045GHzCutsLCPol0.csv','1.045GHzCutsLCPol90.csv']  ## upper-side, placed by the centre of the slot array
# LCUncl = SNFData(J, measDist, folder, files, name = 'LC Uncl')
# phaseFactors.append(0)
# 
# files = ['1.045GHzCutsLEPol0.csv','1.045GHzCutsLEPol90.csv']  ## upper-side, placed by the edge of the slot array
# LEUncl = SNFData(J, measDist, folder, files, name = 'LE Uncl')
# phaseFactors.append(-k*d*1.95)
# 
# datas = [UMUncl,UEUncl,UCUncl,LMUncl,LEUncl,LCUncl]
# angle = (0,0)
# print('pre-optim: ',phaseFactors)
# phaseFactors = scipy.optimize.minimize(calcBroadside, [0,0,0,0,0,0], method='Nelder-Mead', args=(datas, angle))#bounds = [(0,2*pi),(0,2*pi),(0,2*pi),(0,2*pi),(0,2*pi),(0,2*pi)])
# print('optim: ',phaseFactors)
# phaseFactors = phaseFactors.x
# print('optim: ',phaseFactors)
# plotPlanes(datas,[0],phaseFactors)
# plt.show()
#===============================================================================

files = ['1.045GHzCloakedUMtry2Pol90.csv','1.045GHzCloakedUMtry2Pol0.csv']
slots = SNFData(J, measDist, folder, files, name = 'UM Cloaked Full')
slots.plot(pol=2, freq=0)
plt.show()

### cut measurements: (CLOAKED)
phaseFactors = []
files = ['1.045GHzCloakedCutsUMPol0.csv','1.045GHzCloakedCutsUMPol90.csv']  ## upper-side (the side where the two foam blocks are glued together), in-between
UMUncl = SNFData(J, measDist, folder, files, name = 'UM Cloak')
phaseFactors.append(-k*d)
#UMUncl.plot(freq=0)
#plt.show()
 
files = ['1.045GHzCloakedCutsUCPol0.csv','1.045GHzCloakedCutsUCPol90.csv']  ## upper-side, placed by the centre of the slot array
UCUncl = SNFData(J, measDist, folder, files, name = 'UC Cloak')
phaseFactors.append(0)
 
files = ['1.045GHzCloakedCutsUEPol0.csv','1.045GHzCloakedCutsUEPol90.csv']  ## upper-side, placed by the edge of the slot array
UEUncl = SNFData(J, measDist, folder, files, name = 'UE Cloak')
phaseFactors.append(-k*d*1.95)
 
files = ['1.045GHzCloakedCutsLMPol0.csv','1.045GHzCloakedCutsLMPol90.csv']  ## upper-side (the side where the two foam blocks are glued together), in-between
LMUncl = SNFData(J, measDist, folder, files, name = 'LM Cloak')
phaseFactors.append(-k*d)
 
files = ['1.045GHzCloakedCutsLCPol0.csv','1.045GHzCloakedCutsLCPol90.csv']  ## upper-side, placed by the centre of the slot array
LCUncl = SNFData(J, measDist, folder, files, name = 'LC Cloak')
phaseFactors.append(0)
 
files = ['1.045GHzCloakedCutsLEPol0.csv','1.045GHzCloakedCutsLEPol90.csv']  ## upper-side, placed by the edge of the slot array
LEUncl = SNFData(J, measDist, folder, files, name = 'LE Cloak')
phaseFactors.append(-k*d*1.95)
 
datas = [UMUncl,UEUncl,UCUncl,LMUncl,LEUncl,LCUncl]
angle = (0,0)
print('pre-optim: ',phaseFactors)
phaseFactors = scipy.optimize.minimize(calcBroadside, [0,0,0,0,0,0], method='Nelder-Mead', args=(datas, angle))#, bounds = [(0,2*pi),(0,2*pi),(0,2*pi),(0,2*pi),(0,2*pi),(0,2*pi)])
print('optim: ',phaseFactors)
phaseFactors = phaseFactors.x
print('optim: ',phaseFactors)
plotPlanes(datas,[0],phaseFactors)
plt.show()

endTime = timer()
print('Finished in '+str(endTime-startTime)+' s')
pass

### REFERENCES

#[1]: Theory and Practice of Modern Antenna Range Measurements 2nd Expanded Edition, Volume 2 (2020)

#[2] https://mathworld.wolfram.com/SphericalHankelFunctionoftheSecondKind.html

#[3]: Spherical near-field antenna measurements, Hansen (1988)

#[4]: Fully Probe-Corrected Near-Field Far-Field Transformations With Unknown Probe Antennas, Paulus (2023)
