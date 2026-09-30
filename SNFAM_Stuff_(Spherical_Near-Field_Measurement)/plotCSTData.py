'''
Created on 11 okt. 2021

@author: al8032pa
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import os
import matplotlib.lines as mlines

def plotLines():
    print('CST Plotting start.')
    clFN = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/secondfinal_try1_optimum.txt'

    unclFN = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/periodicuncloaked2.txt'
    
    unclData = np.transpose(np.loadtxt(unclFN, delimiter = ' ', skiprows = 630))
    #print(unclData)
    clData = np.transpose(np.loadtxt(clFN, delimiter = ' ', skiprows = 1675))
    
    plt.plot(clData[0], clData[1] + 10*np.log10(.005) - unclData[1] - 10*np.log10(.12), label = 'Second Final Design [CST]', linewidth = 2, color = 'tab:green', alpha = 0.8)
    print('CST Plotting Done.')
    
def plotLinesPaper2(normalize=True):
    print('CST Plotting start.')
    
  #=============================================================================
  #   ###
  #   ### Calculated with the time-domain solver
  #   ## with port
  #   cl = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Cloaked Antenna morepts.txt', delimiter = ' ', skiprows = 3)) ## RCS/m^2
  #   clA = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Cloaked Dipole ACS.txt', skiprows = 3)) ## RCS/mm^2
  #   clR = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Cloaked Dipole RCS.txt', skiprows = 3)) ## RCS/mm^2
  #   uncl = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Uncloaked Dipole morepts.txt', delimiter = ' ', skiprows = 3)) ## RCS/m^2
  #   unclA = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Uncloaked Dipole ACS.txt', skiprows = 3)) ## RCS/mm^2
  #   unclR = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Uncloaked Dipole RCS.txt', skiprows = 3)) ## RCS/mm^2
  # 
  #   ## without port
  #   clnp = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Cloaked Dipole noport ECS.txt', skiprows = 3)) ## RCS/m^2
  #   clAnp = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Cloaked Dipole noport ACS.txt', skiprows = 3)) ## RCS/mm^2
  #   clRnp = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Cloaked Dipole noport RCS.txt', skiprows = 3)) ## RCS/mm^2
  #   unclnp = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Uncloaked Dipole noport ECS.txt', skiprows = 3)) ## RCS/m^2
  #   unclAnp = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Uncloaked Dipole noport ACS.txt', skiprows = 3)) ## RCS/mm^2
  #   unclRnp = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Uncloaked Dipole noport RCS.txt', skiprows = 3)) ## RCS/mm^2
  #=============================================================================
    
    ###
    ### Calculated with the frequency-domain solver
    ## with port
    cl = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Cloaked Dipole ECS fd.txt', skiprows = 3)) ## RCS/m^2
    clA = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Cloaked Dipole ACS fd.txt', skiprows = 3)) ## RCS/mm^2
    clR = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Cloaked Dipole RCS fd.txt', skiprows = 3)) ## RCS/mm^2
    uncl = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Uncloaked Dipole ECS fd.txt', skiprows = 3)) ## RCS/m^2
    unclA = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Uncloaked Dipole ACS fd.txt', skiprows = 3)) ## RCS/mm^2
    unclR = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Uncloaked Dipole RCS fd.txt', skiprows = 3)) ## RCS/mm^2
  
    ## without port
    clnp = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Cloaked Dipole noport ECS fd.txt', skiprows = 3)) ## RCS/m^2
    clAnp = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Cloaked Dipole noport ACS fd.txt', skiprows = 3)) ## RCS/mm^2
    clRnp = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Cloaked Dipole noport RCS fd.txt', skiprows = 3)) ## RCS/mm^2
    unclnp = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Uncloaked Dipole noport ECS fd.txt', skiprows = 3)) ## RCS/m^2
    unclAnp = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Uncloaked Dipole noport ACS fd.txt', skiprows = 3)) ## RCS/mm^2
    unclRnp = np.transpose(np.loadtxt('C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Uncloaked Dipole noport RCS fd.txt', skiprows = 3)) ## RCS/mm^2
    
    ### normalized ECS plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.axvline(x=9.35, color="gray", linewidth=2, linestyle = '--')
    plt.plot(cl[0], cl[1] - uncl[1], linewidth = 2, color = 'tab:orange', alpha = 0.8, label='Cloaked Dipole')
    plt.plot(cl[0], clnp[1] - unclnp[1], linewidth = 2, color = 'tab:orange', alpha = 0.8, linestyle=':')
    plt.axhline(y=0, color="tab:green", linewidth=2, label='Uncloaked Dipole', alpha = 0.8)
    
    ##first legend
    first_legend = ax1.legend(framealpha=0.5, loc = 'lower left')
    ##second legend to distinguish between dashed and regular lines (phi- and theta- pols)
    handleds = []
    line1 = mlines.Line2D([], [], color='black', linestyle='-', linewidth=2, label=r'port termination') ##fake lines to create second legend elements
    line2 = mlines.Line2D([], [], color='black', linestyle=':', linewidth=2, label=r'no port termination') ##fake lines to create second legend elements
    handleds.append(line1)
    handleds.append(line2)
        
    second_legend = ax1.legend(handles=handleds, loc='lower right', framealpha=0.5) #best upper-right
    ax1.add_artist(first_legend) 
    
    plt.xlabel('Frequency [GHz]')
    plt.title('Cloaked vs Uncloaked Dipole Cross-section Reduction')
    plt.ylabel(r'$\sigma_{\mathrm{ext}}/\sigma_{\mathrm{ext}}^{\mathrm{uncloaked}}$ [dB]')
    plt.xlim(5, 15)
    plt.tight_layout()
    plt.grid()
    
    
    ### un-normalized ACS/RCS plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.axvline(x=9.35, color="gray", linewidth=2, linestyle = '--')
    
    plt.plot(clA[0], 10*np.log10(clA[1]*(1e-3)**2 + 1e-20), linewidth = 2, color = 'tab:orange', alpha = 0.8, label=r'$\sigma_\mathrm{abs}$')
    plt.plot(clAnp[0], 10*np.log10(clAnp[1]*(1e-3)**2 + 1e-20), linewidth = 2, color = 'tab:green', alpha = 0.8, label=r'$\sigma_\mathrm{abs}$, no port')
    plt.plot(clA[0], 10*np.log10(clR[1]*(1e-3)**2 + 1e-20), linewidth = 2, color = 'tab:brown', alpha = 0.8, label=r'$\sigma_\mathrm{sca}$')
    plt.plot(clAnp[0], 10*np.log10(clRnp[1]*(1e-3)**2 + 1e-20), linewidth = 2, color = 'tab:blue', alpha = 0.8, label=r'$\sigma_\mathrm{sca}$, no port')
    
    plt.plot(unclA[0], 10*np.log10(unclA[1]*(1e-3)**2 + 1e-20), linewidth = 2, color = 'tab:orange', alpha = 0.8, linestyle=':')
    plt.plot(unclAnp[0], 10*np.log10(unclAnp[1]*(1e-3)**2 + 1e-20), linewidth = 2, color = 'tab:green', alpha = 0.8, linestyle=':')
    plt.plot(unclA[0], 10*np.log10(unclR[1]*(1e-3)**2 + 1e-20), linewidth = 2, color = 'tab:brown', alpha = 0.8, linestyle=':')
    plt.plot(unclAnp[0], 10*np.log10(unclRnp[1]*(1e-3)**2 + 1e-20), linewidth = 2, color = 'tab:blue', alpha = 0.8, linestyle=':')
    plt.ylim(-68, -28)
    
    ##first legend
    first_legend = ax1.legend(framealpha=0.5, ncol=2, loc = 'lower left')
    ##second legend to distinguish between dashed and regular lines (phi- and theta- pols)
    handleds = []
    line1 = mlines.Line2D([], [], color='black', linestyle='-', linewidth=2, label=r'Cloaked') ##fake lines to create second legend elements
    line2 = mlines.Line2D([], [], color='black', linestyle=':', linewidth=2, label=r'Uncloaked') ##fake lines to create second legend elements
    handleds.append(line1)
    handleds.append(line2)
        
    second_legend = ax1.legend(handles=handleds, loc='lower right', framealpha=0.5) #best upper-right
    ax1.add_artist(first_legend) 
    
    plt.xlabel('Frequency [GHz]')
    plt.title('Cloaked vs Uncloaked Dipole Cross-section')
    plt.ylabel(r'Cross-sections [dBsm]')
    plt.xlim(5, 15)
    plt.tight_layout()
    plt.grid()
    
    
    
    fig = plt.figure()
    plt.axvline(x=9.35, color="gray", linewidth=2, linestyle = '--')
    
    #plt.plot(cl[0], cl[1] - uncl[1], linewidth = 2, color = 'tab:orange', alpha = 0.8, label='Cloaked Dipole')
    
    plt.plot(cl[0], cl[1], linewidth = 2, color = 'tab:orange', alpha = 0.8, label=r'Cloaked Dipole $\sigma_\mathrm{ext}$')
    plt.plot(clA[0], 10*np.log10(clA[1]*(1e-3)**2), linewidth = 2, color = 'tab:red', alpha = 0.8, label=r'Cloaked Dipole $\sigma_\mathrm{abs}$')
    plt.plot(clA[0], 10*np.log10(clR[1]*(1e-3)**2), linewidth = 2, color = 'tab:green', alpha = 0.8, label=r'Cloaked Dipole $\sigma_\mathrm{sca}$')
    
    plt.plot(uncl[0], uncl[1], linewidth = 2, color = 'tab:orange', alpha = 0.8, label=r'Uncloaked Dipole $\sigma_\mathrm{ext}$', linestyle=':')
    plt.plot(unclA[0], 10*np.log10(unclA[1]*(1e-3)**2), linewidth = 2, color = 'tab:red', alpha = 0.8, label=r'Uncloaked Dipole $\sigma_\mathrm{abs}$', linestyle=':')
    plt.plot(unclA[0], 10*np.log10(unclR[1]*(1e-3)**2), linewidth = 2, color = 'tab:green', alpha = 0.8, label=r'Uncloaked Dipole $\sigma_\mathrm{sca}$', linestyle=':')
    
    
    
    #plt.ylabel(r'$\sigma_{\text{ecs,cloaked}}/\sigma_{\text{ecs,uncloaked}}$ [dB]')
    plt.ylabel(r'$\frac{\sigma_{\mathrm{ext,cloaked}}}{\sigma_{\mathrm{ext,uncloaked}}}$ [dB]')
    plt.xlabel('Frequency [GHz]')
    plt.title('Cloaked vs Uncloaked Antenna Cross-section')
    #plt.axhline(y=-10, color="gray", linewidth=2, linestyle = '--')
    
    #plt.axhline(y=0, color="tab:green", linewidth=2, label='Uncloaked Dipole')
    plt.legend()
    plt.ylabel(r'$\sigma_{\mathrm{ext}}/\sigma_{\mathrm{ext}}^{\mathrm{uncloaked}}$ [dB]')
    
    plt.xlim(5, 15)
    plt.tight_layout()
    plt.grid()
    
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    
    #===========================================================================
    # plt.title('V-Plane Farfield Patterns')
    # slotBare = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Bare Array + cos taper.txt'
    # slotCl = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Bare Array+24deg Cloakeds + cos taper.txt'
    # slotUncl = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Bare Array+24deg Uncloakeds + cos taper.txt'
    # 
    # #===========================================================================
    # # plt.title('H-Plane Farfield Patterns')
    # # slotBare = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Bare Array + cos taper cut2.txt'
    # # slotCl = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Bare Array+24deg Cloakeds + cos taper cut2.txt'
    # # slotUncl = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Bare Array+24deg Uncloakeds + cos taper cut2.txt'
    # #===========================================================================
    #===========================================================================
    
    #===========================================================================
    # plt.title('V-Plane Farfield Patterns')
    # slotBare = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Bare Array.txt'
    # slotCl = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Bare Array+24deg Cloakeds.txt'
    # slotUncl = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Bare Array+24deg Uncloakeds.txt'
    #===========================================================================
    
    plt.title('H-Plane Farfield Patterns')
    slotBare = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Bare Array cut2.txt'
    slotCl = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Bare Array+24deg Cloakeds cut2.txt'
    slotUncl = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Bare Array+24deg Uncloakeds cut2.txt'
    
    slotBareData = np.transpose(np.loadtxt(slotBare, skiprows = 2))
    slotClData = np.transpose(np.loadtxt(slotCl, skiprows = 2))
    slotUnclData = np.transpose(np.loadtxt(slotUncl, skiprows = 2))
    
    if(normalize):
        max = np.max([np.max(slotBareData[2]), np.max(slotClData[2]), np.max(slotUnclData[2])])
        slotBareData[2] = slotBareData[2] - max
        slotClData[2] = slotClData[2] - max
        slotUnclData[2] = slotUnclData[2] - max
    
    
    ax2.plot(slotBareData[0], slotBareData[2], label = r'Bare Slot Array', linewidth = 1.5, linestyle = '-')
    ax2.plot(slotBareData[0], slotClData[2], label = r'with Cloaked', linewidth = 2, linestyle = ':')
    ax2.plot(slotBareData[0], slotUnclData[2], label = r'with Uncloaked', linewidth = 2, linestyle = '--', alpha = 0.8)
    plt.ylabel('Normalized Directivity [dB]')
    plt.xlabel(r'$\theta$ [degrees]')
    plt.xlim(-85,85)
    if(normalize):
        plt.ylim(-65,1)
    else:
        plt.ylim(-29,38)
    ax2.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()
    print('CST Plotting Done.')

if __name__ == '__main__':
    
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
    plt.rc('legend', fontsize=12)    # legend fontsize
    plt.rc('figure', titlesize=20)  # fontsize of the figure title
    plt.rc('text', usetex=True)
    #plotLines()
    #plt.show()
    plotLinesPaper2()
    pass