'''
Created on 11 okt. 2021

@author: al8032pa
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import os

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
    cl = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Cloaked Antenna morepts.txt'
    uncl = 'C:/Users/al8032pa/Work Folders/Documents/CSTstuff/Paper2 Uncloaked Dipole morepts.txt'
    
    unclData = np.transpose(np.loadtxt(uncl, delimiter = ' ', skiprows = 3))
    clData = np.transpose(np.loadtxt(cl, delimiter = ' ', skiprows = 3))
    plt.axvline(x=9.35, color="gray", linewidth=2, linestyle = '--')
    plt.plot(clData[0], clData[1] - unclData[1], linewidth = 2, color = 'tab:orange', alpha = 0.8, label='Cloaked Dipole')
    #plt.ylabel(r'$\sigma_{\text{ecs,cloaked}}/\sigma_{\text{ecs,uncloaked}}$ [dB]')
    plt.ylabel(r'$\frac{\sigma_{\mathrm{ext,cloaked}}}{\sigma_{\mathrm{ext,uncloaked}}}$ [dB]')
    plt.xlabel('Frequency [GHz]')
    plt.title('Cloaked vs Uncloaked Antenna Cross-section')
    #plt.axhline(y=-10, color="gray", linewidth=2, linestyle = '--')
    
    plt.axhline(y=0, color="tab:green", linewidth=2, label='Uncloaked Dipole')
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
    plt.ylabel('Directivity [dBi]')
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